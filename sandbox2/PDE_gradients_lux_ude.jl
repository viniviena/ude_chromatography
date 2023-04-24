using Pkg
Pkg.activate(".")
Pkg.instantiate()

#Importing ODE, plot and MAT libraries
using OrdinaryDiffEq
using DiffEqFlux
using DiffEqCallbacks
#using DifferentialEquations
using Flux
using Lux
using Plots
using MAT
using DelimitedFiles
using SciMLSensitivity
import Random
using PreallocationTools
using ForwardDiff, Zygote
using ReverseDiff
using Flux
using StatsBase

# Script with auxiliary functions
include("utils.jl")

#----------- Building OCFEM (orthogonal collocation on finite element method)
#for z discretization with cubic hermite polynomials-------------

n_elements = 42 # Number of finite elements
collocation_points = 2 #Collocation points
n_components = 1;  # 2 chemical species
n_phases = 2 #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ = 0.0e0
xₘₐₓ = 1.0e0 # z domain limits
h = (xₘₐₓ - xₘᵢₙ) / n_elements #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components) #make matrices for OCFEM

#Building mass matrix
MM = BitMatrix(Array(make_MM_2(n_elements, n_phases, n_components))) #make mass matrix


#-------- Defining PDE parameters------------


Qf = 5.0e-2
d = 0.5 
L = 2.0 
a = pi*d^2/4
epsilon = 0.5
u = Qf/(a*epsilon)
Pe = 21.095632695978704
Dax = u*L/Pe
cin = 5.5
k_transf = 0.22
k_iso  = 1.8
qmax = 55.54
q_test = qmax*k_iso*cin^1.0/(1.0 + k_iso*cin^1.0)


#Calculating the derivative matrices stencil
y_dy = Array(A * H^-1) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = Array(B * H^-1) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y


#--------Importing experimental data---------------
using DataInterpolations

c_exp_data = readdlm("train_data/traindata_improved_quad_lang_2min.csv", ',', Float64)


# -----Initializing Neural networks---------
import Random

# ----- Lux

rng = Random.default_rng()
Random.seed!(rng, 2)

rbf(x) = exp.(-(x.^2))

nn = Lux.Chain(
  Lux.Dense(2, 10, tanh_fast),
  Lux.Dense(10, 8, tanh_fast),
  Lux.Dense(8, 1)
)

#= nn = Lux.Chain(
  Lux.Dense(2, 22, tanh_fast),
  Lux.Dense(22, 1)
) =#

p_init, st = Lux.setup(rng, nn)


y0_cache = ones(Float64, n_variables)
c0 = 1e-3 #Avoid using 0.0 here 


# Function to build initial condition vector 
# Note that the values at boundaries are guesses as it is determined during initialization
function y_initial(y0_cache, c0)
    var0 = y0_cache[:]

    begin
    j = 0
    #Internal node equations
    cl_idx = 2 + j
    cu_idx = p_order + 2 * n_elements - 3 + j


    cbl_idx = j + 1
    cbu_idx = j + p_order + 2 * n_elements - 2

    #Liquid phase residual
    var0[cl_idx:cu_idx] = ones(cu_idx - cl_idx + 1) * c0

    #Boundary node equations
    var0[cbl_idx] = c0

    var0[cbu_idx] = c0

    end
   

    begin

    ql_idx2 = 1 * (p_order + 2 * n_elements - 2) + 2 + j - 1
    qu_idx2 = p_order + 2 * n_elements - 3 + 1 * (p_order + 2 * n_elements - 2) + j + 1

    #Solid phase residual
    var0[ql_idx2:qu_idx2] .= qmax*k_iso*c0^1.0/(1.0 + k_iso*c0^1.0)

    j = j + p_order + 2 * n_elements - 2
    end

    var0

end


y0 =  y_initial(y0_cache, c0) # Calling function to build initial condition vector


# building rhs function for DAE solver

mutable struct col_model_node1{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13}
n_variables::T1
n_elements::T2
p_order::T3
L::T4
h::T5
u::T6
y_dy::T7
y_dy2::T8
Pe::T9
epsilon::T10
c_in::T11
dy_du::T12
dy2_du::T13
end

#using TimerOutputs
using UnPack


dy_du = dy2_du = ones(Float64, n_variables)


function (f::col_model_node1)(yp, y, p, t)
   #Aliasing parameters

   @unpack n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
   Pe, epsilon, c_in, dy_du, dy2_du  = f 
   
   
   dy_du =  y_dy*y
   dy2_du = y_dy2*y

   
   j = 0
   #---------------------Mass Transfer and equilibrium -----------------

   c = (@view y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1]) 
   q_eq  = qmax*k_iso.*c.^1.0./(1.0 .+ k_iso.*c.^1.0)/q_test #Scaled solid equilibrium concentration

   q = (@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1])/q_test #scaling dependent variables
   x1x2 =  [q_eq q]'

   #-------------------------------mass balance -----------------

   begin
       #Internal node equations
       cl_idx = 2 + j
       cu_idx = p_order + 2 * n_elements - 3 + j

       ql_idx = 1 * (p_order + 2 * n_elements - 2) + 2 + j
       qu_idx = p_order + 2 * n_elements - 3 + 1 * (p_order + 2 * n_elements - 2) + j

       ql_idx2 = 1 * (p_order + 2 * n_elements - 2) + 2 + j - 1
       qu_idx2 = p_order + 2 * n_elements - 3 + 1 * (p_order + 2 * n_elements - 2) + j + 1

       cbl_idx = j + 1
       cbu_idx = j + p_order + 2 * n_elements - 2

       #Liquid phase residual
        
       yp[cl_idx:cu_idx] .= - (1 - epsilon) / epsilon * (@view nn(x1x2, p, st)[1][2:end - 1])  .- u*(@view dy_du[cl_idx:cu_idx]) / h / L  .+  Dax / (L^2) * (@view dy2_du[cl_idx:cu_idx]) / (h^2)


       #Solid phase residual
       yp[ql_idx2:qu_idx2] .= (@view nn(x1x2, p, st)[1][1:end])

       #Boundary node equations
       yp[cbl_idx] = Dax / L * dy_du[cbl_idx] / h - u * (y[cbl_idx] -  c_in)

       yp[cbu_idx] =  dy_du[cbu_idx] / h / L
   end
   nothing
end



# Building ODE problem
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, cin, dy_du, dy2_du);

f_node = ODEFunction(rhs, mass_matrix = MM)

#----- Solving ODE Problem with randomly initialized weights
y0 = y_initial(y0_cache, c0)

tspan = (first(c_exp_data[: , 1]), last(c_exp_data[: , 1])) 

prob_node = ODEProblem(f_node, y0, tspan, Lux.ComponentArray(p_init))

LinearAlgebra.BLAS.set_num_threads(1)

ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())

@time solution_other = solve(prob_node, FBDF(autodiff = false),
 saveat = c_exp_data[2, 1] - c_exp_data[1, 1]); #0.27 seconds after compiling

plot(solution_other.t, Array(solution_other)[Int(n_variables/2), :])
scatter!(c_exp_data[:, 1], c_exp_data[:, 2])


#--------- Training Neural Network ----------

tsave = c_exp_data[2:end, 1]

function predict(θ)
    # --------------------------Sensealg---------------------------------------------
    sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))

    #----------------------------Problem solution-------------------------------------
    abstol = reltol = 5e-7
    tspan = (0.0, maximum(c_exp_data[:, 1])) 

    prob_ = remake(prob_node; p = θ, tspan = tspan)
    
    s_new = Array(solve(prob_, FBDF(autodiff = false), abstol = abstol, reltol = reltol,
    saveat = tsave, sensealg = sensealg))

    #----------------------------Output---------------------------------------------

    (@view s_new[Int(n_variables / 2), 1:end])./cin 
end

#Setting up training data
data_train = c_exp_data[2:end, 2]/cin;

#You can set up weights for the observations here
cond1 = c_exp_data[2:end, 1] .> 50.
cond2 = c_exp_data[2:end, 1] .< 63.
is_bt = cond1 .& cond2
weights = ones(size(data_train))
weights[is_bt] .= 1.0

#Loss function
loss(θ) = sum(abs2, (data_train .- predict(θ)).*weights) #Using either abs or abs2 work well


# ..................testing gradients
θ = copy(Lux.ComponentArray(p_init))

using ReverseDiff

@time loss(θ)
@time predict(θ)
@time regularization(θ)
@time grad_reverse = Zygote.gradient(loss, θ)
@time grad_regularization = Zygote.gradient(regularization, θ)[1]


#------- Maximum Likelihood estimation
using Optimization

#adtype = Optimization.AutoReverseDiff(true)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

iter = 1
callback = function(p,l)
    global iter 
    println(l)
    println(iter)
    iter += 1
    l < 2.0e-3
end

opt = Flux.Optimiser(ADAM(0.05), ExpDecay(1.0, 0.985, 20))

@time results = Optimization.solve(optprob, opt, callback = callback, maxiters = 180)

optf2 = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob2 = Optimization.OptimizationProblem(optf2, results.u)

@time results_2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01), 
callback = callback, maxiters = 75, maxtime = 35*60, allow_f_increases = true)

aaa = predict(results_2.u)

loss(results_2.u)
mae = sqrt(Flux.mse(c_exp_data[2:end, 2], aaa[1:end]*cin))*100
println("MAE is $mae%")

scatter(c_exp_data[2:end, 1], c_exp_data[2:end, 2], label = " Experimental ", legend =:bottomright)
plot!(c_exp_data[2:end, 1], aaa[1:end]*cin, label = "neural UDE", legend=:bottomright)
scatter(c_exp_data[2:end, 2], aaa*cin, label = nothing)
plot!(0:0.5:6.0, 0:0.5:6.0, label = nothing)

plot(c_exp_data[2:end, 1], c_exp_data[2:end, 2] .- aaa*cin, marker = 'o')

writedlm("trained_models/best_improved_quad_10_8_neurons_42fe_lang_tanh_2min_5e-7_abs2.csv", results_2.u)

# ------ Plotting Residuals 
using KernelDensity, Distributions
error = c_exp_data[2:end, 2] .- aaa*cin
error_mean = mean(c_exp_data[2:end, 2] .- aaa*cin)
error_std = std(c_exp_data[2:end, 2] .- aaa*cin)
gauss = Distributions.Normal(error_mean, error_std)
samples_gauss = Distributions.pdf.(gauss, -0.2:0.001:0.2)
kde_samples = KernelDensity.kde(error)
plot(collect(kde_samples.x), kde_samples.density)
plot!(-0.2:0.001:0.2, samples_gauss)
plot!([error_mean], seriestype="vline")
 

#---------------- Test set performance
mutable struct col_model_node_test{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13}
    n_variables::T1
    n_elements::T2
    p_order::T3
    L::T4
    h::T5
    u::T6
    y_dy::T7
    y_dy2::T8
    Pe::T9
    epsilon::T10
    c_in::T11
    dy_du::T12
    dy2_du::T13
end
    
#using TimerOutputs
using UnPack
    
    
dy_du = dy2_du = ones(Float64, n_variables)
    
    
function (f::col_model_node_test)(yp, y, p, t)
    #Aliasing parameters

    @unpack n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
    Pe, epsilon, c_in, dy_du, dy2_du  = f 
    
    
    dy_du =  y_dy*y
    dy2_du = y_dy2*y

    
    j = 0
    #---------------------Mass Transfer and equilibrium -----------------

    c = (@view y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1]) #Scaling dependent variables
    #q_eq  = qmax*k_iso*c./(1.0 .+ k_iso.*c)/q_test
    #q_eq = 25.0*abs.(c).^0.6/q_test
    q_eq  = qmax*k_iso*abs.(c).^1.5./(1.0 .+ k_iso.*abs.(c).^1.5)/q_test

    q = ((@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1]) .- 0.0)./q_test #scaling dependent variables
    x1x2 =  [q_eq q]'

    #-------------------------------mass balance -----------------

    begin
        #Internal node equations
        cl_idx = 2 + j
        cu_idx = p_order + 2 * n_elements - 3 + j

        ql_idx = 1 * (p_order + 2 * n_elements - 2) + 2 + j
        qu_idx = p_order + 2 * n_elements - 3 + 1 * (p_order + 2 * n_elements - 2) + j

        ql_idx2 = 1 * (p_order + 2 * n_elements - 2) + 2 + j - 1
        qu_idx2 = p_order + 2 * n_elements - 3 + 1 * (p_order + 2 * n_elements - 2) + j + 1

        cbl_idx = j + 1
        cbu_idx = j + p_order + 2 * n_elements - 2

        #Liquid phase residual
        
        yp[cl_idx:cu_idx] .= -(1 - epsilon) / epsilon  * (@view nn(x1x2, p, st)[1][2:end - 1]) .- (@view dy_du[cl_idx:cu_idx]) / h / (L / u) .+ 1 / Pe * (@view dy2_du[cl_idx:cu_idx]) / (h^2) / (L / u)


        #Solid phase residual

        yp[ql_idx2:qu_idx2] .= (@view nn(x1x2, p, st)[1][1:end])

        #ex_[i](t)
        #Boundary node equations
        yp[cbl_idx] = dy_du[cbl_idx] / h .- Pe * (y[cbl_idx] .-  c_in(t))

        yp[cbu_idx] =  dy_du[cbu_idx] / h
    end
    nothing
end

using DataInterpolations

t_interp = [0.0:0.1:130.0; 130.0000001; 140.00:5.:250.0; 250.0000001; 260.0:5.0:500.]
c_interp = [fill(5.5, size(0.0:0.1:130., 1)); 0.75; fill(0.75, size(140.00:5.:250., 1)); 9.33;
 fill(9.33, size(260.0:5.0:500., 1))]

scatter(t_interp, c_interp)
c_in_t = LinearInterpolation(c_interp, t_interp)


rhs_test = col_model_node_test(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, c_in_t, dy_du, dy2_du);
f_node_test = ODEFunction(rhs_test, mass_matrix = MM)
y0 = y_initial(y0_cache, 5e-3)
tspan_test = (0.00e0, 400.00e0) 
prob_node_test = ODEProblem(f_node_test, y0, tspan_test, best_w)


@time solution_test = Array(solve(prob_node_test, FBDF(autodiff = false), 
abstol = 1e-6, reltol = 1e-6, tstops = [0.0, 130., 250.], saveat = 2.5e0)); #0.27 seconds after compiling

test_data = readdlm("testdata_improved_quad_sips_25min.csv", ',')
fig = scatter(test_data[:, 1], test_data[:, 2], label = "true")
plot!(fig, 2.5:2.5:400, solution_test[Int(n_variables/2), 2:end], linewidth = 2., label = "UDE learned")
plot!(fig, 1.0:1.0:450, solution_test[Int(n_variables/2) - 50, :], linewidth = 2., label = "UDE learned")
savefig("improved_kldf_test.pdf")

