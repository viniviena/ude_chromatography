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
using PGFPlots
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
#ρ_b = 2.001e-3/(a*L)
cin = 5.5
k_transf = 0.22
k_iso  = 1.8
qmax = 55.54
q_test = qmax*k_iso*cin^1.5/(1.0 + k_iso*cin^1.5)


#params_ode = [11.66, 9.13, 5.08, 5.11, kappaa, kappab, 163.0, 0.42, 11.64, 0.95]

function round_zeros(x)
    if abs(x) < 1e-42
        0.0e0
    else
        Float64(x)
end
end

#Calculating the derivative matrices stencil
y_dy = Array(A * H^-1) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = Array(B * H^-1) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y


#--------Importing experimental data---------------
using DataInterpolations

c_exp_data = readdlm("train_data/traindata_kldf_sips_2min.csv", ',', Float64)


# -----Initializing Neural networks---------
import Random

# ----- Lux

rng = Random.default_rng()
Random.seed!(rng, 11)


rbf(x) = exp.(-(x.^2))

nn = Lux.Chain(
  Lux.Dense(2, 20, tanh_fast),
  Lux.Dense(20, 1)
)

p_init, st = Lux.setup(rng, nn)


#--------------Flux
#= ann_node1 = FastChain(FastDense(2, 15, tanh), FastDense(15, 1)); #ANNₑ,₁
params1 = initial_params(ann_node1)


function my_nn(u, p)
    w1 = reshape((@view p[1:2*15]), 15, 2)
    b1 = @view p[2*15+1:3*15]
    w2 = reshape((@view p[3*15+1:3*15+1*15]), 1, 15)
    b2 = @view p[4*15+1:end]

    (w2 * (tanh.(w1 * u .+ b1*0.0) .+ b2*0.0))
end =#

# ----- Building the actual PDE model--------


y0_cache = ones(Float64, n_variables)
c0 = 1e-3


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
    var0[ql_idx2:qu_idx2] .= qmax*k_iso*c0^1.5/(1.0 + k_iso*c0^1.5)
    #var0[ql_idx2:qu_idx2] .= 25.0*c0.^0.6
    #var0[ql_idx2:qu_idx2] .= radial_surrogate.(c0)
    #var0[ql_idx2:qu_idx2] .= interpolator.(c0)

    j = j + p_order + 2 * n_elements - 2
    end

    var0

end


y0 =  y_initial(y0_cache, c0)


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

   c = (@view y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1]) #Scaling dependent variables
   q_eq  = qmax*k_iso.*abs.(c).^1.5./(1.0 .+ k_iso.*abs.(c).^1.5)/q_test
   #q_eq = 25.0*abs.(c).^0.6/q_test
   #q_eq = qmax*k_iso^(1/t)*p./(1.0 .+ k_iso*abs.(p).^t).^(1/t)*ρ_p  

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

       #(@view nn(x1x2, p, st)[1][2:end - 1])
       #Solid phase residual

       yp[ql_idx2:qu_idx2] .= (@view nn(x1x2, p, st)[1][1:end])

       #(@view nn(x1x2, p, st)[1][1:end])

       #ex_[i](t)
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

#----- non optimized prob
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
    abstol = reltol = 1e-6
    tspan = (0.0, maximum(c_exp_data[:, 1])) #TAVA ERRADOOO

    prob_ = remake(prob_node; p = θ, tspan = tspan)
    

    s_new = Array(solve(prob_, FBDF(autodiff = false), abstol = abstol, reltol = reltol,
    saveat = tsave, sensealg = sensealg))


    #----------------------------Output---------------------------------------------
    # The outputs are composed by the predictions of cᵢ (all times) and qᵢ (at injection times)

    (@view s_new[Int(n_variables / 2), 1:end])./cin 
    #Array(s_new[observed(simple_sys)[1].lhs][2:end])./cin
end

#Setting up training data
data_train = c_exp_data[2:end, 2]/cin;

cond1 = c_exp_data[2:end, 1] .> 50.
cond2 = c_exp_data[2:end, 1] .< 63.
is_bt = cond1 .& cond2
weights = ones(size(data_train))
weights[is_bt] .= 1.0

# Setting up loss function for using with galactic
loss(θ) = sum(abs, (data_train .- predict(θ)).*weights)


#= function regularization(θ)
    #Flux.Losses.mse(_ann1(c0_scaled, θ[1 + 2: 29 + 2]), θ[1])*(1/0.5^2) +
    #Flux.Losses.mse(_ann2(c0_scaled, θ[30 + 2: 30 + 28 + 2]), θ[2])*(1/0.5^2)
    dot((@view θ[1:end]), (@view θ[1:end]))*(1/100)
end =#

# ..................testing gradients
θ = copy(Lux.ComponentArray(p_init))

using ReverseDiff

@time loss(θ)
@time predict(θ)
@time regularization(θ)
@time grad_reverse = ReverseDiff.gradient(loss, θ)
@time grad_regularization = Zygote.gradient(regularization, θ)[1]


#------- MAP estimation
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
    l < 2.0e-1
end

opt = Flux.Optimiser(ADAM(0.05), ExpDecay(1.0, 0.985, 20))

@time results = Optimization.solve(optprob, opt, callback = callback, maxiters = 150)

optf2 = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob2 = Optimization.OptimizationProblem(optf2, results.u)

@time results_2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01), 
callback = callback, maxiters = 75, maxtime = 20*60, allow_f_increases = false)

aaa = predict(results.u)

loss(results.u)
mae = sqrt(Flux.mse(c_exp_data[2:end, 2], aaa[1:end]*cin))*100
println("MAE is $mae%")

scatter(c_exp_data[2:end, 1], c_exp_data[2:end, 2], label = " Experimental ", legend =:bottomright)
plot!(c_exp_data[2:end, 1], aaa[1:end]*cin, label = "neural UDE", legend=:bottomright)
scatter(c_exp_data[2:end, 2], aaa*cin, label = nothing)
plot!(0:0.5:6.0, 0:0.5:6.0, label = nothing)

plot(c_exp_data[2:end, 1], c_exp_data[2:end, 2] .- aaa*cin, marker = 'o')

writedlm("trained_models/best_improved_kldf_20_neurons_42fe_sips_tanh_2min_1e-6.csv", results.u)

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



#-------------- sparse regression

#----model loading
using DelimitedFiles
best_p = Float64.(readdlm("trained_models/best_kldf_20_neurons_42fe_sips_tanh_2min_1e-6.csv"))
best_w = deepcopy((Lux.ComponentArray(p_init)))
#best_w = deepcopy(results_2.u)
neurons = 20
best_w.layer_1.weight  .= reshape(best_p[1:neurons*2], neurons, 2)
best_w.layer_1.bias .= reshape(best_p[neurons*2 + 1:neurons*2 + neurons], neurons, 1)
best_w.layer_2.weight .= reshape(best_p[neurons*2 + neurons + 1: neurons*2 + neurons + neurons], 1, neurons)
best_w.layer_2.bias .= reshape(best_p[neurons*2 + neurons + neurons + 1:end], 1, 1)

#= best_w.layer_1.weight  .= reshape(best_p[1:20], 10, 2)
best_w.layer_1.bias .= reshape(best_p[21:21 + 9], 10, 1)
best_w.layer_2.weight .= reshape(best_p[21 + 9 + 1: 21 + 9 + 1 + 10*8 - 1], 8, 10)
best_w.layer_2.bias .= reshape(best_p[21 + 9 + 1 + 10*8: 21 + 9 + 1 + 10*8 + 7], 8, 1)
best_w.layer_3.weight .= reshape(best_p[21 + 9 + 1 + 10*8 + 7 + 1: 21 + 9 + 1 + 10*8 + 7 + 1 + 7], 1, 8)
best_w.layer_3.bias .= reshape(best_p[21 + 9 + 1 + 10*8 + 7 + 1 + 7 + 1:end], 1, 1)
 =#

y0 = y_initial(y0_cache, 1e-3)
prob_node22 = ODEProblem(f_node, y0, (0.0, 110.0), Lux.ComponentArray(best_w))

saveats = first(c_exp_data[:, 1]):mean(diff(c_exp_data[:, 1]))/10:last(c_exp_data[:, 1])
@time solution_optim = solve(prob_node22, FBDF(autodiff = false),
 abstol = 1e-6, reltol = 1e-6, saveat = saveats); #0.27 seconds after compiling

plot(solution_optim.t, Array(solution_optim)[Int(n_variables/2), :])

loss(best_w)

c_ = solution_optim[Int(n_variables/2), 1:end]
qeq_ = qmax*k_iso.*c_.^1.50./(1 .+ k_iso.*c_.^1.50)./q_test
#qeq_f = 25.0*c_.^0.6./q_test
q_ = Array(solution_optim)[Int(n_variables), 1:end]./q_test

learned_kinetics = nn([qeq_ q_]', best_w, st)[1]
PGFPlots.plot(solution_optim.t[1:end], learned_kinetics[:])

true_dqdt = readdlm("test_data/true_dqdt_kldf_sips_2min.csv", ',')


#----------------Desorption and extrapolation
mutable struct col_model_test{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13}
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
    
    
function (f::col_model_test)(yp, y, p, t)
   #Aliasing parameters

   @unpack n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
   Pe, epsilon, c_in, dy_du, dy2_du  = f 
   
   
   dy_du =  y_dy*y
   dy2_du = y_dy2*y

   
   j = 0
   #---------------------Mass Transfer and equilibrium -----------------

   c = (@view y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1]) #Scaling dependent variables
   q_eq  = qmax*k_iso.*abs.(c).^1.5./(1.0 .+ k_iso.*abs.(c).^1.5)/q_test
   #q_eq = 25.0*abs.(c).^0.6/q_test
   #q_eq = qmax*k_iso^(1/t)*p./(1.0 .+ k_iso*abs.(p).^t).^(1/t)*ρ_p  

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

       #(@view nn(x1x2, p, st)[1][2:end - 1])
       #Solid phase residual

       yp[ql_idx2:qu_idx2] .= (@view nn(x1x2, p, st)[1][1:end])

       #(@view nn(x1x2, p, st)[1][1:end])

       #ex_[i](t)
       #Boundary node equations
       yp[cbl_idx] = Dax / L * dy_du[cbl_idx] / h - u * (y[cbl_idx] -  c_in(t))

       yp[cbu_idx] =  dy_du[cbu_idx] / h / L
   end
   nothing
end

using DataInterpolations

t_interp_lang = [0.0:0.1:110.0; 110.0000001; 120.00:5.:250.0; 250.0000001; 260.0:5.0:500.]
c_interp_lang = [fill(5.5, size(0.0:0.1:110., 1)); 3.58; fill(3.58, size(120.00:5.:250., 1)); 7.33;
 fill(7.33, size(260.0:5.0:500., 1))]

t_interp_sips = [0.0:0.1:110.0; 110.0000001; 120.00:5.:250.0; 250.0000001; 260.0:5.0:500.]
c_interp_sips = [fill(5.5, size(0.0:0.1:110., 1)); 0.75; fill(0.75, size(120.00:5.:250., 1)); 9.33;
 fill(9.33, size(260.0:5.0:500., 1))]

scatter(t_interp_lang, c_interp_lang)
c_in_t = LinearInterpolation(c_interp_sips, t_interp_sips)

rhs_test = col_model_test(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, c_in_t, dy_du, dy2_du);
f_node_test = ODEFunction(rhs_test, mass_matrix = MM)
y0 = y_initial(y0_cache, 1e-3)
tspan_test = (0.00e0, 400.00e0)

prob_node_test = ODEProblem(f_node_test, y0, tspan_test, Lux.ComponentArray(best_w)) 
solution_test = solve(prob_node_test, FBDF(autodiff = false), 
abstol = 1e-7, reltol = 1e-7, tstops = [0.0, 110., 250.], saveat = 2.0e0);

test_data = readdlm("test_data/testdata_kldf_sips_2min.csv", ',')
test_rate = c_exp_data[2, 1] - c_exp_data[1, 1]

using PGFPlots

history = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(history, Axis([Plots.Linear(0.0:test_rate:110.0 |> collect, solution_test[Int(n_variables/2), 1:size(c_exp_data, 1)], mark = "none", style = "blue", legendentry = "UDE prediction - Train"),
            Plots.Linear(c_exp_data[1:end, 1], c_exp_data[1:end, 2], onlyMarks=true, style = "blue, mark = *, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Train"),  
            Plots.Linear(110.0 + test_rate:test_rate:400 |> collect, solution_test[Int(n_variables/2), size(c_exp_data, 1) + 1:end], mark = "none", style = "red!60, dashed", legendentry = "UDE prediction - Test"),
            Plots.Linear(test_data[size(c_exp_data, 1) + 1:end, 1], test_data[size(c_exp_data, 1) + 1:end, 2],onlyMarks=true, style = "red!60, mark = square*, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Test"),
            Plots.Linear([110., 110.], [0., 10.5], mark = "none", style = "black"),
            Plots.Node("Train data", 30, 7, style = "blue"),
            Plots.Node("Test data", 210, 7, style = "red!60")
],
        legendPos="south east", style = "grid = both, ytick = {0, 2, 4, 6, 8, 10}, xtick = {0, 40, 80,...,400}, legend style={nodes={scale=0.5, transform shape}}", xmin = 0, xmax = 400, ymin = 0, ymax = 10, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{c}\,\left[\textrm{mg}\,\textrm{L}^{-1}\right]", title = "Sips isotherm - LDF"))

save("plots/kldf_sips_history.pdf", history)       


history_error = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(history_error, Axis([Plots.Linear(0.0:test_rate:110.0 |> collect, solution_test[Int(n_variables/2), 1:size(c_exp_data, 1)] .- c_exp_data[1:end, 2], mark = "none", style = "blue", legendentry = "Prediction error - Train"),  
            Plots.Linear(110.0 + test_rate:test_rate:400 |> collect, solution_test[Int(n_variables/2), size(c_exp_data, 1) + 1:end] .- test_data[size(c_exp_data, 1) + 1:end, 2], mark = "none", style = "red!60, dashed", legendentry = "Prediction error - Test"),
            Plots.Linear([110., 110.], [-0.5, 0.5], mark = "none", style = "black"),
            Plots.Node("Train data", 30, 0.25, style = "blue"),
            Plots.Node("Test data", 210, 0.25, style = "red!60")
],
        legendPos="south east", style = "grid = both, ytick = {-0.4,-0.2, 0.0, 0.2, 0.4}, xtick = {0, 40, 80,...,400},  legend style={nodes={scale=0.5, transform shape}}", xmin = 0, xmax = 400, ymin = -0.4, ymax = 0.4, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{\varepsilon}\,\left[\textrm{mg}\,\textrm{L}^{-1}\right]", title = "Sips isotherm - LDF"))

save("plots/kldf_sips_history_error.pdf", history_error)



using KernelDensity, Distributions
error = c_exp_data[1:131, 2] .- solution_test[Int(n_variables/2), 1:131]
error_mean = mean(c_exp_data[1:131, 2] .- solution_test[Int(n_variables/2), 1:131])
error_std = std(c_exp_data[1:131, 2] .- solution_test[Int(n_variables/2), 1:131])
gauss = Distributions.Normal(error_mean, error_std)
samples_gauss = Distributions.pdf.(gauss, -0.2:0.001:0.2)
kde_samples = KernelDensity.kde(error)
plot(collect(kde_samples.x), kde_samples.density)
plot!([error_mean], seriestype="vline")

error_test = solution_test[Int(n_variables/2), 132:end] .- test_data[131:end, 2]
kde_test = KernelDensity.kde(error_test)
error_test_mean = mean(error_test)
error_test_std = std(error_test)

histogram = GroupPlot(2, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(histogram, Axis([Plots.Linear(collect(kde_samples.x), kde_samples.density, mark = "none", style = "blue, line width=1.1pt", legendentry = "Density"),
                       Plots.Linear([error_mean, error_mean], [-0.5, 14.0], mark = "none", style = "gray", legendentry = "Mean error"),
                       Plots.Node(L"\textrm{Mean error} = -2.6 \times 10^{-3}", 0.0, 10.5, style = "black")],
legendPos="north west", style = "grid = both, ytick = {0, 2, 4,...,14}, xtick = {-0.4, -0.3,...,0.4}", 
    xmin = -0.4, xmax = 0.4, ymin = -0.1, ymax = 14, xlabel = "Error", ylabel = "Density", title = "Train data - sips improved LDF kinetics"))


push!(histogram, Axis([Plots.Linear(collect(kde_test.x), kde_test.density, mark = "none", style = "red, line width=1.1pt, dashed", legendentry = "Density"),
                        Plots.Linear([error_test_mean, error_test_mean], [-0.5, 14.0], mark = "none", style = "gray", legendentry = "Mean error"),
                        Plots.Node(L"\textrm{Mean error} = 8.5 \times 10^{-4}", 0.0, 10.0, style = "black")],
legendPos="north west", style = "grid = both, ytick = {0, 2, 4,...,14}, xtick = {-0.4, -0.3,...,0.4}", 
    xmin = -0.4, xmax = 0.4, ymin = -0.1, ymax = 14, xlabel = "Error", ylabel = "Density", title = "Test data - Sips improved LDF kinetics"))
    

save("kde_improved_kldf_sips_error.pdf", histogram)


plot(solution_optim.t[1:end], learned_kinetics[:])

uptake = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(uptake, Axis([Plots.Linear(solution_optim.t[1:end], learned_kinetics[1:end], mark = "none", style = "blue", legendentry = "ANN prediction"),
            Plots.Linear(true_dqdt[1:end, 1], true_dqdt[1:end, 2], onlyMarks=true, style = "blue, mark = *, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "True uptake rate (train)"),
],
        legendPos="north east", style = "grid = both, ytick = {0, 1,...,7}, xtick = {0, 10, 20, ..., 120}", xmin = 0, xmax = 110, ymin = 0, ymax = 7, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{Uptake Rate}\,\left[\textrm{mgL}^{-1}\textrm{min}^{-1}\right]", title = "Sips isotherm - LDF"))


save("plots/uptake_kldf_sips.pdf", uptake)
