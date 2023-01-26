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

n_elements = 40 # Number of finite elements
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

Qf = 5e-2 #Feed flow rate (dm3/min)
d = 0.5e0  # Column diameter (dm)
dp = 1.0e-2 # particle diameter (dm)
L = 2.00e0 # Column length (dm)
a = pi * d^2 / 4e0 #Column cross section area (dm2)
epsilon = 0.5e0 # void fraction
u = Qf / (a * epsilon) #dm/min (drif velocity)
Dax = 0.7e0*0.1089e0*10^-2e0*60.0e0 + 0.5e0*dp*u
Pe = u*L/Dax

cin =  5.5e0
qmax = 55.54e0 #mg/g_s*g_s/cm3s*1000cm3/dm3 -> #mg/Lparticle
k_iso = 1.8e0
#q_test = 25.0*cin^0.6
q_test = qmax*k_iso*cin^1.5/(1 + k_iso*cin^1.5)

#params_ode = [11.66, 9.13, 5.08, 5.11, kappaa, kappab, 163.0, 0.42, 11.64, 0.95]

function round_zeros(x)
    if abs(x) < 1e-42
        0.0e0
    else
        Float64(x)
end
end

#Calculating the derivative matrices stencil
y_dy = round_zeros.(Array(A * H^-1)) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = round_zeros.(Array(B * H^-1)) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y


#--------Importing experimental data---------------
using DataInterpolations

c_exp_data = readdlm("traindata_improved_quad_sips_25min.csv", ',', Float64)
 # solid phase concentration measurements


# -----Initializing Neural networks---------
import Random

# ----- Lux

rng = Random.default_rng()
Random.seed!(rng, 13)

rbf(x) = exp.(-(x.^2))

nn = Lux.Chain(
  Lux.Dense(2, 22, tanh_fast),
  Lux.Dense(22, 1)
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
c0 = 5.00e-3

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


#----------- interpolation exogeneous ------

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
   q_eq  = qmax*k_iso*abs.(c).^1.50./(1.0 .+ k_iso.*abs.(c).^1.50)/q_test
   #q_eq = 25.0*abs.(c).^0.6/q_test
   #q_eq = interpolator.(c)/q_test

   q = ((@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1]) .- 0.0)/q_test #scaling dependent variables
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
       yp[cbl_idx] = dy_du[cbl_idx] / h .- Pe * (y[cbl_idx] .-  c_in)

       yp[cbu_idx] =  dy_du[cbu_idx] / h
   end
   nothing
end



# Building ODE problem
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, cin, dy_du, dy2_du);

f_node = ODEFunction(rhs, mass_matrix = MM)

#----- non optimized prob
y0 = y_initial(y0_cache, c0)

tspan = (0.00e0, 130.00e0) 


prob_node = ODEProblem(f_node, y0, tspan, Lux.ComponentArray(p_init))

LinearAlgebra.BLAS.set_num_threads(1)

ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())

@time solution_other = Array(solve(prob_node, FBDF(autodiff = false),
 abstol = 1e-7, reltol = 1e-7, saveat = 2.5e0)); #0.27 seconds after compiling

scatter(c_exp_data[1:end, 1], c_exp_data[1:end, 2])
plot!(c_exp_data[1:end, 1], solution_other[Int(n_variables/2), :])


#--------- Training Neural Network ----------

tsave = c_exp_data[2:end, 1]

function predict(θ)
    # --------------------------Sensealg---------------------------------------------
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))

    #----------------------------Problem solution-------------------------------------
    abstol = reltol = 5e-7
    tspan = (1e-12, maximum(c_exp_data[:, 1])) #TAVA ERRADOOO

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
predict(θ)

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
    l < 3.5e-1
end

opt = Flux.Optimiser(RMSProp(0.08), ExpDecay(1.0, 0.975, 30))

@time results = Optimization.solve(optprob, opt, callback = callback, maxiters = 190)

optf2 = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob2 = Optimization.OptimizationProblem(optf2, results.u)

@time results_2 = Optimization.solve(optprob2, Optim.BFGS(initial_stepnorm = 0.01), 
callback = callback, maxiters = 100, maxtime = 20*60, allow_f_increases = false)

aaa = predict(results.u)

loss(results.u)
mae = sqrt(Flux.mse(c_exp_data[2:end, 2], aaa[1:end]*cin))*100
println("MAE is $mae%")

scatter(c_exp_data[2:end, 1], c_exp_data[2:end, 2], label = " Experimental ", legend =:bottomright)
plot!(c_exp_data[2:end, 1], aaa[1:end]*cin, label = "neural UDE", legend=:bottomright, linewidth = 2.)
scatter(c_exp_data[2:end, 2], aaa*cin, label = nothing)
plot!(0:0.5:6.0, 0:0.5:6.0, label = nothing)

plot(c_exp_data[2:end, 1], c_exp_data[2:end, 2] .- aaa*cin, marker = 'o')

writedlm("best_improved_quad_22neurons_40fe_sips_tanh_25min.csv", results_2.u)

best_w
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
best_p = Float32.(readdlm("best_improved_quad_22neurons_40fe_sips_tanh_25min.csv"))
best_w = deepcopy(Float64.(Lux.ComponentArray(p_init)))
best_w = deepcopy(results.u)
neurons = 22
best_w.layer_1.weight  .= reshape(best_p[1:neurons*2], neurons, 2)
best_w.layer_1.bias .= reshape(best_p[neurons*2 + 1:neurons*2 + neurons], neurons, 1)
best_w.layer_2.weight .= reshape(best_p[neurons*2 + neurons + 1: neurons*2 + neurons + neurons], 1, neurons)
best_w.layer_2.bias .= reshape(best_p[neurons*2 + neurons + neurons + 1:end], 1, 1)

best_w.layer_1.weight  .= reshape(best_p[1:42], 21, 2)
best_w.layer_1.bias .= reshape(best_p[43:43 + 20], 21, 1)
best_w.layer_2.weight .= reshape(best_p[43 + 20 + 1: 43 + 20 + 1 + 20], 1, 21)
best_w.layer_2.bias .= reshape(best_p[41 + 19 + 1 + 19 + 1:end], 1, 1)

y0 = y_initial(y0_cache, 5e-3)
prob_node22 = ODEProblem(f_node, y0, (0.0, maximum(c_exp_data[:, 1])), best_w)

saveats = first(c_exp_data[:, 1]):mean(diff(c_exp_data[:, 1]))/10:last(c_exp_data[:, 1])
@time solution_optim = solve(prob_node22, FBDF(autodiff = false), saveat = saveats, abstol = 1e-7, reltol = 1e-7); #0.27 seconds after compiling
scatter(c_exp_data[1:end, 1], c_exp_data[1:end, 2])
plot!(solution_optim.t[2:end], Array(solution_optim)[Int(n_variables/2), 2:end], linewidth = 2.)

loss(best_w)

c_ = solution_optim[Int(n_variables/2) - 5, 1:end]
qeq_ = qmax*k_iso.*c_.^1.50./(1 .+ k_iso.*c_.^1.50)./q_test
q_ = Array(solution_optim)[Int(n_variables) - 5, 1:end]./q_test
learned_kinetics = nn([qeq_ q_]', best_w, st)[1] # Missing term

plot(solution_optim.t[1:end], learned_kinetics[:])

#Reading true missing term dynamics
true_dqdt = readdlm("true_dqdt_kldf_sips.csv", ',')

scatter(true_dqdt[:, 1], true_dqdt[:, 2])
plot(solution_optim.t[60:end], learned_kinetics[60:end], linewidth = 3.)

using DataDrivenSparse
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSR
using SymbolicRegression
using UnPack
using StableRNGs

@variables z[1:2]
z = collect(z)

polys = []
for i ∈ 0:2, j ∈ -1:1
    poli2 = z[1]^i * z[2]^j
    #poli2 = z[1]^i * z[2]^j * (ℯ^(dot(ones(size(w)), h_exp)))^k
    push!(polys, poli2)
end


h__f = [unique(polys)...]
basis = Basis(h__f, z)

lower = 20
upper = size(solution_optim.t, 1) - 150
X = [qeq_[lower:1:upper]'*q_test; q_[lower:1:upper]'*q_test]
Y = reshape(learned_kinetics[lower:1:upper], 1, size(learned_kinetics[lower:1:upper])[1])
problem_regression = DirectDataDrivenProblem(X, Y)
plot(problem_regression)

#Symbolic
eqsearch_options = SymbolicRegression.Options(binary_operators = [+],
                                              loss = L2DistLoss(),
                                              verbosity = 1, progress = false, npop = 100,
                                              timeout_in_seconds = 150.0)

alg = EQSearch(eq_options = eqsearch_options)

options = DataDrivenCommonOptions(
    maxiters = 100, normalize = DataNormalization(),
     selector = bic, digits = 5,
    data_processing = DataProcessing(split = 1.0, batchsize = 352, 
    shuffle = false, rng = StableRNG(1111)))

res2 = solve(problem_regression, basis, alg, options = options)
#plot(res2)
println(res2)
system_SR = get_basis(res2)
println(system_SR)
passs = get_parameter_map(system_SR)


#Sparse
options = DataDrivenCommonOptions(
    maxiters = 10_000, normalize = DataNormalization(),
     selector = bic, digits = 3,
    data_processing = DataProcessing(split = 1.0, batchsize = 352, 
    shuffle = true, rng = StableRNG(1111)))


opt = STLSQ(exp10.(-1.5:0.05:5.0))
res = solve(problem_regression, basis, opt, options = options)
system = get_basis(res)
pas = get_parameter_map(system)

println(res)
println(system)

plot(plot(problem_regression), plot(res))

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











#-----------------pSGLD(not using right now)
using ProgressBars
using Printf

beta = 0.99;
λ = 1e-5;
V_θ = zeros(length(θ))
a = 0.05; #0.0001; #try making this larger
b = 0.05;
γ = 0.10;

#Visualizing Δt stepping
t = 1:2000
y = @. a*(b + t)^(-γ)
plot(t,y)


function train_loop(θ, V_θ, iters, log_likelihod, log_regularization, params_log)
    t_count = 1
    y0 = y_initial(θ, y0_cache, c0)
    for t in iters

        ∇Likelihood = Zygote.gradient(θ -> loss_initial(θ, y0), θ)[1]
        ∇Prior = Zygote.gradient(regularization, θ)[1]

        if t == 1
            V_θ[:] = ∇Likelihood .* ∇Likelihood
        else
            V_θ *= beta
            V_θ += (1 - beta) * (∇Likelihood .* ∇Likelihood)
        end

        m = λ .+ sqrt.(V_θ)
        ϵ = a * (b + t_count)^-γ

        noise = sqrt.(ϵ./m).*randn(length(θ))

        θ = θ  - (0.5* ϵ * (∇Likelihood + ∇Prior) ./ m + noise)

        y0 = y_initial(θ, y0_cache, c0)

        
        loss_value = loss(θ, y0)
        reg = regularization(θ)
        set_description(iters, string(@sprintf("Loss train %.4e regularization %.4e", loss_value, reg)))
        push!(log_likelihod, loss_value)
        push!(log_regularization, reg)
        push!(params_log, θ)
        

        t_count += 1

    end
    θ
end


log_likelihod = []
log_regularization = []
params_log = []
n_iters = 2000
iters = ProgressBar(1:n_iters)

train_loop(θ, V_θ, iters, log_likelihod, log_regularization, params_log)

aaa = [params_log[i][4] for i in 1:size(params_log, 1)]
plot(aaa[1970:2270])
plot(log_likelihod[1876:1976])
minimum(log_likelihod[1:end])
argmin(log_likelihod[1:end])


y0_best = y_initial(params_log[1976], y0_cache, c0)
sol_best  = predict(params_log[1976], y0_best)

plot(t_exp[1:204], sol_best[1], label = nothing)
scatter!(t_exp[1:204], data_train[1], label = nothing)

x1 = (q_exp_data[:, 1].- 6.00)/(13.00 - 6.00)
x2 = (q_exp_data[:, 3] .- 0.047)/(4.00 - 0.047)
ax1 = scatter(q_exp_data[:, 1], q_exp_data[:, 2], label = nothing)
ax2 = scatter(q_exp_data[:, 3], q_exp_data[:, 4], label = nothing)

q1_pred_best = _ann1([x1 x2]', params_log[1976][1: 29])
q2_pred_best = _ann2([x1 x2]', params_log[1976][30: 30 + 28])

for i in 0:100
    global ax1, ax2
    θ_sampled = params_log[1976 + i]
    q1_pred = _ann1([x1 x2]', θ_sampled[1: 29])
    q2_pred = _ann2([x1 x2]', θ_sampled[30: 30 + 28])
    plot!(ax1, q_exp_data[:, 1], q1_pred[1, :], alpha=0.2, color="#BBBBBB", label = nothing)
    plot!(ax2, q_exp_data[:, 3], q2_pred[1, :], label = nothing, alpha=0.2, color="#BBBBBB")
end

scatter(ax1, q_exp_data[:, 1], q_exp_data[:, 2], label = nothing)
scatter(ax2, q_exp_data[:, 3], q_exp_data[:, 4], label = nothing)
plot!(ax1, q_exp_data[:, 1], q1_pred_best[1, :], label = nothing)
plot!(ax2, q_exp_data[:, 3], q2_pred_best[1, :], label = nothing)