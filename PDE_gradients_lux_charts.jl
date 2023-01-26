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

c_exp_data = readdlm("train_data/traindata_improved_quad_sips_25min.csv", ',', Float64) # solid phase concentration measurements


# -----Initializing Neural networks---------
import Random

# ----- Lux

rng = Random.default_rng()
Random.seed!(rng, 2)

nn = Lux.Chain(
  Lux.Dense(2, 22, tanh_fast),
  Lux.Dense(22, 1)
)

p_init, st = Lux.setup(rng, nn)


using Optimization

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


y0_cache = ones(Float64, n_variables)
c0 = 5.00e-3

function y_initial(y0_cache, c0)

    #var0 = get_tmp(y0_cache, p)
    #c0 = get_tmp(c0, p)
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



#using MKL
#using .BLAS
LinearAlgebra.BLAS.set_num_threads(1)
#Threads.nthreads()
ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())

@time solution_other = Array(solve(prob_node, FBDF(autodiff = false),
 abstol = 1e-7, reltol = 1e-7, saveat = 1.0e0)); #0.27 seconds after compiling


#--------- Training Neural Network ----------

tsave = c_exp_data[2:end, 1]

function predict(θ)
    # --------------------------Sensealg---------------------------------------------
    sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    #sensealg = ForwardDiffSensitivity()
    #----------------------------Problem solution-------------------------------------
    abstol = reltol = 1e-6
    tspan = (1e-12, maximum(c_exp_data[:, 1])) #TAVA ERRADOOO

    prob_ = remake(prob_node; p = θ, tspan = tspan)
    #prob_ = remake(prob_odae; p = θ)

    s_new = Array(solve(prob_, FBDF(autodiff = false), abstol = abstol, reltol = reltol,
    saveat = tsave, sensealg = sensealg))

    #s_new = solve(prob_, QNDF(autodiff = false), abstol = abstol, reltol = reltol,
    #saveat = 0.5, tstops = [0.0, 100.0], sensealg = sensealg)

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
#@time grad_forward = ForwardDiff.gradient(loss, θ)
@time grad_reverse = ReverseDiff.gradient(loss, θ)
@time grad_regularization = Zygote.gradient(regularization, θ)[1]



#-------------- sparse regression

#----model loading
using DelimitedFiles
best_p = Float64.(readdlm("trained_models/best_improved_quad_22neurons_40fe_sips_tanh_25min.csv"))
best_w = deepcopy((Lux.ComponentArray(p_init)))
best_w = deepcopy(results_2.u)
neurons = 22
best_w.layer_1.weight  .= reshape(best_p[1:neurons*2], neurons, 2)
best_w.layer_1.bias .= reshape(best_p[neurons*2 + 1:neurons*2 + neurons], neurons, 1)
best_w.layer_2.weight .= reshape(best_p[neurons*2 + neurons + 1: neurons*2 + neurons + neurons], 1, neurons)
best_w.layer_2.bias .= reshape(best_p[neurons*2 + neurons + neurons + 1:end], 1, 1)


y0 = y_initial(y0_cache, 5e-3)
prob_node22 = ODEProblem(f_node, y0, (0.0, 130.0), Lux.ComponentArray(best_w))

saveats = first(c_exp_data[:, 1]):mean(diff(c_exp_data[:, 1]))/10:last(c_exp_data[:, 1])
@time solution_optim = solve(prob_node22, FBDF(autodiff = false),
 abstol = 1e-7, reltol = 1e-7, saveat = saveats); #0.27 seconds after compiling

plot(solution_optim.t, Array(solution_optim)[Int(n_variables/2), :])

loss(best_w)

c_ = solution_optim[Int(n_variables/2), 1:end]
qeq_ = qmax*k_iso.*c_.^1.50./(1 .+ k_iso.*c_.^1.50)./q_test
#qeq_f = 25.0*c_.^0.6./q_test
q_ = Array(solution_optim)[Int(n_variables), 1:end]./q_test

learned_kinetics = nn([qeq_ q_]', best_w, st)[1]
plot(solution_optim.t[1:end], learned_kinetics[:])

true_dqdt = readdlm("test_data/true_dqdt_improved_quad_sips_25min.csv", ',')


#----------------Desorption and extrapolation
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
    q_eq  = qmax*k_iso*abs.(c).^1.50./(1.0 .+ k_iso.*abs.(c).^1.50)/q_test

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


        #yp[ql_idx2:qu_idx2] .= uptake_reg(q_eq, q)



        #Boundary node equations
        yp[cbl_idx] = dy_du[cbl_idx] / h .- Pe * (y[cbl_idx] .-  c_in(t))

        yp[cbu_idx] =  dy_du[cbu_idx] / h
    end
    nothing
end

using DataInterpolations



#= t_interp = [0.0:5.0:130.0; 130.0000001; 160.00:5.:250.0; 250.0000001; 270.0:5.0:400.]
c_interp = [fill(5.50, size(0.0:5.0:130.0, 1)); 3.5851; fill(3.5851, size(160.00:5.:250.0, 1)); 7.33;
 fill(7.33, size(270.0:5.0:400.0, 1))] =#

t_interp = [0.0:0.1:130.; 130.0000001; 140.00:5.:250.; 250.0000001; 260.0:5.0:500.]
c_interp = [fill(5.5, size(0.0:0.1:130., 1)); 0.75; fill(0.75, size(140.00:5.:250., 1)); 9.33;
fill(9.33, size(260.0:5.0:500., 1))]

c_in_t = LinearInterpolation(c_interp, t_interp)

rhs_test = col_model_node_test(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, c_in_t, dy_du, dy2_du);
f_node_test = ODEFunction(rhs_test, mass_matrix = MM)
y0 = y_initial(y0_cache, 5.0e-3)
tspan_test = (0.0, 400.00e0) 
#tspan_test = (0.0, 1e0)
prob_node_test = ODEProblem(f_node_test, y0, tspan_test, best_w)


test_rate = 2.5
@time solution_test = Array(solve(prob_node_test, FBDF(autodiff = false), 
abstol = 1e-6, reltol = 1e-6, saveat = test_rate,  tstops = [0.0, 130., 250])); #0.27 seconds after compiling

test_data = readdlm("test_data/testdata_improved_quad_sips_25min.csv", ',')

using PGFPlots

history = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(history, Axis([Plots.Linear(0.0:test_rate:130.0 |> collect, solution_test[Int(n_variables/2), 1:size(c_exp_data, 1)], mark = "none", style = "blue", legendentry = "UDE prediction - Train"),
            Plots.Linear(c_exp_data[1:end, 1], c_exp_data[1:end, 2], onlyMarks=true, style = "blue, mark = *, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Train"),  
            Plots.Linear(132.5:2.5:400 |> collect, solution_test[Int(n_variables/2), size(c_exp_data, 1) + 1:end], mark = "none", style = "red!60, dashed", legendentry = "UDE prediction - Test"),
            Plots.Linear(test_data[size(c_exp_data, 1):end, 1], test_data[size(c_exp_data, 1):end, 2],onlyMarks=true, style = "red!60, mark = square*, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Test"),
            Plots.Linear([130., 130.], [0., 10.5], mark = "none", style = "black"),
            Plots.Node("Train data", 30, 7, style = "blue"),
            Plots.Node("Test data", 210, 7, style = "red!60")
],
        legendPos="south east", style = "grid = both, ytick = {0, 2, 4, 6, 8, 10}, xtick = {0, 40, 80,...,400}, legend style={nodes={scale=0.5, transform shape}}", xmin = 0, xmax = 400, ymin = 0, ymax = 10, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{c}\,\left[\textrm{mg}\,\textrm{L}^{-1}\right]", title = "Sips isotherm - Vermeulen's - low sampling rate"))

save("plots/improved_quad_sips_history_lowsample.pdf", history)       


history_error = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(history_error, Axis([Plots.Linear(0.0:test_rate:130.0 |> collect, solution_test[Int(n_variables/2), 1:size(c_exp_data, 1)] .- c_exp_data[1:end, 2], mark = "none", style = "blue", legendentry = "Prediction error - Train"),  
            Plots.Linear(132.5:2.5:400 |> collect, solution_test[Int(n_variables/2), size(c_exp_data, 1) + 1:end] .- test_data[size(c_exp_data, 1):end, 2], mark = "none", style = "red!60, dashed", legendentry = "Prediction error - Test"),
            Plots.Linear([130., 130.], [-0.5, 0.5], mark = "none", style = "black"),
            Plots.Node("Train data", 30, 0.25, style = "blue"),
            Plots.Node("Test data", 210, 0.25, style = "red!60")
],
        legendPos="south east", style = "grid = both, ytick = {-0.4,-0.2, 0.0, 0.2, 0.4}, xtick = {0, 40, 80,...,400},  legend style={nodes={scale=0.5, transform shape}}", xmin = 0, xmax = 400, ymin = -0.4, ymax = 0.4, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{\varepsilon}\,\left[\textrm{mg}\,\textrm{L}^{-1}\right]", title = "Sips isotherm - Vermeulen's - low sampling rate"))

save("plots/improved_quad_sips_history_error_lowsample.pdf", history_error)



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
        legendPos="north east", style = "grid = both, ytick = {0, 1,...,7}, xtick = {0, 10, 20, ..., 120}", xmin = 0, xmax = 120, ymin = 0, ymax = 7, width = "16cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{Uptake Rate}\,\left[\textrm{mgL}^{-1}\textrm{min}^{-1}\right]", title = "Sips isotherm - Vermeulen's - low sampling rate"))


save("plots/uptake_improved_quad_sips_lowsample.pdf", uptake)




#-----------------pSGLD()
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