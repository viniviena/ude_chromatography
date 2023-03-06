using Pkg
Pkg.activate(".")
Pkg.instantiate()

#Importing ODE, plot and MAT libraries
using OrdinaryDiffEq
using DiffEqFlux
#using DiffEqCallbacks
#using DifferentialEquations
#using Flux
using Lux
using PGFPlots
#using MAT
using DelimitedFiles
using SciMLSensitivity
import Random
#using PreallocationTools
#using ForwardDiff, Zygote
#using ReverseDiff
#using Flux
using StatsBase


#----model loading
import Random

rng = Random.default_rng()
Random.seed!(rng, 13)

nn = Lux.Chain(
  Lux.Dense(2, 10, tanh_fast),
  Lux.Dense(10, 8, tanh_fast),
  Lux.Dense(8, 1)
)

nn = Lux.Chain(
  Lux.Dense(2, 17, tanh_fast),
  Lux.Dense(17, 1)
)

p_init, st = Lux.setup(rng, nn)

best_p = Float64.(readdlm("trained_models/best_improved_quad_10_8_neurons_42fe_lang_tanh_2min_5e-7_abs2.csv"))
best_w = deepcopy(Float64.(Lux.ComponentArray(p_init)))
neurons = 17
best_w.layer_1.weight .= reshape(best_p[1:neurons*2], neurons, 2)
best_w.layer_1.bias .= reshape(best_p[neurons*2 + 1:neurons*2 + neurons], neurons, 1)
best_w.layer_2.weight .= reshape(best_p[neurons*2 + neurons + 1: neurons*2 + neurons + neurons], 1, neurons)
best_w.layer_2.bias .= reshape(best_p[neurons*2 + neurons + neurons + 1:end], 1, 1)


best_w.layer_1.weight  .= reshape(best_p[1:20], 10, 2)
best_w.layer_1.bias .= reshape(best_p[21:21 + 9], 10, 1)
best_w.layer_2.weight .= reshape(best_p[21 + 9 + 1: 21 + 9 + 1 + 10*8 - 1], 8, 10)
best_w.layer_2.bias .= reshape(best_p[21 + 9 + 1 + 10*8: 21 + 9 + 1 + 10*8 + 7], 8, 1)
best_w.layer_3.weight .= reshape(best_p[21 + 9 + 1 + 10*8 + 7 + 1: 21 + 9 + 1 + 10*8 + 7 + 1 + 7], 1, 8)
best_w.layer_3.bias .= reshape(best_p[21 + 9 + 1 + 10*8 + 7 + 1 + 7 + 1:end], 1, 1)

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
q_test = qmax*k_iso*cin^1.0/(1.0 + k_iso*cin^1.0)


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
    var0[ql_idx2:qu_idx2] .= qmax*k_iso*c0^1.0/(1.0 + k_iso*c0^1.0)
    #var0[ql_idx2:qu_idx2] .= 25.0*c0.^0.6
    #var0[ql_idx2:qu_idx2] .= radial_surrogate.(c0)
    #var0[ql_idx2:qu_idx2] .= interpolator.(c0)

    j = j + p_order + 2 * n_elements - 2
    end

    var0

end


y0 =  y_initial(y0_cache, c0)


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
    q_eq  = qmax*k_iso.*abs.(c).^1.0./(1.0 .+ k_iso.*abs.(c).^1.0)/q_test
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
    
 
#Importing experimental data
c_exp_data = readdlm("train_data/traindata_improved_quad_lang_2min.csv", ',', Float64)


# Building UDE problem
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, cin, dy_du, dy2_du);
f_node = ODEFunction(rhs, mass_matrix = MM)
prob_node22 = ODEProblem(f_node, y0, (first(c_exp_data[:, 1]), last(c_exp_data[:, 1])), best_w)
saveats = first(c_exp_data[:, 1]):mean(diff(c_exp_data[:, 1]))/10:last(c_exp_data[:, 1])

#Solving UDE Problem
@time solution_optim = solve(prob_node22, FBDF(autodiff = false), 
abstol = 5e-7, reltol = 5e-7, saveat = saveats); #0.27 seconds after compiling

#sum(abs, Array(solution_optim)[Int(n_variables/2), 1:end]/cin .- c_exp_data[:, 2]/cin)

#Veryfing UDE fitting quality
plot(c_exp_data[1:end, 1], c_exp_data[1:end, 2])
plot(solution_optim.t[2:end], Array(solution_optim)[Int(n_variables/2), 2:end])
#savefig(fig, "UDE_fitting_example.png")

#Creating missing term function~
q_eq_vec = []
q_vec = []
U_vec = []
lower = 30
upper = size(solution_optim.t, 1) - 100
for i in 0:10:20 #change qeq
    println(i)
    c_ = solution_optim[Int(n_variables/2) - i, lower:upper]
    qeq_ = qmax*k_iso.*abs.(c_).^1.00./(1 .+ k_iso.*abs.(c_).^1.00)./q_test
    push!(q_eq_vec, qeq_)
    q_ = Array(solution_optim)[Int(n_variables) - i, lower:upper]./q_test
    push!(q_vec, q_)
    X_scaled = [qeq_ q_]' #Predictors
    U = nn(X_scaled, best_w, st)[1] #Missing term/interaction
    push!(U_vec, U)
end

U_vec = transpose(mapreduce(permutedims, vcat, U_vec))
q_eq_vec = mapreduce(permutedims, hcat, q_eq_vec)
q_vec = mapreduce(permutedims, hcat, q_vec)

plot(solution_optim.t[lower:upper], U_vec[:])
plot(saveats[lower:upper], q_vec[:])
y_linear = log.(1.00011 .- (q_vec./q_eq_vec).^2)
plot(saveats[lower:upper], - y_linear[:])


using DataDrivenSparse
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSR
using SymbolicRegression
using UnPack
using StableRNGs

@variables q_ast, q
#z = collect(z)
@variables z[1:2]

polys = []
for i ∈ 0:6, j ∈ 0:6
#    if i == 0 && j == 0 
#        nothing
#    else
    poli2 = q_ast^i * q^j
    push!(polys, poli2)
#    end
end

polynomial_basis(z, 2)

h__f = [unique(polys)...]
#b = polynomial_basis(z, 3)
basis = Basis(h__f, [q_ast, q])

#Defining datadriven problem

#Defining limits to make the problem more simetric (See in Figure)
#= lower = 20 
upper = size(solution_optim.t, 1) - 180 =#

#X = [qeq_[lower:1:upper]'*q_test; q_[lower:1:upper]'*q_test]
#Y = reshape(U[lower:1:upper], 1, size(U[lower:1:upper])[1])

X_expanded = [q_eq_vec[1:1:end]'*q_test; q_vec[1:1:end]'*q_test]
Y_expanded = U_vec
problem_regression = DirectDataDrivenProblem(X_expanded, Y_expanded)
Plots.plot(problem_regression)

#Exporting data for testing Feynmann AI
x_y = [X_expanded' Y_expanded']
writedlm("sparse_reg_data/feyman_AI_data.txt", x_y,  " ")

#Sparse regression
options = DataDrivenCommonOptions(
    maxiters = 500, normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 3,
    data_processing = DataProcessing(split = 0.95, batchsize = Int(round((size(X_expanded, 2)/10))), 
    shuffle = true, rng = StableRNG(1112)))


#Sparse regression
λ = exp10.(-3.0:0.05:0.5)
opt2 = ADMM(λ) # λ < exp10(-1.35) gives error
res = solve(problem_regression, basis, opt2, options = options)
println(res)
system = get_basis(res)
println(system)
maps = get_parameter_map(system)
bic(res)


nn_eqs = get_basis(res)


function parameter_loss(p)
    Y = map(Base.Fix2(nn_eqs, p), eachcol(X_expanded))
    sum(abs2, Y_expanded[:] .- mapreduce(permutedims, vcat, Y))
end

a = parameter_loss(get_parameter_values(nn_eqs))
using Optimization
adtype = Optimization.AutoForwardDiff()
optf = Optimization.OptimizationFunction((x, p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, BFGS(), maxiters = 5000)
parameter_loss(parameter_res.u)
parameter_res.u

#Taylor expanding original terms

using TaylorSeries

c_t = solution_optim[Int(n_variables/2), lower:upper]
qeq_t = qmax*k_iso.*abs.(c_t).^1.00./(1 .+ k_iso.*abs.(c_t).^1.00)
q_t = Array(solution_optim)[Int(n_variables), lower:upper]
dqdt_t = nn([qeq_t/q_test q_t/q_test]', best_w, st)[1]
plot(1:size(qeq_t, 1), dqdt_t[:])
q_t[80]
qeq_t[80]

taylor_expand(x -> 1/x, 20, order = 2)

x, y = set_variables("x y", order = 2)

idx_to_value = 200
t_x = 0.22/2*((x + qeq_t[idx_to_value])^2 - (y + q_t[idx_to_value])^2)/(y + q_t[idx_to_value])

t_x_nn = nn([(x + qeq_t[idx_to_value]/q_test) (y + q_t[idx_to_value]/q_test)]', best_w, st)[1]

approx_dqdt = t_x_nn[1].(qeq_t/q_test, q_t/q_test)
plot(1:size(qeq_t, 1), approx_dqdt)

#Symbolic regression

eqsearch_options = SymbolicRegression.Options(binary_operators = [+, *],
                                              loss = L2DistLoss(),
                                              verbosity = 1, progress = true, npop = 30,
                                              timeout_in_seconds = 80.0)



alg_SR = EQSearch(eq_options = eqsearch_options)

res = solve(problem_regression, basis, alg_SR, options = options)
println(res)
system = get_basis(res)
println(system)

fig2 = Plots.plot(Plots.plot(problem_regression), Plots.plot(res))
savefig(fig2, "sparse_reg_example.png")

stats_reg = Float64.(hcat(bics, lambdas, number_of_terms, rss_vec))
writedlm("sparse_reg_data/stats_improved_kldf_sips_abs2.csv", stats_reg ,',')

using StatsBase
#Plotting
fig2 = GroupPlot(2, 1, groupStyle = "horizontal sep = 2.0cm, vertical sep = 2.0cm");
push!(fig2, Axis([Plots.Linear(stats_reg[1:end, 2], stats_reg[1:end, 1], style = "blue")],
        legendPos="south east", xmode = "log", xlabel = L"\lambda", ylabel = "Bayesian Information Criterion"))

push!(fig2, Axis([Plots.Linear(stats_reg[1:end, 2], stats_reg[1:end, 3], style = "blue, mark = square*")],
legendPos="south east", xmode = "log", xlabel = L"\lambda", ylabel = "Number of active terms"))
save("sparse_reg_data/kldf_sipsss.pdf", fig2)


Y = map(Base.Fix2(nn_eqs, parameter_res.u), eachcol(X_expanded))
dqdt_reg = mapreduce(permutedims, vcat, Y)

fig3 = GroupPlot(3, 1, groupStyle = "horizontal sep = 3.0cm, vertical sep = 2.0cm");
push!(fig3, Axis([Plots.Linear(1:1:size(U_vec, 2) |> collect, q_eq_vec[:]*q_test, legendentry = L"q^*", mark = "none", style = "blue!60"),
Plots.Linear(1:1:size(U_vec, 2) |> collect, q_vec[:]*q_test, legendentry = L"q", mark = "none", style = "red!60")],legendPos="south east", xlabel = "Sample ID", ylabel = "adsorbed amount (mg/L)"))
push!(fig3, Axis([Plots.Linear(1:1:size(U_vec, 2) |> collect, Y_expanded[:], legendentry = L"\partial q / \partial t", mark = "none", style = "blue!60"),
Plots.Linear(1:1:size(U_vec, 2) |> collect, dqdt_reg[:], legendentry = L"\partial \hat{q} / \partial t", mark = "none", style = "black!60, dashed")],legendPos="north east", xlabel = "Sample ID", ylabel = "uptake rate (mg/L)"))
push!(fig3, Axis([Plots.Linear(1:1:size(U_vec, 2) |> collect, U_vec[:] - dqdt_reg[:], legendentry = L"error", mark = "none", style = "red!60, dashed")],legendPos="north east", xlabel = "Sample ID", ylabel = "error (mg/L)"))
save("sparse_reg_data/improved_quad_lang_performance_6terms.pdf", fig3)



plot(1:1:size(U_vec, 2) |> collect, dqdt_reg)
plot!(1:1:size(U_vec, 2) |> collect, Y_expanded[:])



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
   q_eq  = qmax*k_iso.*abs.(c).^1.0./(1.0 .+ k_iso.*abs.(c).^1.0)
   #q_eq = 25.0*abs.(c).^0.6/q_test
   #q_eq = qmax*k_iso^(1/t)*p./(1.0 .+ k_iso*abs.(p).^t).^(1/t)*ρ_p  

   q = (@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1]) #scaling dependent variables
   x1x2 =  [q_eq q]'

   û = mapreduce(permutedims, vcat, map(Base.Fix2(nn_eqs, p), eachcol(x1x2)))
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
        
       yp[cl_idx:cu_idx] .= - (1 - epsilon) / epsilon * (@view û[2:end - 1])  .- u*(@view dy_du[cl_idx:cu_idx]) / h / L  .+  Dax / (L^2) * (@view dy2_du[cl_idx:cu_idx]) / (h^2)

       #(@view nn(x1x2, p, st)[1][2:end - 1])
       #Solid phase residual

       yp[ql_idx2:qu_idx2] .= û

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

#scatter(t_interp_lang, c_interp_lang)
c_in_t = LinearInterpolation(c_interp_lang, t_interp_lang)

rhs_test = col_model_test(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, c_in_t, dy_du, dy2_du);
f_node_test = ODEFunction(rhs_test, mass_matrix = MM)
y0 = y_initial(y0_cache, 1e-3)
tspan_test = (0.00e0, 400.00e0)

prob_node_test = ODEProblem(f_node_test, y0, tspan_test, parameter_res.u) 
solution_test = solve(prob_node_test, FBDF(autodiff = false), 
abstol = 1e-6, reltol = 1e-6, tstops = [0.0, 110., 250.], saveat = 2.0e0);


test_data = readdlm("test_data/testdata_improved_quad_lang_2min.csv", ',')
test_rate = c_exp_data[2, 1] - c_exp_data[1, 1]

using PGFPlots

history = GroupPlot(1, 1, groupStyle = "horizontal sep = 2.75cm, vertical sep = 2.0cm");
push!(history, Axis([Plots.Linear(0.0:test_rate:110.0 |> collect, solution_test[Int(n_variables/2), 1:size(c_exp_data, 1)], mark = "none", style = "blue", legendentry = "Polynomial prediction - Train"),
            Plots.Linear(c_exp_data[1:end, 1], c_exp_data[1:end, 2], onlyMarks=true, style = "blue, mark = *, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Train"),  
            Plots.Linear(110.0 + test_rate:test_rate:400 |> collect, solution_test[Int(n_variables/2), size(c_exp_data, 1) + 1:end], mark = "none", style = "red!60, dashed", legendentry = "Polynomial prediction - Test"),
            Plots.Linear(test_data[size(c_exp_data, 1) + 1:end, 1], test_data[size(c_exp_data, 1) + 1:end, 2],onlyMarks=true, style = "red!60, mark = square*, mark options={scale=0.9, fill=white, fill opacity = 0.1}", legendentry = "Observations - Test"),
            Plots.Linear([110., 110.], [0., 10.5], mark = "none", style = "black"),
            Plots.Node("Train data", 30, 7, style = "blue"),
            Plots.Node("Test data", 210, 7, style = "red!60")
],
        legendPos="south east", style = "grid = both, ytick = {0, 2, 4, 6, 8, 10}, xtick = {0, 40, 80,...,400}, legend style={nodes={scale=0.5, transform shape}}", xmin = 0, xmax = 400, ymin = 0, ymax = 10, width = "14cm", height = "6cm", xlabel = "time [min]",
       ylabel=L"\textrm{c}\,\left[\textrm{mg}\,\textrm{L}^{-1}\right]", title = "Langmuir isotherm - improved LDF - Sparse regression"))


save("sparse_reg_data/improved_quad_lang_history_sparsereg.pdf", history)