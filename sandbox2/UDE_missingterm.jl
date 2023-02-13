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
using Plots
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

p_init, st = Lux.setup(rng, nn)

best_p = Float32.(readdlm("trained_models/best_improved_quad_10_8_neurons_42fe_lang_tanh_2min_5e-7.csv"))
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
c0 = 0.0


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
saveat = saveats, abstol = 1e-7, reltol = 1e-7); #0.27 seconds after compiling

#Veryfing UDE fitting quality
scatter(c_exp_data[1:end, 1], c_exp_data[1:end, 2])
plot!(solution_optim.t[2:end], Array(solution_optim)[Int(n_variables/2), 2:end], linewidth = 2.)
#savefig(fig, "UDE_fitting_example.png")

#Creating missing term function~
q_eq_vec = []
q_vec = []
U_vec = []
lower = 30 
upper = size(solution_optim.t, 1) - 100
for i in 10:10:10
    c_ = solution_optim[Int(n_variables/2) - i, lower:upper]
    qeq_ = qmax*k_iso.*c_.^1.00./(1 .+ k_iso.*c_.^1.00)./q_test
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



# Finding missing term with sparse regression
#Expected missing term is: 0.11*z[1]^2*z[2]^-1 - 0.11*z[2] 

#Simplest expression I've got was φ₁ = p₂ + p₁*(z[2]^-1) + p₄*(z[1]^2)*(z[2]^-1) + p₃*(z[2]^-1)*z[1]
#= p₁ => -2.61895
p₂ => -7.22365
p₃ => 3.36801
p₄ => 0.07539 =# #STLSQ


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
for i ∈ -1:3, j ∈ -1:3
    if i == 0 && j == 0 
        nothing
    else
    poli2 = z[1]^i * z[2]^j
    push!(polys, poli2)
    end
end

h__f = [unique(polys)...]
basis = Basis(h__f, z)

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

#Sparse regression
options = DataDrivenCommonOptions(
    maxiters = 15_000, normalize = DataNormalization(),
    selector = bic, digits = 5,
    data_processing = DataProcessing(split = 1.0, batchsize = Int(round((size(X_expanded, 2)/1))), 
    shuffle = false, rng = StableRNG(1111)))

bics = []
lambdas = []
number_of_terms = []
rss_vec = []

for λ in exp10.(-2.0:0.1:0.0)
    println("lambda is $λ")
    println("\n")
    opt = STLSQ(λ) # λ < exp10(-1.35) gives error
    res = solve(problem_regression, basis, opt, options = options)
    system = get_basis(res);
    pas = get_parameter_map(system);
    println(system)
    println("\n")
    println("bic is", bic(res))
    println(res)

    push!(bics, bic(res))
    push!(lambdas, λ)
    push!(number_of_terms, size(pas, 1))
    push!(rss_vec, rss(res))
end

λ = exp10.(-3.0:0.05:0.0)
λ = 0.0398107
opt = STLSQ(λ) # λ < exp10(-1.35) gives error
res = solve(problem_regression, basis, opt, options = options)
println(res)
system = get_basis(res)
println(system)
get_parameter_map(system)


fig2 = Plots.plot(Plots.plot(problem_regression), Plots.plot(res))
savefig(fig2, "sparse_reg_example.png")

stats_reg = hcat(bics, lambdas, number_of_terms, rss_vec)
writedlm("sparse_reg_data/stats_improved_quad_lang.csv", stats_reg ,',')

@which plot(res)

log10(0.07943282347242814)