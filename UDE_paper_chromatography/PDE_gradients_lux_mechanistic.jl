using Pkg
Pkg.activate(".")
Pkg.instantiate()

#Importing ODE, plot and MAT libraries
using OrdinaryDiffEq
#using DiffEqFlux
#using DiffEqCallbacks
#using DifferentialEquations
#using Flux
#using Lux
using Plots
using MAT
using DelimitedFiles
using SciMLSensitivity
import Random
#using PreallocationTools
using ForwardDiff, Zygote
using ReverseDiff
#using Flux
using StatsBase
pgfplotsx()
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

Qf = 5.0e-2 #Flow rate (dm^3/min)
d = 0.5  # Column Diameter (dm)
L = 2.0 # Bed Length (dm)
a = pi*d^2/4 # Column area (dm^2)
epsilon = 0.5 #Bed porosity
u = Qf/(a*epsilon) #Interstitial velocity (dm/min)
Pe = 21.095632695978704 #Peclet Number
Dax = u*L/Pe # Axial dispersion (dm^2/min)
cin = 5.5 # Feed concentration (mg/L)
k_transf = 0.22 #Mass transfer coefficient (1/min)
k_iso  = 1.8 # Isotherm affinity parameter (L/mg)
qmax = 55.54 # Isotherm saturation parameter (mg/L)
q_test = qmax*k_iso*cin^1.0/(1.0 + k_iso*cin^1.0) #Scale parameter for amount adsorbed in solid phase

#Plotting isotherm
c_iso = 0.0:0.05:10.0
q_iso = @. qmax*k_iso*c_iso/(1 + k_iso*c_iso)
plot(c_iso, q_iso, label = "test", tex_output_standalone = true, xlabel = "c (mg/L)")


#Calculating the derivative matrices stencil
y_dy = Array(A * H^-1) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = Array(B * H^-1) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y


#Initial condition vector
y0_cache = ones(Float64, n_variables)
c0 = 1e-3 # Initial liquid phase concentration

#Defining a function that creates the initial condition vector
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
   q_eq  = qmax*k_iso.*abs.(c).^1.0./(1.0 .+ k_iso.*abs.(c).^1.0) #Change exponent according to isotherm
   q = (@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1]) #scaling dependent variables

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
        
       yp[cl_idx:cu_idx] .= -(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1] + 0.2789*q_eq[2:end - 1].*exp.(-q[2:end-1]./2.0./ q_eq[2:end-1]) - q[2:end-1]) .- u*(@view dy_du[cl_idx:cu_idx]) / h / L  .+  Dax / (L^2) * (@view dy2_du[cl_idx:cu_idx]) / (h^2)

       # Choose appropriate kinetics to place in the above equation

       #(@view nn(x1x2, p, st)[1][2:end - 1])
       #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1] - q[2:end - 1])
       #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1] + 0.2789*q_eq[2:end - 1].*exp.(-q[2:end-1]./2.0./ q_eq[2:end-1]) - q[2:end-1])
       #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1].^2/2.0./q[2:end - 1] - q[2:end - 1]./2.0)
       #Solid phase residual

       # Choose kinetics to match the above choice
       #yp[ql_idx2:qu_idx2] .= k_transf * (q_eq - q)
       yp[ql_idx2:qu_idx2] .= k_transf * (q_eq + 0.2789*q_eq.*exp.(-q./2.0./q_eq) - q)
       #yp[ql_idx2:qu_idx2] .= k_transf * (q_eq.^2/2.0./q - q./2.0)


       #Boundary node equations
       yp[cbl_idx] = Dax / L * dy_du[cbl_idx] / h - u * (y[cbl_idx] -  c_in)

       yp[cbu_idx] =  dy_du[cbu_idx] / h / L
   end
   nothing
end


#------- Generating training set data
# Building ODE problem
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, cin, dy_du, dy2_du);

f_node = ODEFunction(rhs, mass_matrix = MM)

#Solving ODE problem
y0 = y_initial(y0_cache, c0)

tspan = (0.0, 110.0) 

prob_node = ODEProblem(f_node, y0, tspan, 2.0)

LinearAlgebra.BLAS.set_num_threads(1) #Reduce runtime in Ryzen 6800H CPUs

ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())

@time solution_other = solve(prob_node, FBDF(autodiff = false),
 abstol = 1e-6, reltol = 1e-6, saveat = 1.0); #0.27 seconds after compiling

plot(solution_other.t, Array(solution_other)[Int(n_variables/2), :])

#Adding Gaussian Noise to simulated solution
using Distributions

#Used truncated gaussian to avoid negative concentrations
samples = [rand(Truncated(Normal(i, 0.05), 0.0, 15)) for i in Array(solution_other)[Int(n_variables/2), :]];
scatter!(solution_other.t, samples)

using DelimitedFiles
dataset = hcat(solution_other.t, samples);
writedlm("train_data/traindata_improved_quad_lang_2min.csv", dataset, ",")

#Saving uptake rate
q_ = Array(solution_other)[Int(n_variables), :];
c_ = Array(solution_other)[Int(n_variables/2), :];
q_star = 1.8*55.54.*c_.^1.0./(1. .+ 1.8.*c_.^1.0); #Change according to equilibrium isotherm

dqdt = 0.22*(q_star + 0.2789*q_star.*exp.(-q_./2.0./q_star) - q_);
#dqdt = 0.22*(q_star - q_);
#dqdt = 0.22(q_star.^2/2.0./q_ - q_./2.0)
t_dqdt = hcat(solution_other.t[1:end], dqdt)
writedlm("test_data/true_dqdt_improved_quad_sips_2min.csv", t_dqdt, ',')

#Taylor series expansion on original solution_optim
using TaylorSeries

q_ast, q = set_variables("q_ast q", order = 2)
q_ast, q = set_variables("q_ast q", order = 1)

idx_to_value = 58
idx_to_value = 50

t_x = 0.22/2*((q_ast + q_star[idx_to_value])^2/(q + q_[idx_to_value]) - (q + q_[idx_to_value]))

t_x = 0.22*((q_ast + q_star[idx_to_value]) + 0.2789*(q_ast + q_star[idx_to_value]).*exp.(-(q + q_[idx_to_value])./2.0./(q_ast + q_star[idx_to_value])) - (q + q_[idx_to_value]))

t_x.(q_star .- q_star[idx_to_value], q_ .- q_[idx_to_value])
dqdt[idx_to_value]
approx_dqdt = t_x.(q_star .- q_star[idx_to_value], q_ .- q_[idx_to_value])
plot(approx_dqdt, label = "taylor expanded")

fig = plot(approx_dqdt, label = "Taylor expansion", xlabel = "sample ID")
plot!(fig, dqdt, label = "True uptake rate", ylabel = "Uptake rate (mg/L/min)", legend=:topright, seriestype=:scatter) 
vline!(fig, [idx_to_value], color = "gray", label = "Taylor expansion point")
savefig(fig, "taylor_improved_ldf.pdf")
q_star[idx_to_value]
q_[idx_to_value]


#------- Generating test set data

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

function (f::col_model_test)(yp, y, p, t)
    #Aliasing parameters
 
    @unpack n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
    Pe, epsilon, c_in, dy_du, dy2_du  = f 
    
    
    dy_du =  y_dy*y
    dy2_du = y_dy2*y
 
    
    j = 0
    #---------------------Mass Transfer and equilibrium -----------------
 
    c = (@view y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1]) #Scaling dependent variables
    q_eq  = qmax*k_iso.*abs.(c).^1.5./(1.0 .+ k_iso.*abs.(c).^1.5)

    q = (@view y[2 + (p_order + 2*n_elements - 2) - 1: p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1]) #scaling dependent variables
    #x1x2 =  [q_eq q]'
 
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
         
        yp[cl_idx:cu_idx] .= -(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1].^2/2.0./q[2:end - 1] - q[2:end - 1]./2.0) .- u*(@view dy_du[cl_idx:cu_idx]) / h / L  .+  Dax / (L^2) * (@view dy2_du[cl_idx:cu_idx]) / (h^2)
 
        #(@view nn(x1x2, p, st)[1][2:end - 1])
        #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1] - q[2:end - 1])
        #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1] + 0.2789*q_eq[2:end - 1].*exp.(-q[2:end-1]./2.0./ q_eq[2:end-1]) - q[2:end-1])
        #-(1 - epsilon) / epsilon  * k_transf * (q_eq[2:end - 1].^2/2.0./q[2:end - 1] - q[2:end - 1]./2.0)
        #Solid phase residual
 
        #yp[ql_idx2:qu_idx2] .= k_transf * (q_eq - q)
        #yp[ql_idx2:qu_idx2] .= k_transf * (q_eq + 0.2789*q_eq.*exp.(-q./2.0./q_eq) - q)
        yp[ql_idx2:qu_idx2] .= k_transf * (q_eq.^2/2.0./q - q./2.0)
 
        #(@view nn(x1x2, p, st)[1][1:end])
 
        #ex_[i](t)
        #Boundary node equations
        yp[cbl_idx] = Dax / L * dy_du[cbl_idx] / h - u * (y[cbl_idx] -  c_in(t))
 
        yp[cbu_idx] =  dy_du[cbu_idx] / h / L
    end
    nothing
 end


# Feed concentration signal (built with interpolation and tstops instead of callbacking)
using DataInterpolations

t_interp_lang = [0.0:0.1:110.0; 110.0000001; 120.00:5.:250.0; 250.0000001; 260.0:5.0:500.]
c_interp_lang = [fill(5.5, size(0.0:0.1:110., 1)); 3.58; fill(3.58, size(120.00:5.:250., 1)); 7.33;
 fill(7.33, size(260.0:5.0:500., 1))]

t_interp_sips = [0.0:0.1:110.0; 110.0000001; 120.00:5.:250.0; 250.0000001; 260.0:5.0:500.]
c_interp_sips = [fill(5.5, size(0.0:0.1:110., 1)); 0.75; fill(0.75, size(120.00:5.:250., 1)); 9.33;
 fill(9.33, size(260.0:5.0:500., 1))]

scatter(t_interp_sips, c_interp_sips)
c_in_t = LinearInterpolation(c_interp_sips, t_interp_sips)


# Building and solving ODE problem
rhs_test = col_model_test(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
Pe, epsilon, c_in_t, dy_du, dy2_du);
f_node_test = ODEFunction(rhs_test, mass_matrix = MM)
y0 = y_initial(y0_cache, 1e-3)
tspan_test = (0.00e0, 400.00e0)

prob_node_test = ODEProblem(f_node_test, y0, tspan_test, Nothing) 
solution_test = solve(prob_node_test, FBDF(autodiff = false), 
abstol = 1e-6, reltol = 1e-6, tstops = [0.0, 110., 250.], saveat = 2.0e0);

#Adding Gaussian noise to simulated data
using Distributions

samples_test = [rand(Truncated(Normal(i, 0.05), 0.0, 15)) for i in Array(solution_test)[Int(n_variables/2), :]];

scatter(solution_test.t, samples_test)

dataset_test = hcat(solution_test.t, samples_test);

using DelimitedFiles

writedlm("test_data/testdata_improved_quad_sips_2min.csv", dataset_test, ",")



