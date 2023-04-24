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

n_elements = 50 # Number of finite elements
collocation_points = 2 #Collocation points
n_components = 1;  # 2 chemical species
n_phases = 1 #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ = 0.0e0
xₘₐₓ = 1.0e0 # z domain limits
h = (xₘₐₓ - xₘᵢₙ) / n_elements #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components) #make matrices for OCFEM

#Building mass matrix
MM = BitMatrix(Array(make_MM_2(n_elements, n_phases, n_components))) #make mass matrix

function round_zeros(x)
    if abs(x) < 1e-42
        0.0e0
    else
        Float64(x)
end
end


#-------- Defining PDE parameters------------

#Benzene

Qf = 6.0 #cm3/min
d = 1.5 # cm 
L = 4.0 #cm 
a = pi*d^2/4
epsilon = 0.42
kiso = 1111.0 #mL/g
ρ = 0.140 # g/mL
rf = 1 + ρ*kiso/epsilon # 1.0 + g/mL*mL/g  
u = Qf / (a * epsilon) #cm/min
u_eff = u/rf
Deff = 4.72e-9*10^4*60 #cm2/min
D = Deff*rf
cin = 0.1 #mM


#params_ode = [11.66, 9.13, 5.08, 5.11, kappaa, kappab, 163.0, 0.42, 11.64, 0.95]

#Calculating the derivative matrices stencil
y_dy = round_zeros.(Array(A * H^-1)) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = round_zeros.(Array(B * H^-1)) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y

stencil = - u_eff / h / L * y_dy .+ Deff / L^2 / h^2 * y_dy2

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
    var0[cbl_idx] = cin

    var0[cbu_idx] = c0

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

   #-------------------------------mass balance -----------------

   begin
       #Internal node equations
       cl_idx = 2 + j
       cu_idx = p_order + 2 * n_elements - 3 + j

       cbl_idx = j + 1
       cbu_idx = j + p_order + 2 * n_elements - 2

       #Liquid phase residual
       yp[cl_idx:cu_idx] .=  - u_eff*(@view dy_du[cl_idx:cu_idx]) / h / L  .+  Deff / (L^2) * (@view dy2_du[cl_idx:cu_idx]) / (h^2)

       #yp[cl_idx:cu_idx] .= stencil[cl_idx:cu_idx, 1:end]*y

       yp[cbl_idx] = D/L/h* dy_du[cbl_idx]  - u * (y[cbl_idx] -  cin)

       yp[cbu_idx] = dy_du[cbu_idx] / h / L

       #yp[cbu_idx] = 0.0
   end
   nothing
end



# Building ODE problem
rhs = col_model_node1(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, 
15, epsilon, cin, dy_du, dy2_du);

f_node = ODEFunction(rhs, mass_matrix = MM)

#----- non optimized prob

tspan = (0.0, 450) 

prob_node = ODEProblem(f_node, y0, tspan, 2.0)

LinearAlgebra.BLAS.set_num_threads(1)

ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())

@time solution_other = solve(prob_node, FBDF(autodiff = false)); #0.27 seconds after compiling
plot(solution_other.t, Array(solution_other)[Int(n_variables), :]/cin)

U = Array([0.2113248654, 0.7886751346])
xmin = 0.0
xmax = L
x = []
for i=1:n_elements
    append!(x, ones(2)*h*(i-1) + [U[1]*h + xmin, U[2]*h + xmin])
end
x = union(x);
x = [0.0; x; 4.0]

using SpecialFunctions
x_ = last(x)
t = solution_other.t
analytical_sol = @. 0.5*(erfc((x_ - u_eff*t)/(2*sqrt(Deff*t)))  + exp(u_eff*x_/Deff)*erfc((x_ + u_eff*t)/(2*sqrt(Deff*t))))

scatter!(t, analytical_sol)
plot(t, analytical_sol - Array(solution_other)[Int(n_variables), :]/cin)




