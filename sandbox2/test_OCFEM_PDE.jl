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

n_elements = 200 # Number of finite elements
collocation_points = 2 #Collocation points
n_components = 1;  # 2 chemical species
n_phases = 1 #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ = -2.0
xₘₐₓ = 2.0 # z domain limits
h = (xₘₐₓ - xₘᵢₙ) / n_elements #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components) #make matrices for OCFEM

#Building mass matrix
MM = BitMatrix(Array(make_MM_2(n_elements, n_phases, n_components))) #make mass matrix #make matrices for OCFEM

#Problem
#∂u/∂t = D/L^2×∂²u/∂x² - ∂u/∂x/L   -2 < x < 2 → -0.5 < x/L (x*) < 0.5 
# u(x, t = 0) = exp(-x²)


#Calculating the derivative matrices stencil
y_dy = Array(A * H^-1) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = Array(B * H^-1) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y
U = Array([0.2113248654, 0.7886751346])


xmin = -2.0
xmax = 2.0
x = []
for i=1:n_elements
    append!(x, ones(2)*h*(i-1) + [U[1]*h + xmin, U[2]*h + xmin])
end
x = union(x);
x = [-2.0; x; 2.0]



# Initial condition
u₀ = zeros(n_variables) 
u₀[2:end - 1] .= exp.(-x[2:end - 1].^2)
u₀[1] = 1/sqrt(1 + 4*D*0)*exp(-(2.0 + 0)^2/(1 + 4*D*0)) 
u₀[end] = 1/sqrt(1 + 4*D*0)*exp(-(2.0 - 0)^2/(1 + 4*D*0))
t_test = 0.0:0.01:2.0
lb  = @. 1/sqrt(1 + 4*D*t_test)*exp(-(2.0 + t_test)^2/(1 + 4*D*t_test))
ub = @. 1/sqrt(1 + 4*D*t_test)*exp(-(2.0 - t_test)^2/(1 + 4*D*t_test))
plot!(t_test, ub)


#Params 
D = 2e-5
Pe = h/(2*D)
L = 2 - (-2)

#Stencil
rhs = D*y_dy2/(h^2) - y_dy/h

#ODE - inplace
function ode(dy, y, p, t)
dy[1] = y[1] - 1/sqrt(1 + 4*D*t)*exp(-(2.0 + t)^2/(1 + 4*D*t))    
dy[2:end - 1] = @view(rhs[2:end - 1, 1:end])*y
dy[end] = y[end] - 1/sqrt(1 + 4*D*t)*exp(-(2.0 - t)^2/(1 + 4*D*t))
end

#Solving
f_node = ODEFunction(ode, mass_matrix = MM)
tspan = (0.0, 2.0) 
prob_node = ODEProblem(f_node, u₀, tspan, 2)
ccall((:openblas_get_num_threads64_,Base.libblas_name), Cint, ())
@time solution = solve(prob_node, abstol = 1e-6, reltol = 1e-6, FBDF(autodiff = false), saveat = h);

#Plotting
u_sol = Array(solution)
a_sol = H^-1*u_sol
x_analitcal = -2.0:h:2.0 |> collect
idx = 2
analitcal_sol = @. 1/sqrt(1 + 4*D*solution.t)*exp(-(x_analitcal[idx] - solution.t)^2/(1 + 4*D*solution.t))
scatter(solution.t, analitcal_sol)
plot!(solution.t, a_sol[Int(idx*2 - 1), :], linewidth = 2.0)

