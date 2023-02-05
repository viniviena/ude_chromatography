using Pkg
Pkg.activate(".")
Pkg.instantiate()

#Importing ODE, plot and MAT libraries

using Plots
import Random
using StatsBase
using LinearAlgebra
using LinearSolve

# Script with auxiliary functions
include("utils.jl")

n_elements = 20 # Number of finite elements
collocation_points = 2 #Collocation points
n_components = 1;  # 2 chemical species
n_phases = 1 #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ = 0.0e0
xₘₐₓ = 1.0e0 # z domain limits
h = (xₘₐₓ - xₘᵢₙ) / n_elements #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components) #make matrices for OCFEM

H = Array(H);
A = Array(A);
B = Array(B);

rhs = zeros(n_variables)
rhs[1] = 1
lhs = zeros(n_variables, n_variables)
α_2 = -3.0^2

lhs[2:end - 1, 1:end ] .= B[2:end - 1, 1:end ]/(h^2) + α_2*H[2:end - 1, 1:end]
lhs[1, 1] = 1.0
lhs[end, end] = 1.0

prob = LinearProblem(lhs, rhs)
sol = solve(prob)
u_prime = lhs\rhs

y = H*sol.u
y_prime = H*u_prime
x = 0.0:h:1.0 |> collect
analytical = cosh.(3.0*(1.0 .- x))./cosh(3.0)
plot(0.0:h:1.0, sol.u[1:2:end], label = "numerical")
scatter!(x, analytical, label = "analytical")
plot(x, analytical - sol.u[1:2:end])