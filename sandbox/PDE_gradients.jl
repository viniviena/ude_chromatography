cd(@__DIR__)
using Pkg;
Pkg.activate(".")
Pkg.instantiate()

#Importing ODE, plot and MAT libraries
using OrdinaryDiffEq
using DiffEqFlux
#using DiffEqCallbacks
using DifferentialEquations
using Flux
using Plots
using MAT
using DelimitedFiles
using SciMLSensitivity
import Random
using PreallocationTools
using ForwardDiff, Zygote, FiniteDiff

# Script with auxiliary functions
include("../utils.jl")

#----------- Building OCFEM (orthogonal collocation on finite element method)
#for z discretization with cubic hermite polynomials-------------

n_elements = 20 # Number of finite elements
collocation_points = 2 #Collocation points
n_components = 2;  # 2 chemical species
n_phases = 2 #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components * n_phases * (p_order + 2 * n_elements - 2)
xₘᵢₙ = 0.0
xₘₐₓ = 1.0 # z domain limits
h = (xₘₐₓ - xₘᵢₙ) / n_elements #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components) #make matrices for OCFEM

#Building mass matrix
MM = Array(make_MM(n_elements, n_phases, n_components)) #make mass matrix

#-------- Defining PDE parameters------------

Qf = 7.88e-3 #Feed flow rate (L/min)
d = 0.26  # Column diameter (dm3)
L = 1.15 # Column length (dm)
a = pi * d^2 / 4 #Column cross section area (dm2)
epsilon = 0.42 # void fraction
u = Qf / (a * epsilon) #dm/min (drif velocity)

kappaa = 0.001 * 3 / 245.5E-4 # min-1 (not being used)
kappab = 0.003 * 3 / 245.5E-4 #min -1 (not being used)
params_ode = [11.66, 9.13, 5.08, 5.11, kappaa, kappab, 163.0, 0.42, 11.64, 0.95]

#Calculating the derivative matrices stencil
y_dy = Array(A * H^-1) # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = Array(B * H^-1) # y = H*a and d2y_dx2 = B*a = (B*H-1)*y

#--------Importing experimental data---------------
vars = matread("../Global_Poh_ProPro.mat")
exp_data = vars["Global_Poh_ProPro"] #liquid phase concentration measurements
t_exp = exp_data[1:end, end] #time records

q_exp_data = readdlm("../q_exp_new2.csv", ',', Float64) # solid phase concentration measurements

# -----Initializing Neural networks---------
import Random
Random.seed!(10)
ann_node1 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1)); #ANNₑ,₁
ann_node2 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1)); #ANNₑ,₂
kldf_nn1 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1, sigmoid), (x, a) -> x * 3) #ANNₘ,₁
kldf_nn2 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1, sigmoid), (x, a) -> x * 3) #ANNₘ,₂


#getting params
net_params1 = Float64.(initial_params(ann_node1));
net_params2 = Float64.(initial_params(ann_node2));
net_params3 = Float64.(initial_params(kldf_nn1));
net_params4 = Float64.(initial_params(kldf_nn2));

#Here I had to define my own activation functions because I've got error trying to use Flux with modelingtoolkit together.
function my_sigmoid(x)
    1 / (1 + exp(-x)) * 3
end

function my_gelu(x)
    0.5 * x * (1 + tanh.(sqrt(2 / pi) * (x .+ 0.044715 * x .^ 3)))
end

#Rebuilding the NNs because I had trouble making modelingtoolkit work with Flux
function _ann1(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7+1:3*7]
    w2 = reshape(p[3*7+1:3*7+1*7], 1, 7)
    b2 = p[4*7+1:end]

    (w2 * (tanh.(w1 * u .+ b1)) .+ b2)

end

function _ann2(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7+1:3*7]
    w2 = reshape(p[3*7+1:3*7+1*7], 1, 7)
    b2 = p[4*7+1:end]

    (w2 * (tanh.(w1 * u .+ b1)) .+ b2)

end

function _ann3(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7+1:3*7]
    w2 = reshape(p[3*7+1:3*7+1*7], 1, 7)
    b2 = p[4*7+1:end]

    my_sigmoid.((w2 * (tanh.(w1 * u .+ b1)) .+ b2))

end

function _ann4(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7+1:3*7]
    w2 = reshape(p[3*7+1:3*7+1*7], 1, 7)
    b2 = p[4*7+1:end]

    my_sigmoid.((w2 * (tanh.(w1 * u .+ b1)) .+ b2))

end


# ----- Building the actual PDE model--------

#building initial condition vector
#Here we had to build a function that recalculates the initial condition ->
#every time the ANNs weights are update due to imposed initial condition on qᵢ
#As shown above → qᵢ(t = 0, z) =  ANNₑ,ᵢ(c,θₑ,ᵢ)

# cache u0 to put down
#y0_cache = dualcache(ones(n_variables), 12)
#c0 = dualcache([13.230000, 0.00000], 12)

y0_cache = ones(n_variables);
c0 = [13.230000, 0.00000];

function y_initial(θ, y0_cache, c0)

    #var0 = get_tmp(y0_cache, p)
    #c0 = get_tmp(c0, p)
    var0 = y0_cache[:]
    j = 0

    for i = 1:n_components
        #Internal node equations
        cl_idx = 2 + j
        cu_idx = p_order + 2 * n_elements - 3 + j


        cbl_idx = j + 1
        cbu_idx = j + p_order + 2 * n_elements - 2

        #Liquid phase residual
        var0[cl_idx:cu_idx] = ones(cu_idx - cl_idx + 1) * c0[i]

        #Boundary node equations
        var0[cbl_idx] = c0[i]

        var0[cbu_idx] = c0[i]

        j = j + p_order + 2 * n_elements - 2
    end

    x1 = (var0[2+0-1:p_order+2*n_elements-3+0+1] .- 6.0) ./ (13.0 - 6.0)
    x2 = (var0[2+(p_order+2*n_elements-2)-1:p_order+2*n_elements-3+(p_order+2*n_elements-2)+1] .- 0.047) ./ (4.0 - 0.047)
    p11 = @view θ[1+2:29+2]
    p22 = @view θ[30+2:60]
    q_star1 = _ann1([x1 x2]', p11)
    q_star2 = _ann2([x1 x2]', p22)
    q_star = [q_star1; q_star2]


    j = 0

    for i = 1:n_components

        ql_idx2 = 2 * (p_order + 2 * n_elements - 2) + 2 + j - 1
        qu_idx2 = p_order + 2 * n_elements - 3 + 2 * (p_order + 2 * n_elements - 2) + j + 1

        #Solid phase residual
        var0[ql_idx2:qu_idx2] = q_star[i, :]

        j = j + p_order + 2 * n_elements - 2

    end
    var0
end

# building rhs function for DAE solver

struct col_model_node{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10}
n_variables::T1
n_elements::T2
p_order::T3
L::T4
h::T5
u::T6
y_dy::T7
y_dy2::T8
Pe::T9
eps::T10
end

using TimerOutputs
using UnPack


tmr = TimerOutput();

function (f::col_model_node)(yp, y, p, t)
   #Aliasing parameters

   @unpack n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, Pe, eps = f 
   

   dy_du = [dot((@view y_dy[i, :]),y) for i in 1:Int(n_variables)] # ∂y/∂u where u is the local spatial coordinate
   dy2_du = [dot((@view y_dy2[i, :]),y) for i in 1:Int(n_variables)] # ∂²y/∂u² where u is the local spatial coordinate

   j = 0

   #---------------------Mass Transfer and equilibrium -----------------

   x1 = ((@view y[2+0-1:p_order+2*n_elements-3+0+1]) .- 6.0) ./ (13.0 - 6.0) #Scaling dependent variables
   x2 = ((@view y[2+(p_order + 2*n_elements - 2)-1: p_order + 2*n_elements - 3 + (p_order+2*n_elements-2) + 1]) .- 0.047) ./ (4.0 - 0.047) #scaling dependent variables
   p1 = @view p[1+2:29+2]
   p2 = @view p[30+2:60]
   p3 = @view p[61:61+28]
   p4 = @view p[61+28+1:end]
   q_star = [_ann1([x1 x2]', p1); _ann2([x1 x2]', p2)]
   K_transf_empirical = [_ann3([x1 x2]', p3); _ann4([x1 x2]', p4)]

   #-------------------------------mass balance -----------------

   for i = 1:n_components
       #Internal node equations
       cl_idx = 2 + j
       cu_idx = p_order + 2 * n_elements - 3 + j

       ql_idx = 2 * (p_order + 2 * n_elements - 2) + 2 + j
       qu_idx = p_order + 2 * n_elements - 3 + 2 * (p_order + 2 * n_elements - 2) + j

       ql_idx2 = 2 * (p_order + 2 * n_elements - 2) + 2 + j - 1
       qu_idx2 = p_order + 2 * n_elements - 3 + 2 * (p_order + 2 * n_elements - 2) + j + 1

       cbl_idx = j + 1
       cbu_idx = j + p_order + 2 * n_elements - 2

       #Liquid phase residual

       yp[cl_idx:cu_idx] = -(1 - eps) / eps * K_transf_empirical[i, 2:end-1] .* (q_star[i, 2:end-1] .- y[ql_idx:qu_idx]) .- dy_du[cl_idx:cu_idx] / h / (L / u) .+ 1 / Pe * dy2_du[cl_idx:cu_idx] / (h^2) / (L / u)


       #Solid phase residual

       yp[ql_idx2:qu_idx2] = K_transf_empirical[i, 1:end] .* (q_star[i, :] .- y[ql_idx2:qu_idx2])


       #Boundary node equations
       yp[cbl_idx] = dy_du[cbl_idx] / h .- Pe * (y[cbl_idx] .- p[i])

       yp[cbu_idx] = dy_du[cbu_idx] / h

       j = j + p_order + 2 * n_elements - 2
   end
   nothing
end


#Setting up callback

inputs = Float64[0.0 11.64 0.9538; 30.0 8.8 2.53; 58.0 6.23 3.954; 86.0 3.6715 5.377; 116.0 1.333 6.674] #t_injection, C₁, C₂
dosetimes = inputs[:, 1]

function affect!(integrator)
    ind_t = findall(t -> t == integrator.t, dosetimes)
    integrator.p[1] = inputs[ind_t[1], 2]
    integrator.p[2] = inputs[ind_t[1], 3]
end

cb2 = PresetTimeCallback(dosetimes, affect!, save_positions=(false, false))

# Building ODE problem
rhs = col_model_node(n_variables, n_elements, p_order, L, h, u, y_dy, y_dy2, params_ode[7], params_ode[8]);

f_node = ODEFunction(rhs, mass_matrix = MM)

tspan = (0.0, 147.8266667)

parameters = [11.64; 0.95; net_params1; net_params2; net_params3; net_params4] #injection concentration augumented with ANN params)

y0 = y_initial(parameters, y0_cache, c0)

prob_node = ODEProblem(f_node, y0, tspan, parameters)

@time solution = solve(prob_node, FBDF(autodiff=false), callback=cb2, saveat = t_exp[1:204]); #0.27 seconds after compiling

#Jacobian sparsity

using Symbolics

du0 = copy(y0)
jac_sparsity = Symbolics.jacobian_sparsity((du,u)-> rhs(du,u, [11.64; 0.95; net_params1; net_params2; net_params3; net_params4], 0.0), du0, y0)
f_node_jac = ODEFunction(rhs, mass_matrix = MM, jac_prototype = float.(jac_sparsity));
prob_node_jac = ODEProblem(f_node_jac, y0, tspan, p)

BLAS.set_num_threads(1)


#--------- Training Neural Network ----------

qa_index = Int(n_variables / 4 * 2 + 1):Int(n_variables / 4 * 3) #indices for taking q₁
qb_index = Int(n_variables / 4 * 3 + 1):n_variables #indices for taking q₂

#Prediction function

function predict(θ)
    #------------------------Initial condition---------------------------------------
    #y0 = y_initial(θ, (y0_cache, c0)) # As mentioned, I have to update initial conditions at every θ update
    y0 = y_initial(θ, y0_cache, c0);
    # --------------------------Sensealg---------------------------------------------
    sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())
    #----------------------------Problem solution-------------------------------------
    abstol = 1e-5
    reltol = 1e-5
    tspan = (0.0, 147.8266667)
    prob_ = remake(prob_node, u0=y0, tspan=tspan, p=θ)
    s_new = Array(solve(prob_, FBDF(autodiff=false), callback=cb2,
        saveat=t_exp[1:204], sensealg = sensealg))
    #----------------------------Output---------------------------------------------
    # The outputs are composed by the predictions of cᵢ (all times) and qᵢ (at injection times)

    [s_new[Int(n_variables / 4), 1:end], s_new[Int(n_variables / 2), 1:end],
        s_new[qa_index, [44, 84, 124, 163, 204]], s_new[qb_index, [44, 84, 124, 163, 204]]]
end

#Setting up training data
data_train = [exp_data[1:204, 1], exp_data[1:204, 2],
    repeat(q_exp_data[:, 2]', Int(n_variables / 4)), repeat(q_exp_data[:, 4]', Int(n_variables / 4))]

# Setting up loss function for using with galactic
function loss_dae(θ)
    pred = predict(θ)
    #data = data_train
    sum(Flux.Losses.mse.(data_train[1:4], pred[1:4])), pred
end

#testing loss
@time losvt, pred_123 = loss_dae([11.64; 0.95; net_params1; net_params2; net_params3; net_params4])
println("Testing loss", losvt)


#Here I set up another loss function to use in a home made Adam, because I was not able to use SciML or flux to train it.

loss(θ) = sum(Flux.Losses.mse.(data_train[1:4], predict(θ)[1:4]))
function loss_1(θ)
prediction =  predict(θ)
loss1 = sum(abs2, reshape(prediction[1] -  data_train[1], :, 1))
loss2 = sum(abs2, reshape(prediction[2] -  data_train[2], :, 1))
loss3 = sum(abs2, reshape(prediction[3] -  data_train[3], :, 1))
loss4 = sum(abs2, reshape(prediction[4] -  data_train[4], :, 1))
loss1 + loss2 + loss3 + loss4 

end



θ = [11.64; 0.95; net_params1; net_params2; net_params3; net_params4]
loss_1(θ)
@time loss(θ)
grad = Zygote.gradient(loss, θ)

@show t_exp[1:204]
