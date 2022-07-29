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
using Zygote



#parameters
Qf = 7.88e-3 #Feed flow rate (L/min)
d = 0.26  # Column diameter (dm3)
L = 1.15 # Column length (dm)
a = pi*d^2/4 #Column cross section area (dm2)
ε = 0.42 # void fraction
u = Qf/(a*epsilon) #dm/min (drif velocity)
V = a*L # tank volume (dm3)
k_transf = 0.001*3/245.5E-4

#-----#isotherm
Qₘ = 9.13;
K =  11.66;

function trueODEfunc(du, u, p, t)
#parameters
Cf = p

du[1] =  Qf/(ε*V)*(Cf - u[1]) - (1-ε)/ε*du[2]
du[2] = k_transf*(Qₘ*K*u[1]/(1 + K*u[1]) - u[2])

end

#Initial conditions
c0 = 0.0
q_u_0 = Qₘ*c0*K/(1 + c0*K)
u0 = [c0; q_u_0]

tspan = (0.0, 150.0)
prob_trueode = ODEProblem(trueODEfunc, u0, tspan, 1.0)

# Dataset size
datasize = 151
tsteps = range(tspan[1], tspan[2], length = datasize)

#callback
inputs  = [0.0 1.0; 100.0 2.0]
dosetimes = inputs[:,1]

function affect!(integrator)
    ind_t = findall(t -> t==integrator.t, dosetimes)
    integrator.p = inputs[ind_t[1], 2]
end
 
cb2 = PresetTimeCallback(dosetimes, affect!, save_positions = (false, false));

ode_data = Array(solve(prob_trueode, AutoVern9(KenCarp4(autodiff = false)),
                  saveat = tsteps, callback = cb2))[1, 1:end]

ode_data2 = [Qₘ*1.0*K/(1 + 1.0*K); Qₘ*2.0*K/(1 + 2.0*K)]

train_data = [ode_data, ode_data2]
#---------------neural ode

import Random
Random.seed!(10)
ann_node1 = FastChain(FastDense(1, 7, tanh), FastDense(7, 1)); 

#getting params
net_params1 = initial_params(ann_node1);

function _ann1(u, p)
    w1 = reshape(@view(p[1:1*7]), 7, 1)
    b1 = @view(p[1*7 + 1: 2*7])
    w2 = reshape(@view(p[2*7 + 1: 2*7 + 1*7]), 1, 7)
    b2 = @view(p[3*7 + 1:end])
    
    (w2*(tanh.(w1*u .+ b1)) .+ b2)
    
end

function dudt(du,u,p,t)
    Cf = p[1]
    q_star = _ann1(u[1], p[3:end])[1]
    du[1] =  Qf/(ε*V)*(Cf - u[1]) - (1-ε)/ε*du[2]
    du[2] = k_transf*(q_star - u[2])
end

#callback
inputs  = [0.0 1.0; 100.0 2.0]

function affect_node!(integrator)
    ind_t = findall(t -> t==integrator.t, dosetimes)
    integrator.p[1] = inputs[ind_t[1], 2]
end
 
cb_node = PresetTimeCallback(dosetimes, affect_node!, save_positions = (false, false));

θ = [1.0; _ann1(c0, net_params1); net_params1]
prob_ude = ODEProblem(dudt, u0, tspan, θ)

sol_node = Array(solve(prob_ude, AutoVern9(KenCarp4(autodiff = false)),
                  saveat = tsteps, callback = cb_node))

# training neural ode


function predict(θ)
u0 = [c0, θ[2]]

sensealg = sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP())

prob_ = remake(prob_ude, u0 = u0, tspan = tspan, p = θ)

sol_node = Array(solve(prob_ude, AutoVern9(KenCarp4(autodiff = false)), sensealg = sensealg, 
                  saveat = tsteps, callback = cb_node))
[sol_node[1, 1:end], sol_node[2, [101,end]]]

end


function loss(θ)
pred = predict(θ)
#data = data_train
sum(Flux.Losses.mse.(train_data[1:2], pred[1:2])) 
end

grad = Zygote.gradient(loss, θ)