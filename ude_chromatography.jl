##--------------------Model----------------------------------------------
# ----- PDEs--------
# ∂cᵢ/∂t = (v÷(Pe×L))x∂²cᵢ/∂z² - (v/L)×∂cᵢ/∂z - ((1-ε)/ε)×(3/Rₚ)×ANNₘ,ᵢ(c,θₘ,ᵢ)×(ANNₑ,ᵢ(c,θₑ,ᵢ) - qᵢ)
# ∂qᵢ/∂t = 3/Rₚ×ANNₘ,ᵢ(c,θₘ)×(ANNₑ,ᵢ(c,θₑ,ᵢ) - qᵢ)

#ANN are the neural networks that will learn both equilibrium and mass transfer coefficient.

#------bcs z = 1
# at z = 0 → ∂cᵢ/∂z = Pe×(cᵢ - f(t)) # f is the injection function
# at z = 1 → ∂cᵢ/∂z  = 0
#----- ics
# at t = 0
# cᵢ(t = 0, z) = c⁰ (constant)
# qᵢ(t = 0, z) = ANNₑ,ᵢ(c,θₑ,ᵢ) calculated by ANN
#----- parameters
# Pe → Peclet number, ε → void fraction, v → drift velocity, Rₚ → solid phase pellet radius
# L → column length 
#---------------------------------------------------------------------

# Script with auxiliary functions
include("utils.jl")

#----------- Building OCFEM (orthogonal collocation on finite element method)
#for z discretization with cubic hermite polynomials-------------

n_elements = 20 # Number of finite elements
collocation_points = 2; #Collocation points
n_components = 2;  # 2 chemical species
n_phases = 2; #2 phases → 1 liquid + 1 solid
p_order = 4 #Polynomial order + 1
n_variables = n_components*n_phases*(p_order + 2*n_elements-2);
xₘᵢₙ = 0.0; xₘₐₓ = 1.0 # z domain limits
h = (xₘₐₓ -  xₘᵢₙ)/n_elements; #finite elements' sizes

H, A, B = make_OCFEM(n_elements, n_phases, n_components); #make matrices for OCFEM

#Building mass matrix
MM = make_MM(n_elements,n_phases, n_components); #make mass matrix

#-------- Defining PDE parameters------------

Qf = 7.88e-3 #Feed flow rate (L/min)
d = 0.26  # Column diameter (dm3)
L = 1.15 # Column length (dm)
a = pi*d^2/4 #Column cross section area (dm2)
epsilon = 0.42 # void fraction
u = Qf/(a*epsilon) #dm/min (drif velocity)

kappaa = 0.001*3/245.5E-4 # min-1 (not being used)
kappab = 0.003*3/245.5E-4 #min -1 (not being used)
params_ode = [11.66, 9.13, 5.08, 5.11, kappaa, kappab, 163., 0.42, 11.64, 0.95];

#Calculating the derivative matrices stencil
y_dy = A*H^-1; # y = H*a and dy_dx = A*a = (A*H-1)*y
y_dy2 = B*H^-1; # y = H*a and d2y_dx2 = B*a = (B*H-1)*y


#Importing ODE, plot and MAT libraries
using DiffEqSensitivity
using OrdinaryDiffEq
using DiffEqFlux
using DiffEqCallbacks
using Flux
using Plots
using MAT
using DelimitedFiles
using SciMLSensitivity


#--------Importing experimental data---------------
vars = matread("Global_Poh_ProPro.mat");
exp_data = vars["Global_Poh_ProPro"]; #liquid phase concentration measurements
t_exp = exp_data[1:end, end]; #time records

q_exp_data = readdlm("q_exp_new2.csv", ',', Float64); # solid phase concentration measurements


# -----Initializing Neural networks---------
import Random
Random.seed!(10)
ann_node1 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1)); #ANNₑ,₁
ann_node2 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1)); #ANNₑ,₂
kldf_nn1 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1, sigmoid),(x, a) -> x*3) #ANNₘ,₁
kldf_nn2 = FastChain(FastDense(2, 7, tanh), FastDense(7, 1, sigmoid),(x, a) -> x*3) #ANNₘ,₂


#getting params
net_params1 = Float64.(initial_params(ann_node1));
net_params2 = Float64.(initial_params(ann_node2));
net_params3 = Float64.(initial_params(kldf_nn1));
net_params4 = Float64.(initial_params(kldf_nn2));

#Here I had to define my own activation functions because I've got error trying to use Flux with modelingtoolkit together.
function my_sigmoid(x)
    1/(1 + exp(-x))*3   
    end

function my_gelu(x)
0.5*x*(1+tanh.(sqrt(2/pi)*(x .+ 0.044715*x.^3)))   
end

#Rebuilding the NNs because I had trouble making modelingtoolkit work with Flux
function _ann1(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7 + 1: 3*7]
    w2 = reshape(p[3*7 + 1: 3*7 + 1*7], 1, 7)
    b2 = p[4*7 + 1:end]
    
    (w2*(tanh.(w1*u .+ b1)) .+ b2)
    
end

function _ann2(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7 + 1: 3*7]
    w2 = reshape(p[3*7 + 1: 3*7 + 1*7], 1, 7)
    b2 = p[4*7 + 1:end]
    
    (w2*(tanh.(w1*u .+ b1)) .+ b2)
    
end

function _ann3(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7 + 1: 3*7]
    w2 = reshape(p[3*7 + 1: 3*7 + 1*7], 1, 7)
    b2 = p[4*7 + 1:end]
    
    my_sigmoid.((w2*(tanh.(w1*u .+ b1)) .+ b2))
    
end

function _ann4(u, p)
    w1 = reshape(p[1:2*7], 7, 2)
    b1 = p[2*7 + 1: 3*7]
    w2 = reshape(p[3*7 + 1: 3*7 + 1*7], 1, 7)
    b2 = p[4*7 + 1:end]
    
    my_sigmoid.((w2*(tanh.(w1*u .+ b1)) .+ b2))
    
end

# ----- Building the actual PDE model--------

#building initial condition vector
#Here we had to build a function that recalculates the initial condition ->
#every time the ANNs weights are update due to imposed initial condition on qᵢ
#As shown above → qᵢ(t = 0, z) =  ANNₑ,ᵢ(c,θₑ,ᵢ)

using PreallocationTools

# cache u0 to put down
u0 = dualcache(ones(n_variables), 12);
c0 = dualcache([13.230000, 0.00000], 12);

function y_initial(p, (u0, c0))
    
    var0 = get_tmp(u0, p)
    c0 = get_tmp(c0, p)
    j = 0

    for i = 1:n_components
        #Internal node equations
        cl_idx = 2 + j
        cu_idx = p_order + 2*n_elements - 3 + j


        cbl_idx = j + 1
        cbu_idx = j + p_order + 2*n_elements - 2

        #Liquid phase residual
        var0[cl_idx:cu_idx] = ones(cu_idx-cl_idx + 1)*c0[i]

        #Boundary node equations
        var0[cbl_idx] = c0[i]

        var0[cbu_idx] = c0[i]

        j = j + p_order + 2*n_elements - 2
    end

    x1 =  (var0[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1].- 6.)./(13. - 6.)
    x2 =  (var0[2 + (p_order + 2*n_elements - 2) - 1:p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1] .- 0.047)./(4. - 0.047)
    p11 = @view p[1 + 2:29 + 2]
    p22 = @view p[30 + 2:60]
    q_star1 = _ann1([x1 x2]', p11)
    q_star2 = _ann2([x1 x2]', p22)
    q_star = [q_star1; q_star2]


    j=0

    for i = 1:n_components

        ql_idx2 = 2*(p_order + 2*n_elements - 2) + 2 + j - 1
        qu_idx2 = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j + 1

        #Solid phase residual
        var0[ql_idx2:qu_idx2] = q_star[i, :]

        j = j + p_order + 2*n_elements - 2

    end
    var0
end

# building rhs function for DAE solver

function col_model_node!(yp,y,p,t)
    #Aliasing parameters
    Pe, eps = params_ode[7:8]

    dy_du = [dot(y_dy[i, :], y) for i in 1:Int(n_variables)] # ∂y/∂u where u is the local spatial coordinate
    dy2_du = [dot(y_dy2[i, :], y)  for i in 1:Int(n_variables)] # ∂²y/∂u² where u is the local spatial coordinate
    
    j = 0

    #---------------------Mass Transfer and equilibrium -----------------
    
    x1 =  (y[2 + 0 - 1:p_order + 2*n_elements - 3 + 0 + 1].- 6.)./(13. - 6.) #Scaling dependent variables 
    x2 =  (y[2 + (p_order + 2*n_elements - 2) - 1:p_order + 2*n_elements - 3 + (p_order + 2*n_elements - 2) + 1] .- 0.047)./(4. - 0.047) #scaling dependent variables
    p1 = @view p[1 + 2:29 + 2]
    p2 = @view p[30 + 2:60]
    p3 = @view p[61:61 + 28]
    p4 = @view p[61 + 28 + 1:end]
    #q_star1 = _ann1([x1 x2]', p1)
    #q_star2 = _ann2([x1 x2]', p2)
    #K_transf_empirical1 = _ann3([x1 x2]', p3)
    #K_transf_empirical2 = _ann4([x1 x2]', p4)
    q_star = [_ann1([x1 x2]', p1); _ann2([x1 x2]', p2)]
    K_transf_empirical = [_ann3([x1 x2]', p3); _ann4([x1 x2]', p4)]
    
  

    #-------------------------------mass balance -----------------
        
    for i = 1:n_components
        #Internal node equations
        cl_idx = 2 + j
        cu_idx = p_order + 2*n_elements - 3 + j

        ql_idx = 2*(p_order + 2*n_elements - 2) + 2 + j
        qu_idx = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j

        ql_idx2 = 2*(p_order + 2*n_elements - 2) + 2 + j - 1
        qu_idx2 = p_order + 2*n_elements - 3 + 2*(p_order + 2*n_elements - 2) + j + 1

        cbl_idx = j + 1
        cbu_idx = j + p_order + 2*n_elements - 2

        #Liquid phase residual
    
        @. yp[cl_idx:cu_idx] = -(1-eps)/eps*K_transf_empirical[i, 2:end-1].*(q_star[i, 2:end-1] .- y[ql_idx:qu_idx]) .- dy_du[cl_idx:cu_idx]/h/(L/u) .+ 1/Pe*dy2_du[cl_idx:cu_idx]/(h^2)/(L/u)
        

        #Solid phase residual
        
        @. yp[ql_idx2:qu_idx2] =  K_transf_empirical[i, 1:end].*(q_star[i, :] .- y[ql_idx2:qu_idx2])
   
        
        #Boundary node equations
         yp[cbl_idx] = dy_du[cbl_idx]/h .- Pe*(y[cbl_idx] .- p[i])

        yp[cbu_idx] = dy_du[cbu_idx]/h

        j = j + p_order + 2*n_elements - 2
    end
    nothing
end


#Setting up callback

inputs  = Float64[0.0 11.64 0.9538; 30.0 8.8 2.53; 58. 6.23 3.954; 86. 3.6715 5.377; 116. 1.333 6.674] #t_injection, C₁, C₂
dosetimes = inputs[:,1]

function affect!(integrator)
    ind_t = findall(t -> t==integrator.t, dosetimes)
    integrator.p[1] = inputs[ind_t[1], 2]
    integrator.p[2] = inputs[ind_t[1], 3]
end
 
cb2 = PresetTimeCallback(dosetimes, affect!, save_positions = (false, false));

# Building ODE problem
f_node = ODEFunction(col_model_node!, mass_matrix = MM);
tspan = (0.0f0, 147.8266667f0)
p = [11.64; 0.95; net_params1; net_params2; net_params3; net_params4] #injection concentration augumented with ANN params
y0 = y_initial(p, (u0, c0))
prob_node =  ODEProblem(f_node, y0, tspan, p);


#Moving to modelingtoolkit
using ModelingToolkit
f_optim = modelingtoolkitize(prob_node); #takes a few minutes
prob_optim = ODEProblem(f_optim, y0, tspan, p = [11.64; 0.95; net_params1; net_params2; net_params3; net_params4],
    jac = true, sparse = false); #takes a few minutes

BLAS.set_num_threads(1) # If I don't do it, gives me stackoverflow error

#testing solution time
jacvec = DiffEqSensitivity.EnzymeVJP()
#jacvec = ReverseDiffVJP(true)
#sensealg = DiffEqSensitivity.InterpolatingAdjoint(autojacvec = jacvec, checkpointing = true)
sensealg = DiffEqSensitivity.QuadratureAdjoint(autojacvec = jacvec)
@time solution = Array(solve(prob_optim, Rodas5(autodiff = false), callback = cb2, 
abstol = 1e-5, reltol = 1e-5, sensealg = sensealg ,
saveat = t_exp[1:204])) #0.27 seconds after compiling

#--------- Training Neural Network ----------

qa_index = Int(n_variables/4*2 + 1):Int(n_variables/4*3) #indices for taking q₁  
qb_index = Int(n_variables/4*3 + 1):n_variables #indices for taking q₂

#Prediction function

function predict(θ)
    #------------------------Initial condition---------------------------------------
    y0 = y_initial(θ, (u0, c0)) # As mentioned, I have to update initial conditions at every θ update

    # --------------------------Sensealg---------------------------------------------
    #sensealg = DiffEqSensitivity.InterpolatingAdjoint(autojacvec = DiffEqSensitivity.ReverseDiffVJP(true), checkpointing = true)
    #sensealg = EnzymeVJP()
    #sensealg = InterpolatingAdjoint(autojacvec = ZygoteVJP(), checkpointing = true)
    sensealg = DiffEqSensitivity.QuadratureAdjoint(autojacvec = DiffEqSensitivity.ZygoteVJP())
    #sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP())
    #sensealg = ForwardDiffSensitivity()
    #sensealg = ForwardSensitivity()
    #----------------------------Problem solution-------------------------------------
    abstol = 1e-5
    reltol = 1e-5
    prob_ = remake(prob_optim, u0 = y0, tspan = tspan, p = θ)
    s_new = Array(solve(prob_, Rodas5(autodiff = false), callback = cb2, sensealg = sensealg,
            saveat = t_exp[1:end], abstol = abstol, reltol=reltol))
    #----------------------------Output---------------------------------------------
    # The outputs are composed by the predictions of cᵢ (all times) and qᵢ (at injection times)

    [s_new[Int(n_variables/4), 1:end], s_new[Int(n_variables/2), 1:end], 
        s_new[qa_index, [44, 84, 124, 163, 204]], s_new[qb_index, [44, 84, 124, 163, 204]]]
end

#Setting up training data
data_train = [exp_data[1:204, 1], exp_data[1:204, 2], 
    repeat(q_exp_data[:, 2]', Int(n_variables/4)) , repeat(q_exp_data[:, 4]', Int(n_variables/4))];

# Setting up loss function for using with galactic
function loss_dae(θ)
    pred = predict(θ)
    #data = data_train
    sum(Flux.Losses.mse.(data_train[1:4], pred[1:4])), pred
end

#testing loss
@time losvt, pred_123 = loss_dae([11.64; 0.95; net_params1; net_params2; net_params3; net_params4]);
println("Testing loss", losvt)


#Here I set up another loss function to use in a home made Adam, because I was not able to use SciML or flux to train it.

loss(θ) = sum(Flux.Losses.mse.(data_train[1:4], predict(θ)[1:4]))

using ForwardDiff, Zygote, Enzyme
#using Enzyme

#testing gradient calculation (takes a lot of time and almost 10gb of ram peak, Zygote do not work)
#@time grad = Enzyme.autodiff(Enzyme.ReverseMode, loss, Active, Active([11.64; 0.95; net_params1; net_params2; net_params3; net_params4]));

θ = [11.64; 0.95; net_params1; net_params2; net_params3; net_params4]

loss(θ)
grad = Zygote.gradient(loss, θ)

#I made my own adam
function adam(loss, θ, n_iter, α, β1, β2, ϵ = 1e-8)
    losses = []
    
    #Initialize momentum
        
    m = zeros(size(θ))
    v = zeros(size(θ))
    m̂ = zeros(size(θ))
    v̂ = zeros(size(θ))    
        for t = 1:n_iter
        grad = ForwardDiff.gradient(loss, θ) #The only I could use
        @. m = β1*m + (1-β1)*grad
        @. v = β2*v + (1-β2)*grad^2
        @. m̂ = m/(1-β1^t)
        @. v̂ = v/(1-β2^t)
        lr = (α[1]-α[2])*(1 - (t - 1)/n_iter)^1.0 + α[2] 
        @. θ = θ - lr*m̂/(sqrt(v̂) + ϵ)
        push!(losses, loss(θ))
        if length(losses)%20==0
            println(losses[end])
        end
        end
        return losses, θ      
    end


@time loss_test, params_test = adam(loss, [11.64; 0.95; net_params1; net_params2; net_params3; net_params4], 500, [0.01, 0.008], 0.9, 0.999);