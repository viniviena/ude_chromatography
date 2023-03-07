using HTTP, CSV, DataFrames
X_url = "https://raw.githubusercontent.com/viniviena/repositoryname/master/Xs.csv"
Y_url = "https://raw.githubusercontent.com/viniviena/repositoryname/master/Ys.csv"
X_csv = CSV.File(HTTP.get(X_url).body, header = false)
Y_csv = CSV.File(HTTP.get(Y_url).body, header = false)
X = DataFrame(X_csv) |> Array |> transpose
Y = DataFrame(Y_csv) |> Array |> transpose

using DataDrivenSparse
using ModelingToolkit
using DataDrivenDiffEq
using DataDrivenSR
using SymbolicRegression
using UnPack
using StableRNGs
using Plots
gr()

@variables q_ast, q

polys = []
for i ∈ 0:6, j ∈ 0:6
    poli2 = q_ast^i * q^j
    push!(polys, poli2)
end

h__f = [unique(polys)...]
basis = Basis(h__f, [q_ast, q])

#Defining regression problem
problem_regression = DirectDataDrivenProblem(X, Y)
plot(problem_regression)

#Sparse regression
options = DataDrivenCommonOptions(
    maxiters = 500, normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 3,
    data_processing = DataProcessing(split = 0.95, batchsize = Int(round((size(X, 2)/10))), 
    shuffle = true, rng = StableRNG(1112)))

λ = exp10.(-3.0:0.05:0.5)
opt2 = ADMM(λ) # λ < exp10(-1.35) gives error
res = solve(problem_regression, basis, opt2, options = options)
println(res)
system = get_basis(res)
println(system)
maps = get_parameter_map(system)
bic(res)

#Symbolic Regression
eqsearch_options = SymbolicRegression.Options(binary_operators = [+],
                                              loss = L2DistLoss(),
                                              verbosity = 1, progress = true, npop = 30,
                                              timeout_in_seconds = 80.0)

alg_SR = EQSearch(eq_options = eqsearch_options)

options_SR = DataDrivenCommonOptions(
    maxiters = 100, normalize = DataNormalization(ZScoreTransform),
    selector = bic, digits = 3,
    data_processing = DataProcessing(split = 0.995, batchsize = Int(round((size(X, 2)/10))), 
    shuffle = true, rng = StableRNG(1112)))


res_SR = solve(problem_regression, basis, alg_SR, options = options_SR)
println(res_SR)
system_SR = get_basis(res_SR)
println(system_SR)
maps = get_parameter_map(system_SR)
bic(res_SR)

