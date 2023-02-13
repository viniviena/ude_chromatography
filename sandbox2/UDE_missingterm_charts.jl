using Pkg
Pkg.activate(".")

using PGFPlots
using DelimitedFiles

data = readdlm("sparse_reg_data/stats_improved_quad_lang.csv", ',')

fig2 = GroupPlot(2, 1, groupStyle = "horizontal sep = 2.0cm, vertical sep = 2.0cm");
push!(fig2, Axis([Plots.Linear(data[1:end, 2], data[1:end, 1], style = "blue")],
        legendPos="south east", xmode = "log", xlabel = L"\lambda", ylabel = "Bayesian Information Criterion"))

push!(fig2, Axis([Plots.Linear(data[1:end, 2], data[1:end, 3], style = "blue, mark = square*")],
legendPos="south east", xmode = "log", xlabel = L"\lambda", ylabel = "Number of active terms"))
#= push!(fig, Axis([Plots.Linear(data[1:end, 2], data[1:end, end], style = "blue, mark = triangle*")],
legendPos="south east", xmode = "log", xlabel = L"\lambda", ylabel = "Mean squared error", ymode =  "log")) =#
save("sparse_reg_data/improved_quad_lang.pdf", fig2)
