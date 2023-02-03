using Pkg
Pkg.activate(".")

using Plots
using TikzPictures
pgfplotsx()
using MAT
using DelimitedFiles

data = readdlm("sparse_reg_data/stats_lang_improved_kldf.csv", ',')

plot(randn(19))
savefig(fig, "test.pdf")
