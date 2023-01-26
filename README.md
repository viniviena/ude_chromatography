## This folder contains the code for UDE in column chromatography. A slightly different application of the /sandbox folder but very similar.


**The idea is replacing the mass transfer kinetics by a ANN and then run sparse regression on the identified ANN**


- PDE_gradients_lux.jl is the code for training UDE and carrying sparse regression on the ANN.
- PDE_gradients_lux_charts.jl is the code I use for plotting using PGFPlots.jl library
 
