## This folder contains the code for UDE in column chromatography. A slightly different application of the /sandbox folder but very similar.


**The idea is replacing the mass transfer kinetics by a ANN and then run sparse regression on the identified ANN**

- PDE_gradients_lux.jl is the code for data generation.
- PDE_gradients_lux.jl is the code for training UDE
- PDE_gradients_lux_charts.jl is the code I use for plotting using PGFPlots.jl library
- UDE_missingter.jl is the code for sparse/symbolic regression on ANNs' output
- /train_data has the training data files
- /test_data has the test data files
- /plots has the plots for training, test and ANN predictions

