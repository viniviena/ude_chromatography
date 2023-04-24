# Efficient hybrid modeling and sorption model discovery for non-linear advection-diffusion-sorption systems: A systematic scientific machine learning approach

## Overview

This repository holds the code in Julia Language for the manuscript entitled as above and avaiable at https://arxiv.org/abs/2303.13555 (pre-print).

## Usage

For reproducing the results, you should point to 4 files:

- PDE_gradients_lux_mechanistic.jl: Holds the code where the in-silico data set is generated for 3 kinetic laws and 2 isotherms. It also shows the Taylor series analysis  for sparse/symbolic regression.
* You do not need to run it again as the training and test datas are saved in the directories **train_data/ and test_data/**

- PDE_gradients_lux_ude.jl: Holds the code used for training the hybrid UDE-based model. 
* You can find the weights of the trained ANNs in the directory trained_models/.
 
- PDE_gradients_lux_charts.jl: Separate file for building the plots using PGFPlots.jl library.
- UDE_missingterm.jl: File where the trained ANNs are regressed using both sparse and symbolic regression.






