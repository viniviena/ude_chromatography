# Efficient hybrid modeling and sorption model discovery for non-linear advection-diffusion-sorption systems: A systematic scientific machine learning approach

## Overview
This repository holds the code in Julia Language for the manuscript entitled as above and avaiable at https://arxiv.org/abs/2303.13555 (pre-print).

## Abstract
This study presents a systematic machine learning approach for creating efficient hybrid models and discovering sorption uptake models in non-linear advection-diffusion-sorption systems. It demonstrates an effective method to train these complex systems using gradient based optimizers, adjoint sensitivity analysis, and JIT-compiled vector Jacobian products, combined with spatial discretization and adaptive integrators. Sparse and symbolic regression were employed to identify missing functions in the artificial neural network. The robustness of the proposed method was tested on an in-silico data set of noisy breakthrough curve observations of fixed-bed adsorption, resulting in a well-fitted hybrid model. The study successfully reconstructed sorption uptake kinetics using sparse and symbolic regression, and accurately predicted breakthrough curves using identified polynomials, highlighting the potential of the proposed framework for discovering sorption kinetic law structures. 

## Usage

For reproducing the results, you should point to 4 files:


- PDE_gradients_lux_mechanistic.jl: Holds the code where the in-silico data set is generated for 3 kinetic laws and 2 isotherms. It also shows the Taylor series analysis  for sparse/symbolic regression (You do not need to run it again as the training and test datas are saved in the directories **train_data/ and test_data/**)

- PDE_gradients_lux_ude.jl: Holds the code used for training the hybrid UDE-based model. (You can find the weights of the trained ANNs in the directory trained_models/.). You may notice that training loss do not drop monotonically with ADAM - it sometimes grow suddenly but them go back to the previous trend. 
 
- PDE_gradients_lux_charts.jl: Separate file for building the plots using PGFPlots.jl library.

- UDE_missingterm.jl: File where the trained ANNs are regressed using both sparse and symbolic regression.


## Citation

@misc{santana2023efficient,
      title={Efficient hybrid modeling and sorption model discovery for non-linear advection-diffusion-sorption systems: A systematic scientific machine learning approach}, 
      author={Vinicius V. Santana and Erbet Costa and Carine M. Rebello and Ana Mafalda Ribeiro and Chris Rackauckas and Idelfonso B. R. Nogueira},
      year={2023},
      eprint={2303.13555},
      archivePrefix={arXiv},
      primaryClass={cs.CE}
}

