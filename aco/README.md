# SOA Drive Signal Optimisation with Ant Colony Optimisation

This project formed part of the journal paper 'An Artificial Intelligence Approach to Optimal Control of Sub-Nanosecond SOA-Based Optical Switches' (link will be provided when published). All data (https://doi.org/10.5522/04/12356696.v1) used in the paper for ACO was generated with this code. The ACO algorithm is used to optimise the driving signal for SOA switches both in simulation and experiment. Researchers should be able to download this project and blackbox the ACO algorithm to optimise a range of SOA transfer functions/simulation objects without needing to write any code (other than minor directory variable adjustments).

## How to run

The file 'aco_optimise.py' should be able to run, provided that the required dependencies are installed.

The script saves a csv file containing the values of the metric (MSE by default) at each generation.
To reproduce results from the paper, do not change the value of the hyperparameters from what they are set as by defulault.