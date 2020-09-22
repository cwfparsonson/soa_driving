# SOA Drive Signal Optimisation with Particle Swarm Optimisation

This project formed part of the journal paper 'An Artificial Intelligence Approach to Optimal Control of Sub-Nanosecond SOA-Based Optical Switches' (link will be provided when published). All data (https://doi.org/10.5522/04/12356696.v1) used in the paper for PSO was generated with this code. The PSO algorithm is used to optimise the driving signal for SOA switches both in simulation and experiment. Researchers should be able to download this project and blackbox the PSO algorithm to optimise a range of SOA transfer functions/simulation objects without needing to write any code (other than minor directory variable adjustments). 

## File Descriptions
- analyse.py: Module for analysing signal performances (e.g. rise time, settling time, overshoot etc.)
- devices.py: Module for interfacing with SOA experimental setup (not needed if only using transfer function simulations)
- distort_tf.py: Module for distorting the original SOA transfer function to generate new transfer functions and therefore simulate different SOAs (useful for testing algorithm generalisability)
- get_fopdt_params.py: Backend module for analyse.py
- optimisation.py: The main module that holds the PSO algorithm
- signalprocessing.py: Module for generating standard literature SOA drive signals (e.g. PISIC, MISIC etc.) and evaluating optical response cost (in terms of e.g. mean squared error)

## Getting Started

### Prerequisites
It is recommended to use the Anaconda environment manager to install package dependencies for this project. The following tools are prerequisites:
```
Python 3.6.0
numpy 1.15.4
pyvisa 1.10.1
matplotlib 3.1.2
pandas 0.25.3
scipy 1.1.0
```
### Installing
Open Git Bash. Change the current working directory to the location where you want to clone this project, and run:
```
git clone https://github.com/cwfparsonson/soa_driving.git
```
Use anaconda to install the required packages:
```
conda install <package>
```
