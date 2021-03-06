# SOA Drive Signal Optimisation with Particle Swarm Optimisation

**Documentation**: https://soa-driving.readthedocs.io/

This project formed part of the journal paper 'An Artificial Intelligence Approach to Optimal Control of Sub-Nanosecond SOA-Based Optical Switches' (link will be provided when published). All data (https://doi.org/10.5522/04/12356696.v1) used in the paper for PSO was generated with this code. The PSO algorithm is used to optimise the driving signal for SOA switches both in simulation and experiment. Researchers should be able to download this project and blackbox the PSO algorithm to optimise a range of SOA transfer functions/simulation objects without needing to write any code (other than minor directory variable adjustments). 

## File Descriptions
- analyse.py: Module for analysing signal performances (e.g. rise time, settling time, overshoot etc.)
- devices.py: Module for interfacing with SOA experimental setup (not needed if only using transfer function simulations)
- distort_tf.py: Module for distorting the original SOA transfer function to generate new transfer functions and therefore simulate different SOAs (useful for testing algorithm generalisability)
- get_fopdt_params.py: Backend module for analyse.py
- optimisation.py: The main module that holds the PSO algorithm
- signalprocessing.py: Module for generating standard literature SOA drive signals (e.g. PISIC, MISIC etc.) and evaluating optical response cost (in terms of e.g. mean squared error)

## Getting Started


### Installing
Open Git Bash. Change the current working directory to the location where you want to clone this project, and run:
```
git clone https://github.com/cwfparsonson/soa_driving.git
```
In `soa_driving/pso`, use either conda or pip to install the required packages
```
conda install --file requirements/default.txt
```
```
pip install -r requirements/default.txt
```
In `soa_driving/pso`, make the soa module importable from anywhere on your machine:
```
python setup.py develop
```


### Running the Project
To get started, change the `directory` variable in optimisation.py to match your own directory for where you want PSO data to be saved, and run:
```
python optimisation.py
```
This should perform a quick PSO optimisation on different SOA transfer functions in parallel. The results will be tracked and printed in your console, and saved in your specified directory. To change the PSO hyperparameters and/or the default transfer function(s), open optimisation.py, scroll down to the bottom, and edit the code. Note that by default this will run your experiments in paralellel with multiprocessing rather than sequentially to save time. Edit the script to change this.

The below is an example of how to simulate 10 different SOAs and optimise each of them individually to demonstrate the generalisability of PSO to any SOA. 

```python
from soa import devices, signalprocessing, analyse, distort_tf
from soa.optimisation import PSO, run_test

import numpy as np
import multiprocessing
import pickle
from scipy import signal
import os
import matplotlib.pyplot as plt


# set dir to save data
directory = '../../data/'

# init basic params
num_points = 240
time_start = 0
time_stop = 20e-9
t = np.linspace(time_start, time_stop, num_points)

# set PSO params
n = 3 
iter_max = 150
rep_max = 1 
max_v_f = 0.05 
init_v_f = max_v_f 
cost_f = 'mSE' 
w_init = 0.9
w_final = 0.5
on_suppress_f = 2.0

# define initial drive signal
init_OP = np.zeros(num_points) # initial drive signal (e.g. a step)
init_OP[:int(0.25*num_points)],init_OP[int(0.25*num_points):] = -1, 0.5

# initial transfer function numerator and denominator coefficients
num = [2.01199757841099e85]
den = [
    1.64898505756825e0,
    4.56217233166632e10,
    3.04864287973918e21,
    4.76302109455371e31,
    1.70110870487715e42,
    1.36694076792557e52,
    2.81558045148153e62,
    9.16930673102975e71,
    1.68628748250276e81,
    2.40236028415562e90,
]
tf = signal.TransferFunction(num, den)
tfs, _ = distort_tf.gen_tfs(num_facs=[1.0,1.2,1.4], 
                        a0_facs=[0.8],
                        a1_facs=[0.7,0.8,1.2],
                        a2_facs=[1.05,1.1,1.2],
                        all_combos=False)


# get initial output of initial signal and use to generate a target set point
init_PV = distort_tf.getTransferFunctionOutput(tf,init_OP,t)
sp = analyse.ResponseMeasurements(init_PV, t).sp.sp

# run PSO tests in parallel with multiprocessing
pso_objs = multiprocessing.Manager().list()
jobs = []
test_nums = [test+1 for test in range(len(tfs))]
direcs = [directory + '/test_{}'.format(test_num) for test_num in test_nums]
for tf, direc in zip(tfs, direcs):
    if os.path.exists(direc) == False:
        os.mkdir(direc)
    p = multiprocessing.Process(target=run_test, 
                                args=(direc, 
                                      tf, 
                                      t, 
                                      init_OP, 
                                      n, 
                                      iter_max, 
                                      rep_max, 
                                      init_v_f, 
                                      max_v_f, 
                                      w_init, 
                                      w_final, 
                                      True, 
                                      'pisic_shape', 
                                      on_suppress_f, 
                                      True, 
                                      None, 
                                      cost_f, 
                                      None, 
                                      True, 
                                      True,
                                      sp, 
                                      pso_objs,))

    jobs.append(p)
    p.start()
for job in jobs:
    job.join()

# pickle PSO objects so can re-load later if needed
PIK = directory + '/pickle.dat'
data = pso_objs
with open(PIK, 'wb') as f:
    pickle.dump(data, f)
```

### Additional Notes
- As a general rule-of-thumn, increasing n (the number of particles) is a reliable way to improve PSO performance and find more optimal solutions.

- The are 2 primary modes of using this PSO implementation: (1) Simulation (using a transfer function that simulates SOAs), or (2) Experimental (applying the PSO algorithm to a real experimental setup). To use the experimental setup, you will need all the same equipment, modules, specific GPIB addresses etc. that were used in the Connet lab in UCL's EEE Robert's building (contact the Optical Networks Group for more info). Users outside of ONG will need to write code to interface with their own equipment. To use the simulation (i.e. the transfer function), the user should not need to write any code themselves. Simply changing the below 'directory' variable to point this programme to where to store data should be sufficient. 

- It is not required to use the `run_test()` function when running multiple simulations, but it is recommended since multiprocessing will run the PSOs in parallel and signficantly speed up your workflow. 

- While this code is not the 'cleanest', clear comments have been inserted so that anyone wishing to delve deeper into and use this PSO implementation (beyond running simple transfer function simulations) can follow the logic and implement the same functionality in their own programmes.


### Citing This Project
If you use this project or implement derivatives of it, please cite the paper at the top.













