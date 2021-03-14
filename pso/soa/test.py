
from soa import upsampling
import numpy as np
import multiprocessing
import pickle
from scipy import signal
import os
import matplotlib.pyplot as plt

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

init_OP = np.zeros(40) # initial drive signal (e.g. a step)

init_OP[:int(0.25*40)],init_OP[int(0.25*40):] = -1, 0.5


init_OP = np.tile(init_OP, 3)

t2 = np.linspace(0, 20e-9, 240)

def __find_x_init(tf):
    """
    This method calculates the state-vector from a long -1 drive signal. 
    Must call before sending / receiving signals to / from transfer function 
    model
    Args:
    - tf = transfer function
    Returns:
    - X0 = system's state-vector result for steady state
    """
    U = np.array([-1.0] * 480)
    T = np.linspace(0, 40e-9, 480)
    (_, _, xout) = signal.lsim2(tf, U=U, T=T, X0=None, atol=1e-13)
    X0 = xout[-1]

    return X0
        

def __getTransferFunctionOutput(tf, U, T, X0, q, atol=1e-12):
    """
    This method sends a drive signal to a transfer function model and gets 
    the output
    Args:
    - tf = transfer function
    - U = signal to drive transfer function with
    - T = array of time values
    - X0 = initial value
    - atol = scipy ode func parameter
    Returns:
    - PV = resultant output signal of transfer function
    """

    

    T = np.linspace(T[0], T[-1], 240)

    U = np.array(U)
    sample = 240
    p = upsampling.ups(sample)
    input_init = np.copy(U)
    

    for _ in range(3):
        PV = np.array([])
        input = input_init[:40]
        input = p.create(input)

        
        (_, PV, X0_init) = signal.lsim2(tf, input, T, X0=X0, atol=atol)
        X0 = X0_init[0] 
        input_init = input_init[40:]
    
    min_PV = np.copy(min(PV))
    if min_PV < 0:
        for i in range(0, len(PV)):
            PV[i] = PV[i] + abs(min_PV)

    return PV

X0 = __find_x_init(tf)

PV = __getTransferFunctionOutput(tf, init_OP, t2, X0, 3)
t3 = np.linspace(0, 20e-9, 120)
plt.figure(1) 
plt.plot(t2, PV, c='b')
plt.show()