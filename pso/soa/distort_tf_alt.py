from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import copy
from soa import upsampling


def find_x_init(tf):
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



def getTransferFunctionOutput(tf, U, T, q, atol=1e-12):
    """
    This method sends a drive signal to a transfer function model and gets the 
    output

    Args:
    - tf = transfer function
    - U = signal to drive transfer function with
    - T = array of time values
    - X0 = initial value
    - atol = scipy ode func parameter

    Returns:
    - PV = resultant output signal of transfer function
    """
    T = np.linspace(0, 20e-9, 240)
    X0 = find_x_init(tf)
    U = np.array(U)
    p = upsampling.ups(240)
    U = p.create(U)
    input_init = np.copy(U)

    PV = np.zeros((q, 240))
    
    for i in range(q):
        PV = np.array([])
        input = input_init[:40]
        input = p.create(input)

        (_, PV[i], X0_init) = signal.lsim2(tf, input, T, X0=X0, atol=atol)
        X0 = X0_init[-1] 
        input_init = input_init[40:]
        
    min_PV = np.copy(min(PV[-1]))
    if min_PV < 0:
        for j in range(0, len(PV[-1])):
            PV[-1][j] = PV[-1][j] + abs(min_PV)

    
    return PV