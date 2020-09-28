import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import arange
from SignalPerformanceValues import *
from distort_tf import find_x_init

def getOPGraph(numberOfPoints,resolution,upperLimit,lowerLimit):
    """Return graph describing the point-by-point value selection to generate a driving signal.
    Arguments:
    numberOfPoints: number of points on the signal to be defined
    resolution: number of different values that each point on the signal can take
    upperLimit/lowerLimit: upper/lower limit of the range of values that can be taken by points on the signal.
    """
    tmp = np.linspace(lowerLimit,upperLimit,resolution)
    graph = np.zeros((numberOfPoints*resolution,numberOfPoints*resolution))

    for i in arange(0,len(graph)-resolution,resolution):
        for j in range(i, i+resolution):
            graph[j][i+resolution:i+2*resolution] = tmp
    
    return graph

def getMSEMetric(OP,t,graph,start,SP,tf,resolution_bits=0):
    """Return the MSE metric based on an input SP (set-point) ideal signal.
    Arguments:
    OP: initial driving signal
    t: time axis
    graph: graph (numpy array) as returned by getOPGraph
    start: point at which the modulated (ACO derived) part of the signal begins
    SP: set-point to be used as ideal signal for MSE
    tf: transfer function simulating the SOA
    """
    assert resolution_bits in [0,1,2,3,4], \
            "must have either 1, 2, 3 or 4 resolution bits"

    op = OP.copy()

    def MSEMetric(path):

        subVec = getSubVectorFromPath(path,graph)
        for i in range(len(subVec)):
            op[start+(i*np.power(2,resolution_bits)):start+2*(i*np.power(2,resolution_bits))] = subVec[i]

        output = get_tf_output(t,op,tf)
        score = np.mean(np.square(output - SP))

        return score
    
    return MSEMetric

def getSubVectorFromPath(path,graph):
    """Convert a ACO derived path + graph to a modulated signal region.
    """
    subVec = [graph[int(path[i][0])][int(path[i][1])] for i in range(len(path))]
    return subVec

def getTransferFunctionOutput(tf, U, T, X0, atol=1e-12):
    """
    This method sends a drive signal to a transfer function model and gets the output

    Args:
    - tf = transfer function
    - U = signal to drive transfer function with
    - T = array of time values
    - X0 = initial value
    - atol = scipy ode func parameter

    Returns:
    - PV = resultant output signal of transfer function
    """
    (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)

    return PV

def get_tf_output(T,U,tf):

    X0 = find_x_init(tf)
    PV = getTransferFunctionOutput(tf,U=U,T=T,X0=X0,atol=1e-13)

    # ENSURE LOWEST POINT OF SIGNAL >= 0 (OCCURS FOR SIMULATIONS) OTHERWISE WILL MESS UP ST, OS & RT ANALYSIS
    min_PV = np.copy(min(PV))
    if min_PV < 0:
        for i in range(0, len(PV)):
            PV[i] = PV[i] + abs(min_PV) # translate signal up

    return PV
