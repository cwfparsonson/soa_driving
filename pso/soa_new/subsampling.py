import numpy as np
import matplotlib.pyplot as plt



def getSampledInitialDrivingSignal(points_in_output_PSO, actual_input_signal):

    #points_in_output_PSO is an integer defined of 240 points in the search space
    #actual_input_signal will be defined in the pso class

    p2p = points_in_output_PSO/len(actual_input_signal)

    """p2p = int(16)"""

    #p2p is the ratio of input output (240) points to the actual input
    #e.g. if m = 12 and desired points in output signal is 240 then p2p = 10
    #and hence, for every value sampled in input, this will be duplicated 10 times

    """return p2p"""

    num_points = len(actual_input_signal)
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,num_points)
    init_OP = np.zeros(num_points)
    # for transfer function (low point MUST be -1):
    init_OP[:int(0.25*num_points)], init_OP[int(0.25*num_points):] = -1, 0.5 



    x = int(0)
    i = int(0)
    set_counter = p2p
    subsampled = []


    while x<240:

        if x == set_counter:
            i = i+1
            set_counter = set_counter+p2p

        subsampled.append(init_OP[i])

        x = x+1


    #print(subsampled)
    #print(len(subsampled))
    #print("")
    #print(init_OP)
    #print(len(init_OP))

    
    return subsampled

"""   def SampledOutput:



if __name__ == '__main__':
    num_points = 240
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,num_points)
    init_OP = np.zeros(num_points)
    # for transfer function (low point MUST be -1):
    init_OP[:int(0.25*num_points)], init_OP[int(0.25*num_points):] = -1, 0.5

    signals = []
    for tf in tfs:
        signals.append(getTransferFunctionOutput(tf,init_OP,t))
    plot_output(signals,labels)

    """
