import numpy as np
import matplotlib.pyplot as plt
from soa import subsampling


if __name__ == '__main__':

    points = 240
    num_points = 60
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,num_points)

    init_OP = np.zeros(num_points)                                           #initialise driving signal 
    init_OP[:int(0.25*num_points)],init_OP[int(0.25*num_points):] = -1, 0.5
    
    print(len(init_OP))

    init_OP = subsampling.getSampledInitialDrivingSignal(240, init_OP)

    print(len(init_OP))

    t = np.linspace(time_start,time_stop,points)
