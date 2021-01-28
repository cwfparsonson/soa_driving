import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import pandas as pd

sys.path.append('C:\\Users\\billv\\3project\\soa_driving\\pso\\soa')

from soa import analyse

class ups:
    
    # each upsampling instance can be set to have different points
    def __init__(self, points):
        self.points = points
    

    # main method of upsampling class
    def create(self, *args):
    # *args: list as input     
        input = np.array(*args)
        
        if input.size != self.points:
            n = self.points // input.size
            
            new_input = np.repeat(input, n)
            
            return new_input
        
        else:
            return input
    
if __name__ == '__main__':
    num_points = 10
    U = np.zeros(num_points) # initial drive signal (e.g. a step)
    U[:int(0.25*num_points)],U[int(0.25*num_points):] = -1, 0.5
    up = 240
    p = ups(up)
    U = p.create(U)
    UT = []
    for i in U:
        UT.append(np.array([i]))

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

    time_start = 0
    time_stop = 20e-9
    T = np.linspace(time_start, time_stop, num_points)
    # need to upsample time as well
    
    U2 = np.array([-1.0] * 480)
    T2 = np.linspace(0, 40e-9, 480)
    (_, _, xout) = signal.lsim2(tf, U=U2, T=T2, X0=None, atol=1e-13)
    X0 = xout[-1]

    TX  = np.linspace(T[0], T[-1], up)

    (_, PV, _) = signal.lsim2(tf, np.array(UT), TX, X0=X0, atol=1e-12)


    # ensure lower point of signal >=0 (can occur for sims), otherwise
    # will destroy st, os and rt analysis

    
    (_, PV, _) = signal.lsim2(tf, PV, TX, X0=X0, atol=1e-12)

    min_PV = np.copy(min(PV))
    if min_PV < 0:
        for i in range(0, len(PV)):
            PV[i] = PV[i] + abs(min_PV)
    



    def timing_analysis(PV_str, curr_iter, t):
        PV_df = pd.read_csv(PV_str)
        PV = PV_df['Data'].tolist()
        responseMeasurementsObject = analyse.ResponseMeasurements(PV, t) 

        rt = responseMeasurementsObject.riseTime
        st = responseMeasurementsObject.settlingTime
        os = responseMeasurementsObject.overshoot
        st_index = responseMeasurementsObject.settlingTimeIndex

        return [rt, st, os, st_index]   

    
    
    PV_strs = [('C:\\Users\\billv\\3project\\pythoncode\\Opt\\optimised_PV_20.csv', 20), ('C:\\Users\\billv\\3project\\pythoncode\\Opt\\optimised_PV_40.csv', 40), ('C:\\Users\\billv\\3project\\pythoncode\\Opt\\optimised_PV_60.csv', 60), ('C:\\Users\\billv\\3project\\pythoncode\\Opt\\optimised_PV_80.csv', 80), ('C:\\Users\\billv\\3project\\pythoncode\\Opt\\optimised_PV_120.csv', 120)]
    measurements = []
    for PV_str in PV_strs:
        T = np.linspace(time_start, time_stop, PV_str[1])
        measurements.append(timing_analysis(PV_str[0], 150, TX))
    
    num_points = []
    for num_point in PV_strs:
        num_points.append(num_point[1])

    df = pd.DataFrame(measurements)
    df.columns = ['Rise Time', 'Settling Time', 'Overshoot', 'ST index']
    df.index  = num_points
    print(df)
    
    
    
    
    plt.figure()
    plt.scatter(x = df.iloc[:, 0]*pow(10,9), y = df.iloc[:, 1]*pow(10,9), s = 75, c = df.iloc[:, 2], cmap = 'Oranges')
    
    for i, n in enumerate(num_points):
        plt.annotate(n, (df.iloc[i, 0]*pow(10,9), df.iloc[i, 1]*pow(10,9)), xytext = (df.iloc[i, 0]*pow(10,9) + 0.005, df.iloc[i, 1]*pow(10,9) + 0.01))
    
    plt.title('Scatter Plot')
    plt.xlabel('Rise Time')
    plt.ylabel('Settling time')
    #plt.show()
    # plt.savefig('C:\\Users\\billv\\3project\\pythoncode\\graph')

    x = np.arange(5)
    rt = df.iloc[:, 0]*pow(10,9)
    st = df.iloc[:, 1]*pow(10,9)
    ov = df.iloc[:, 2]
    num = df.index

    s_factor = 1/(rt*st*ov*pow(1.05, num))

    df.insert(4, 'S_factor', s_factor, True)
    
    print(df)
    fig, ax = plt.subplots()

    plt.bar(x, df.iloc[:, 4])

    plt.xticks(x, (20, 40, 60, 80, 120))
    plt.show()

    