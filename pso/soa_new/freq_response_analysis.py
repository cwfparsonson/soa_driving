import sys
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import scipy
import math

#sys.path.append(r'C:\Users\Christopher\OneDrive - University College London\ipes_cdt\phd_project\projects\soa_driving\code\soa_driving\optimisation\python\\')
from soa.optimisation import *




#num_points_list = [12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240]
num_points = 10
#num_points_list = [10]
time_start = 0.0
time_stop = 20e-9 # 18.5e-9
t = np.linspace(time_start,time_stop,240)



# DEFINE TRANSFER FUNCTION
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

#for num_points in num_points_list:
    
init_OP = np.zeros(num_points)
init_OP[:int(0.25*num_points)], init_OP[int(0.25*num_points):] = -1, 0.5 # for transfer function (low point MUST be -1)


signal = distort_tf.getTransferFunctionOutput(tf,init_OP,t)
w, y = scipy.signal.freqresp((num, den), n=20000)
zero_freq_amp = y[0]
mag = 10 * np.log10(abs(y/zero_freq_amp))
# mag = 10 * np.log10(abs(y))

print(signal)
print(t)

plt.figure(1)
plt.plot(t, signal)


"""
plt.figure(2)
plt.axhline(y=-3, color='r', linestyle='--')
plt.plot(w*1e-9*0.159155, mag)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, None])

plt.figure(3)
plt.axhline(y=-3, color='r', linestyle='--')
plt.plot(w*1e-9*0.159155, mag)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, 5])
plt.ylim([-5, 3])
plt.plot()

plt.figure(4)
plt.axhline(y=-3, color='r', linestyle='--')
plt.plot(w*1e-9*0.159155, mag)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, 0.5])
plt.ylim([-5, 3])
plt.plot()
"""

plt.figure(5)
plt.axhline(y=-3, color='r', linestyle='--')
plt.plot(w*1e-9*0.159155, mag, c='midnightblue')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Gain (dB)')
plt.xlim([0, 2])
plt.ylim([-20, 3])
plt.xticks([0.5, 1.0, 1.5, 2.0])

foo = np.interp(-3, mag, w)
print("-3dBm = " + str(foo) + " GHz")

plt.title("number of points " + str(num_points))
plt.plot()



plt.show()












