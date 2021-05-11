import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y = []

with open('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Final_Learning_Curves/points=120/MSE_Vs_iter_cost_spread.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x,y, label='Run 1')
plt.xlabel('x')
plt.ylabel('y')
plt.title('points = 120')
plt.legend()
plt.show()

"""
numpy

x, y = np.loadtxt('/Users/hadi/Desktop/SOA_Complexity/PSO_Data/Final_Learning_Curves/points=120/MSE_Vs_iter_cost_spread.csv', delimiter=',', unpack=True)
plt.plot(x,y, label='Run 1')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
"""