from Ant import *
from AntColony import *
import numpy as np
from metrics import *
from numpy import arange
import matplotlib.pyplot as plt
from SignalPerformanceValues import *
import globals
from plots import *
from distort_tf import *
import multiprocessing

print('starting')

OP = np.zeros(240)
OP[:60] = -1.0
OP[60:] = 0.5
t = np.linspace(0.0,20.0e-9,240)

tfs, labels = gen_tfs(num_facs=[1.0,1.2,1.4], 
                    a0_facs=[0.8],
                    a1_facs=[0.7,0.8,1.2],
                    a2_facs=[1.0,1.1,1.2],
                    all_combos=False)

sp_tf = tfs[0]
output0 = get_tf_output(t,OP,sp_tf)
output0Params = ResponseMeasurements(output0,t)
rt0 = output0Params.riseTime
st0 = output0Params.settlingTime
os0 = output0Params.overshoot

sp = SetPoint(output0Params)
SP = sp.sp

ants = 200
generations = 75
pheremone = 0.25
exploration = 0.1
evaporation = 0.5
start = 60
number_of_points = 180
vertical_resolution = 50
graph = getOPGraph(number_of_points,vertical_resolution,0.75,0.25)

tt = time.time()
tf = tfs[0]
save_dir = #NAME DIR PATH HERE
os.mkdir(save_dir)
function = getMSEMetric(OP,t,graph,start,SP,tf)

test1 = AntColony(
                graph=graph, \
                numberOfAnts=ants, \
                generations=generations, \
                metricFunction=function, \
                evaporationConstant=evaporation, \
                pheremoneExponent=pheremone, \
                heuristicExponent=globals.heuristicExponent, \
                pheremoneMatrix=globals.pheremoneMatrix, \
                heuristicMatrix=globals.heuristicMatrix, \
                Q=globals.Q, \
                explorationProbability=exploration, \
                onlyBestAntUpdate=globals.onlyBestAntUpdate, \
                repeatNode=globals.repeatNode, \
                universalStartingNode=globals.universalStartingNode, \
                movesPerGeneration=number_of_points - 1, \
                saveFilePath=save_dir \
                )

test1.optimize()
print(time.time() - tt)