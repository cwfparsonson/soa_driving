num_points = 240
rise = 60
start = rise
generations = 80

numberOfPoints = 100
resolution = 100

t_max = 20.0e-9
t_min = 0.0

graph_min = 0.0
graph_max = 1.0

OP_min = 0.0
OP_max = 0.5

directory = None

numberOfAnts=[100,200,500]
evaporationConstant=[0.5,0.1,0.9]
pheremoneExponent=[0.5,1.0,2.0]
heuristicExponent=1.0


pheremoneMatrix=None
heuristicMatrix=None
Q=None
explorationProbability=[0.1,0.25,0.5]
onlyBestAntUpdate=True

repeatNode=False

universalStartingNode=(0,0)

directories = [
                '100_0.5_0.1_0.1', \
                '100_0.5_0.1_0.5', \
                '100_0.5_0.1_0.25', \
                '100_0.5_0.5_0.1', \
                '100_0.5_0.5_0.5', \
                '100_0.5_0.5_0.25', \
                '100_0.5_0.9_0.1', \
                '100_0.5_0.9_0.5', \
                '100_0.5_0.9_0.25', \
                '100_1.0_0.1_0.1', \
                '100_1.0_0.1_0.5', \
                '100_1.0_0.1_0.25', \
                '100_1.0_0.5_0.1', \
                '100_1.0_0.5_0.5', \
                '100_1.0_0.5_0.25', \
                '100_1.0_0.9_0.1', \
                '100_1.0_0.9_0.5', \
                '100_1.0_0.9_0.25', \
                '10_0.5_0.5_0.1', \
                '50_0.5_0.5_0.1', \
                '200_0.5_0.5_0.1', \
                '500_0.5_0.5_0.1', \
                ]

