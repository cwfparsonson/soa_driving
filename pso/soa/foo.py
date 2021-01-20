import numpy as np
import matplotlib.pyplot as plt

"""
x = int(0)
i = int(0)
p2p = int(3)
set_counter = int(3)
subsampled_step = []
init_op = [1,2,3,4,5,6,7,8,9]

#subsampled_step.append(init_op[3])

for x in range(5):
    subsampled_step.append(init_op[x])
    x=x+1
    
print(subsampled_step)



while x<27:
    
    if x == set_counter:
        i = i+1
        set_counter = set_counter+p2p
        
    #subsampled_step [x] = init_op[i]
    
    subsampled_step.append(init_op[i])
    
    x = x+1

print(subsampled_step)
    
"""
    


def subsampled_signal(points_in_output_PSO=240, points_of_initial_signal):

    #points_in_output_PSO is an integer defined of 240 points in the search space
    #points_of_initial_signal will be defined in the pso class

    p2p = points_in_output_PSO/points_of_initial_signal


    #p2p is the ratio of input output (240) points to the actual input
    #e.g. if m = 12 and desired points in output signal is 240 then p2p = 10
    #and hence, for every value sampled in input, this will be duplicated 10 times



    num_points = points_of_initial_signal
    time_start = 0.0
    time_stop = 20e-9 # 18.5e-9
    t = np.linspace(time_start,time_stop,num_points)
    init_OP = np.zeros(num_points)
    # for transfer function (low point MUST be -1):
    init_OP[:int(0.25*num_points)], init_OP[int(0.25*num_points):] = -1, 0.5 
    """ what does this do?"""


    x = int(0)
    i = int(0)
    set_counter = p2p
    subsampled_step = []


    while x<240:

        if x == set_counter:
            i = i+1
            set_counter = set_counter+p2p

        subsampled_step.append(init_OP[i])

        x = x+1


    print(subsampled_step)
    print(len(subsampled_step))
    print("")
    print(init_OP)
    print(len(init_OP))
    return subsampled_step

    return subsampled_step

if __name__ == '__main__':
    subsampled_signal()








