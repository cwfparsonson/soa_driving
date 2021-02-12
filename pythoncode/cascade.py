import numpy as np
import random
import math
from soa import devices, signalprocessing, analyse, distort_tf

from scipy import signal

from collections import defaultdict

from soa.upsampling import ups




def PSO(n, iter_max, m, q):
    '''
    n : number of particles
    iter_max: max iterations
    m : dimensions
    q : number of cascaded SOAs
    '''
    x = np.tile(np.zeros((n, m)), (q,1))

    pbest = np.tile(np.copy(x), (q,1))
    
    # array for g best gbest[0] is the global best positions for first SOAs etc
    gbest = np.zeros(q, m)

    x_value = np.zeros(n)
    pbest_value = np.copy(x_value)
    
    v = np.tile(np.zeros((n, m)), (q,1))

    rel_improv = np.zeros((n, q))
    c1_max = 2.5 
    c2_max = 2.5 
    c1_min = 0.1
    c2_min = 0.1
    curr_iter = 1

    w_init = 0.9
    w_final = 0.5
    c1 = c2 = 0.2

    x = np.zeros((n,m))
    
    w = np.ones((n,q)) * w_init
    c1 = np.ones((n,q)) * c1
    c2 = np.ones((n,q)) * c2


    SOA_x = defaultdict(list)

    for i in range(n):
        for j in range(q):
            SOA_x[i].append(x[n * j + i])
    

    SOA_v = defaultdict(list)

    for i in range(n):
        for j in range(q):
            SOA_v[i].append(v[n * j + i])
    
    SOA_pbest = defaultdict(list)
    
    for i in range(n):
        for j in range(q):
            SOA_pbest[i].append(v[n * j + i])
    

    LB = np.zeros(m) # lower bound on particle positions
    UB = np.zeros(m) # upper bound on particle positions
    for g in range(0, m):
        LB[g] = -1.0
        UB[g] = 1.0
    v_LB = np.zeros(m) # lower bound on particle velocities
    v_UB = np.zeros(m) # upper bound on particle velocities
    for g in range(0, m):
        v_UB[g] = UB[g] * 0.05
        v_LB[g] = v_UB[g] * (-1)


    # To cascade: tile x and v  but keep x-values and pbest_values
    while curr_iter <= iter_max:

        for j in range(0, n):
            for q in range(q):
                rel_improv[j][q] = (pbest_value[j] - x_value[j]) \
                    / (pbest_value[j] + x_value[j]) 
                w[j][q] = w_init + ( (w_final - w_init) * \
                    ((math.exp(rel_improv[j][q]) - 1) / (math.exp(rel_improv[j][q]) + 1)) ) 
                c1[j][q] = ((c1_min + c1_max)/2) + ((c1_max - c1_min)/2) + \
                    (math.exp(-rel_improv[j][q]) - 1) / (math.exp(-rel_improv[j][q]) + 1) 
                c2[j][q] = ((c2_min + c2_max)/2) + ((c2_max - c2_min)/2) + \
                    (math.exp(-rel_improv[j][q]) - 1) / (math.exp(-rel_improv[j][q]) + 1) 
        
        # update particle velocities -->modify
            for j in range(0, n):
                for q in range(0, q):
                    for g in range(0, m):
                        SOA_v[j][q][g] = (w[j][q,g] * SOA_v[j][q, g]) + (c1[j][q] * random.uniform(0, 1) \
                            * (SOA_pbest[j][q, g] - SOA_x[j][q, g]) + (c2[j][q] * \
                                random.uniform(0, 1) * (gbest[q][g] - SOA_x[j][q,g])))

        # handle velocity boundary violations --> modify
        for j in range(0, n):
            for q in range(q):
                for g in range(0, m):
                    if SOA_v[j][q, g] > v_UB[g]:
                        SOA_v[j][q, g] = v_UB[g]
                    if SOA_v[j][q, g] < v_LB[g]:
                        SOA_v[j][q, g] = v_LB[g]
        
        # update particle positions --> modify
        for j in range(0, n):
            for q in range(q):
                SOA_x[j][q, :] = SOA_x[j][q, :] + SOA_v[j][q, :]
        
        # handle position boundary violations --> modfy
        for j in range(0, n):
            for q in range(q):
                for g in range(0, m):
                    if SOA_x[j][q, g] < LB[g]:
                        SOA_x[j][q, g] = LB[g]
                    elif SOA_x[j][q, g] > UB[g]:
                        SOA_x[j][q, g] = UB[g]

        # eval particle positions ---> modify
        x_value = __evaluateParticlePositions(SOA_x, m = m, q = q, curr_iter=curr_iter, plot=True)
        
        # update local best particle positions & fitness vals
        for j in range(0, n):
            if x_value[j] < pbest_value[j]:
                pbest_value[j] = x_value[j]
                for q in range(q):
                    for g in range(0, m):
                        pbest[j][q, g] = x[j][q, g]
            
        min_cost_index = np.argmin(pbest_value)
        for g in range(m):
            gbest_cost = pbest_value[min_cost_index]
        
        '''
        # update global best particle positions & history
        min_cost_index = np.argmin(pbest_value)
        if pbest_value[min_cost_index] < gbest_cost_history[-1]:
            for g in range(0, m):
                gbest[g] = pbest[min_cost_index, g]
            rt_st_os_analysis = np.vstack((rt_st_os_analysis, 
                                            __analyseSignal(gbest, 
                                                                curr_iter)))
            gbest_cost = pbest_value[min_cost_index]
            gbest_cost_history = np.append([gbest_cost_history], [gbest_cost])
            iter_gbest_reached = np.append([iter_gbest_reached], [curr_iter])
        cost_reduction = ((gbest_cost_history[0] - gbest_cost) \
            / gbest_cost_history[0])*100

        print('Reduced cost by ' + str(cost_reduction) + '% so far')


        __savePsoData(x, 
                            x_value, 
                            curr_iter, 
                            pbest, 
                            gbest_cost_history, 
                            iter_gbest_reached, 
                            rt_st_os_analysis) 
        '''
        print('Num iterations completed: ' + str(curr_iter) + ' / ' + str(iter_max))
        curr_iter += 1 

def __evaluateParticlePositions(self, particles, m, q, curr_iter=None, plot=False, is_first = False):
    """
    This method evaluates the positions of each particle in an array

    Args:
    - particles = array of particles to evaluate
    - plot = set whether want to plot (and save) the resultant PV and OP when 
    each particle is used to drive SOA
    - curr_iter = current iteration pso is on (only needed if plot == True)

    Returns:
    - x_value = array of fitness values for each particle position
    """

    
    curr_outputs = np.zeros((self.n, len(self.t2))) 
    '''
    plt.figure(1)
    plt.figure(2)
    '''
    x_value = np.zeros(self.n) # int current particle fitnesses/costs storage
    for j in range(0, self.n): 
        particle = particles[j]
        OP = np.copy(self.init_OP) 
        OP = np.tile(OP, (q,1))
        particleIndex = 0
        # Question about this part
        for q in range(q):
            for signalIndex in range(0, m):
                OP[q][signalIndex] = particle[q, particleIndex]
                particleIndex += 1 

        if self.sim_model != None:
            PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                    OP, 
                                                    self.t2, 
                                                    self.X0, casc = q) 
        else:
            PV = self.__getSoaOutput(OP) 

        x_value[j] = signalprocessing.cost(self.t2, 
                                            PV, 
                                            cost_function_label=self.cost_f, 
                                            st_importance_factor=self.st_importance_factor, 
                                            SP=self.SP).costEval 

 
        curr_outputs[j, :] = PV
    
    
    '''
        if plot == True:
            plt.figure(1) 
            plt.plot(self.t2, PV, c='b') 
            plt.figure(2)
            plt.plot(self.t, OP, c='r')
    
    if plot == True:
    # get best fitness for analysis
        min_cost_index = np.argmin(x_value)       
        if self.sim_model != None:              
            best_PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                        particles[min_cost_index,:], 
                                                        self.t2, 
                                                        self.X0) 
        else:
            best_PV = self.__getSoaOutput(particles[min_cost_index, :])      

    
    if plot == True:
        # finalise and save plot
        plt.figure(1)
        plt.plot(self.t2, self.SP, c='g', label='Target SP')
        plt.plot(self.t2, self.init_PV, c='r', label='Initial Output')
        plt.plot(self.t2, best_PV, c='c', label='Best fitness')
        st_index = analyse.ResponseMeasurements(best_PV, self.t2).settlingTimeIndex
        plt.plot(self.t2[st_index], 
                    best_PV[st_index], 
                    marker='x', 
                    markersize=6, 
                    color="red", 
                    label='Settling Point')
        plt.legend(loc='lower right')
        plt.title('PSO-Optimised Output Signals After ' + str(curr_iter) + \
            ' Generations for ' + str(self.num_points) + ' Points')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig(self.path_to_pso_data + str(curr_iter) + '_gen_outputs.png')  
        plt.close()

        plt.figure(2)
        plt.plot(self.t, 
                    particles[min_cost_index, :], 
                    c='c', 
                    label='Best fitness')
        plt.legend(loc='lower right')
        plt.title('PSO-Optimised Input Signals After ' + str(curr_iter) + \
            ' Generations')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.savefig(self.path_to_pso_data + str(curr_iter) + '_gen_inputs.png')  
        plt.close()

    if self.record_extra_info == True:
        # save current particle outputs
        curr_outputs_df = pd.DataFrame(curr_outputs)
        curr_outputs_df.to_csv(self.path_to_pso_data + 'curr_outputs_' + \
            str(curr_iter) + '.csv', 
                                index=None, 
                                header=False)
    if is_first == True:
        file = open(self.path_to_pso_data + 'Test.txt', "w") 
        file.write("Works") 
        file.close()
    '''
    return x_value

def __getTransferFunctionOutput(self, tf, U, T, X0, casc = 1, atol=1e-12):
        """
        This method sends a drive signal to a transfer function model and gets 
        the output

        Args:
        - tf = transfer function
        - U = signal to drive transfer function with
        - T = array of time values
        - X0 = initial value
        - atol = scipy ode func parameter

        Returns:
        - PV = resultant output signal of transfer function
        """
        for q in casc:
            U[q] = np.array(U[q])
            p = ups(240)
            U[q] = p.create(U[q])

        T = np.linspace(T[0], T[-1], 240)

        PV = np.array()

        for _ in range(casc):
            (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)
            X0 = PV
            min_PV = np.copy(min(PV))
            if min_PV < 0:
                for i in range(0, len(PV)):
                    PV[i] = PV[i] + abs(min_PV)

        # ensure lower point of signal >=0 (can occur for sims), otherwise
        # will destroy st, os and rt analysis

        return PV