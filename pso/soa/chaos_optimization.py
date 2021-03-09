import numpy as np
import random
import math

from soa import optimisation
from soa import devices, signalprocessing, analyse, distort_tf
from soa import upsampling
from scipy import signal 

class chaos:

    def __init__(self, 
                n,
                m, 
                q,
                sim_model,
                t2,
                X0,
                cost_f,
                st_importance_factor,
                SP,
                map_type  = 'logistic',
                change_range = False,
                min_val = -2.5,
                max_val = 2.5, 
                rep = 50):
        
        self.n = n
        self.m = m
        self.q = q
        self.sim_model = sim_model
        self.t2 = t2
        self.X0 = X0
        self.cost_f = cost_f
        self.st_importance_factor = st_importance_factor
        self.SP = SP
        self.m_c = self.m * self.q

        self.min_val = min_val
        self.max_val = max_val

        self.LB = np.zeros(self.m_c)
        self.UB = np.zeros(self.m_c)
        
        for g in range(0, self.m_c):
            self.LB[g] = self.min_val
            self.UB[g] = self.max_val

        self.change_range = change_range
        
        self.map_type = map_type
        self.rep = rep


    def cls(self, x, pbest, pbest_value, gbest, gbest_cost, gbest_cost_history):
        
        dummy = np.tile(np.copy(gbest) , (self.n, 1))

        dummy_value = np.copy(pbest_value)

        z = np.interp(np.copy(random.choice(x)), [self.min_val, self.max_val], [0, 1])

        fitness = np.zeros(self.rep)

        tmp = np.copy(x[0])

        min_range = np.copy(self.LB)
        max_range = np.copy(self.UB)

        gamma = 1.0

        # Chaotic Search Using Tent Mapping
        for i in range(0, self.rep):
            
            # Criterion that new gbest was found
            achieved = False

            # Get a random particle
            p = np.copy(random.choice(dummy))
            
            # Random Cascaded SOAs
            r = np.random.randint(self.q - 1)
            
            # Logistic Mapping/Tent Mapping
            if self.map_type == 'logistic':
                z = 4 * z * (1 - z)
            
            elif self.map_type == 'tent':
                conds = [z < 0.5, z >= 0.5, z == 0]
                funcs = [lambda z: 2 * z, lambda z: 2 * (1 - z), lambda z: z + random.uniform(0,1)]
                z = np.piecewise(z, conds, funcs)


            # Randomize part of particle using chaotic mapping
            for g in range(r * self.m, (r + 2) * self.m):
                    p[g] = np.interp(z[g], [0, 1], [self.LB[g], self.UB[g]])
            
            # Get and Evaluate Output
            PV_chaos = self.__getTransferFunctionOutput(self.sim_model, p, self.t2, self.X0)
            fitness[i] = signalprocessing.cost(self.t2, 
                                               PV_chaos, 
                                               cost_function_label=self.cost_f, 
                                               st_importance_factor=self.st_importance_factor, 
                                               SP=self.SP).costEval 

            d_idx = random.sample(range(1, self.n), self.n)
            for j in range(0, len(d_idx)):
                # Consider if generated particle has better fitness than existing
                if fitness[i] < dummy_value[d_idx[j]]:
                    
                    dummy_value[d_idx[j]] = fitness[i]
                    
                    for g in range(0, self.m_c):
                        
                        dummy[d_idx[j], g] = p[g]

                    break     

            # Keep Track of best Particle in case gbest is not updated
            if fitness[i] == min(fitness):
                
                for g in range(0, self.m_c):
                    
                    tmp[g] = p[g]


            # Condition for better gbest/Break if found
            if fitness[i] < gbest_cost:
                
                achieved = True
                
                for g in range(0, self.m_c):
                    
                    gbest[g] = p[g]
                    pbest[0, g] = p[g]
                    x[0, g] = p[g]

                    if self.change_range:
                        min_range[g] = max(min_range[g], gbest[g] - gamma * (max_range[g] - min_range[g]))
                        max_range[g] = min(max_range[g], gbest[g] + gamma * (max_range[g] - min_range[g]))
                        gamma = gamma / 1.2
                    
                    
                gbest_cost = fitness[i]
                cost_reduction = ((gbest_cost_history[0] - gbest_cost) \
                    / gbest_cost_history[0])*100
                
                print('----------------------------------------------------------')               
                print(f'Chaos Search Reduced by {cost_reduction} %')
                print('----------------------------------------------------------')
            
        idx = random.sample(range(1, self.n), 4 * self.n // 5)
        
        if not achieved:

            for g in range(0, self.m_c):
                x[idx[0], g] = tmp[g]

                if pbest_value[idx[0]] < min(fitness):
                        
                    pbest[idx[0], g] = tmp[g]
                    pbest_value[idx[0]] = min(fitness)

            for i in range(1, len(idx)):
                
                for g in range(0, self.m_c):

                        x[idx[i], g] = dummy[idx[i], g]

                        if pbest_value[idx[i]] > dummy_value[idx[i]]:

                            pbest[idx[i], g] = dummy[idx[i], g]

                            pbest_value[idx[i]] = dummy_value[idx[i]]

        else:
            j = 0
            for i in range(0, len(idx)):
                
                for g in range(0, self.m_c):

                    x[idx[i], g] = dummy[j, g]

                    if pbest_value[idx[i]] > dummy_value[j]:

                        pbest[idx[i], g] = dummy[j, g]

                        pbest_value[idx[i]] = dummy_value[j]
                j += 1
        

        return (x, pbest, pbest_value, gbest, gbest_cost, achieved)    

    


    def __getTransferFunctionOutput(self, tf, U, T, X0, atol=1e-12):
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

        

        T = np.linspace(T[0], T[-1], 240)

        U = np.array(U)
        sample = 240
        p = upsampling.ups(sample)
        input_init = np.copy(U)
        

        for _ in range(self.q):
            PV = np.array([])
            input = input_init[:self.m]
            input = p.create(input)

            
            (_, PV, X0_init) = signal.lsim2(tf, input, T, X0=X0, atol=atol)
            X0 = X0_init[-1]
            min_PV = np.copy(min(PV))
            if min_PV < 0:
                for i in range(0, len(PV)):
                    PV[i] = PV[i] + abs(min_PV)
            
            input_init = input_init[self.m:]


        return PV