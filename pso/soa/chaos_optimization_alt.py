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
                min_val = - 1.0,
                max_val = 1.0, 
                rep = 10):
        
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

        self.a = 0.7


    def cls(self, x, x_value, pbest, pbest_value, gbest, gbest_cost, gbest_cost_history):
        
        dummy = np.tile(gbest, (self.n, 1))

        dummy_value = np.copy(pbest_value)

        z = np.interp(np.copy(random.choice(pbest)), [self.min_val, self.max_val], [0, 1])

        fitness = np.zeros(self.rep)

        # Criterion that new gbest was found
        achieved = False

        # Chaotic Search Using Tent Mapping
        for i in range(0, self.rep):
            
            # Get the best particle
            p = np.copy(dummy[np.argsort(dummy_value)[0]])
            
            # Random Cascaded SOAs
            c = np.random.randint(low = 1, high = 4)
            
            # Logistic Mapping/Tent Mapping
            z = self.mapping(z)

            # Randomize part of particle using chaotic mapping
            for g in range((c - 1) * self.m, c * self.m):

                if random.uniform(0, 1) > 0.3:
                    
                    p[g] = np.interp(z[g], [0, 1], [self.LB[g], self.UB[g]])
            
            # Get and Evaluate Output
            fitness[i] = self.get_cost(p)

            idx = np.argsort(dummy_value)[-1]

            if dummy_value[idx] > fitness[i]:

                dummy_value[idx] = fitness[i]

                for g in range(0, self.m_c):

                    dummy[idx, g] = p[g]


            # Condition for better gbest/Break if found
            if fitness[i] < gbest_cost:
                
                achieved = True

                for g in range(0, self.m):
                    gbest[g] = p[g]
                    pbest[0, g] = p[g]
                    x[0, g] = p[g]
                
                tmp = np.copy(self.LB)
                
                if self.change_range:
                    self.a = self.a * (1- (gbest_cost_history[-1] - gbest_cost) / gbest_cost_history[-1])
                    for g in range(0, self.m):
                        self.LB[g] = max(self.LB[g], gbest[g] - self.a * (self.UB[g] - self.LB[g]))
                        self.UB[g] = min(self.UB[g], gbest[g] + self.a * (self.UB[g] - self.LB[g]))
                
                print((tmp == self.LB).all())
                    
                x_value[0] = fitness[i]
                pbest_value[0] = fitness[i]  
                gbest_cost = fitness[i]
                
                cost_reduction = ((gbest_cost_history[0] - gbest_cost) \
                    / gbest_cost_history[0])*100 
                
                
                print('----------------------------------------------------------')               
                print(f'Chaos Search Reduced by {cost_reduction} %')
                print('----------------------------------------------------------')                
            

        tmp = np.copy(x)
        (x, x_value, pbest, pbest_value) = self.update(x, x_value, pbest, pbest_value, dummy, dummy_value)
        print((x==tmp).all())
     
        return (x, x_value, pbest, pbest_value, gbest, gbest_cost, achieved)    


    def mapping(self, z):
        
        if self.map_type == 'logistic':
            z = 4 * z * (1 - z)
        
        elif self.map_type == 'tent':
            conds = [z < 0.5, z >= 0.5, z == 0]
            funcs = [lambda z: 2 * z, lambda z: 2 * (1 - z), lambda z: z + random.uniform(0,1)]
            z = np.piecewise(z, conds, funcs)

        return z      
    
    
    def get_cost(self, p):

        fitness = np.zeros(self.q)
        
        PV_chaos = self.__getTransferFunctionOutput(self.sim_model, p, self.t2, self.X0)
        for i in range(self.q):
            fitness[i] = signalprocessing.cost(self.t2, 
                                            PV_chaos[i], 
                                            cost_function_label=self.cost_f, 
                                            st_importance_factor=self.st_importance_factor, 
                                            SP=self.SP[i]).costEval

        return np.sum(fitness)

    
    def update(self, x, x_value, pbest, pbest_value, dummy, dummy_value):

        elite_idxs = np.argsort(dummy_value)[:4 * self.n // 5]

        print(elite_idxs)

        for j,idx in enumerate(elite_idxs):

            for g in range(self.m_c):
                x[j + 1, g] = dummy[idx, g]
                
                x_value[j + 1] = dummy_value[j]

            if dummy_value[idx] < pbest_value[j]:

                pbest_value[j + 1] = dummy_value[idx]
                
                for g in range(0, self.m_c):
                    pbest[j + 1, g] = dummy[idx, g]

               
        
        return (x, x_value, pbest, pbest_value)


    def update2(self, x, pbest, pbest_value, dummy, dummy_value, fitness, tmp, achieved):   
        
        idx = random.sample(range(1, self.n), 3 * self.n // 5)
        
        if not achieved:

            if pbest_value[idx[0]] > min(fitness):
                
                for g in range(0, self.m_c):
                    
                    x[idx[0], g] = tmp[g]                   
                    
                    pbest[idx[0], g] = tmp[g]
                    
                    pbest_value[idx[0]] = min(fitness)

            for i in range(1, len(idx)):
                
                for g in range(0, self.m_c):

                        x[idx[i], g] = dummy[idx[i], g]

                        if pbest_value[idx[i]] > dummy_value[idx[i]]:

                            pbest[idx[i], g] = dummy[idx[i], g]

                            pbest_value[idx[i]] = dummy_value[idx[i]]

        else:
            
            for i in range(0, len(idx)):
                
                for g in range(0, self.m_c):

                    x[idx[i], g] = dummy[idx[i], g]

                    if pbest_value[idx[i]] > dummy_value[idx[i]]:

                        pbest[idx[i], g] = dummy[idx[i], g]

                        pbest_value[idx[i]] = dummy_value[idx[i]]
       

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
        
        PV = np.zeros((self.q, sample))
        
        for i in range(self.q):
            
            input = input_init[:self.m]
            input = p.create(input)

            
            (_, PV[i], X0_init) = signal.lsim2(tf, input, T, X0=X0, atol=atol)

            X0 = X0_init[-1]
            
            input_init = input_init[self.m:]
        
        min_PV = np.copy(min(PV[-1]))
        if min_PV < 0:
            for j in range(0, len(PV[-1])):
                PV[-1][j] = PV[-1][j] + abs(min_PV)


        return PV