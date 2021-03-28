from matplotlib.pyplot import xcorr
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
            c = np.random.randint(low = 0, high = 2)
            
            # Logistic Mapping/Tent Mapping
            z = self.mapping(z)

            # Randomize part of particle using chaotic mapping
            for g in range(c*self.m, (c+1)*self.m):      
                p[g] = (np.interp(z[g], [0, 1], [self.LB[g], self.UB[g]]) + gbest[g]) / 2.0
            # Get and Evaluate Output
            fitness[i] = self.get_cost(p)

            idx = np.argsort(dummy_value)[-1]

            if dummy_value[idx] > fitness[i]:
                print('changed')

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
            

        
        (x, x_value, pbest, pbest_value) = self.update(x, x_value, pbest, pbest_value, dummy, dummy_value)

        if self.rep >= 5:
            self.rep = self.rep - 5
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
        for j in range(self.q):
            if j > 0:
                X0 =  self.__find_x_init(tf[j-1])

            input = input_init[:self.m]
            input = p.create(input)

            
            (_, PV[j], _) = signal.lsim2(tf[j], input, T, X0=X0, atol=atol) 
            input_init = input_init[self.m:]
        
            min_PV = np.copy(min(PV[j]))
            if min_PV < 0:
                for i in range(0, len(PV[j])):
                    PV[j][i] = PV[j][i] + abs(min_PV)

        return PV
    
    def __find_x_init(self, tf):

    
        """
        This method calculates the state-vector from a long -1 drive signal. 
        Must call before sending / receiving signals to / from transfer function 
        model
        Args:
        - tf = transfer function
        Returns:
        - X0 = system's state-vector result for steady state
        """
        U = np.array([-1.0] * 480)
        T = np.linspace(0, 40e-9, 480)
        (_, _, xout) = signal.lsim2(tf, U=U, T=T, X0=None, atol=1e-13)
        X0 = xout[-1]

        return X0


class ol:

    def __init__(self, 
                m,
                q,
                sim_model,
                t2,
                X0,
                cost_f,
                st_importance_factor,
                SP):
        
        self.m = m

        self.q = q

        self.m_c = self.m * self.q

        self.D = self.q

        self.M = 2**math.ceil(math.log(self.D + 1, 2))

        self.u = int(math.log(self.M, 2))
          
        self.sim_model = sim_model
        
        self.t2 = t2
        
        self.X0 = X0
        
        self.cost_f = cost_f
        
        self.st_importance_factor = st_importance_factor
        
        self.SP = SP
        
    
    def evaluate(self, pbest, gbest):
        
        L = self.OA()

        f = np.ones(len(L))

        for i in range(len(L)):

            signal_b = np.zeros(self.m_c)

            for j in range(len(L[0])):

                if L[i, j] == 1:
                    
                    for g in range(j * self.m, (j + 1) * self.m):
                        signal_b[g] = pbest[g]
                
                else:
                    for g in range(j * self.m, (j + 1) * self.m):
                        signal_b[g] = gbest[g]
                
            f[i] = self.get_cost(signal_b)

        idx = np.argsort(f)[0]

        signal_b_fit = f[idx]

        for j in range(self.D):
            
            if L[idx, j] == 1:

                for g in range(j * self.m, (j + 1) * self.m):
                
                    signal_b[g] = pbest[g]

            else:

                for g in range(j * self.m, (j + 1) * self.m):
                
                    signal_b[j] = gbest[j]

        signal_p, signal_p_fit = self.factor_analysis(L, f, pbest, gbest)

        print(signal_b_fit, signal_p_fit)

        if signal_b_fit < signal_p_fit:
            return signal_p
        
        else:
            return signal_b        

    def OA(self):

        L = np.zeros((self.M + 1, self.M))

        for a in range(1, self.M + 1):
            
            for k in range(1, self.u + 1):
                
                b = pow(2, k -1)
                
                L[a][b] = math.floor((a - 1)/pow(2, self.u - k)) % 2

        for a in range(1, self.M  + 1):
            
            for k in range(2, self.u + 1):
                
                b = pow(2, k - 1)
                
                for s in range(1, b):
                    
                    L[a][b + s] = (L[a][s] + L[a][b]) % 2
                
        L = np.piecewise(L, [L == 0, L == 1], [1, 2])
        
        L = L[1:, 1:]

        print(L)

        return L
    
    
    def factor_analysis(self, L, f, pbest, gbest):
        
        S = np.zeros((self.D ,2))

    
        for g in range(0, self.D):

            factor_sum = {'p':0, 'g':0}
            count = {'p':0, 'g': 0}

            for i in range(len(L)):

                if L[i, g] == 1:

                    factor_sum['p'] += f[i]
                    count['p'] += 1
                
                else:

                    factor_sum['g'] += f[i]
                    count['g'] += 1

  
            S[g, 0] = factor_sum['p'] / count['p']

            S[g, 1] = factor_sum['g'] / count['g']
            
        
        signal_p = np.zeros(self.m_c)

        for j in range(self.D):
            
            if S[j, 0] < S[j, 1]:
                
                for g in range(j * self.m, (j + 1) * self.m):
                
                    signal_p[g] = pbest[g]
            
            else:

                for g in range(j * self.m, (j + 1) * self.m):
                    
                    signal_p[g] = gbest[g]
        
        signal_p_fit = self.get_cost(signal_p)

        return signal_p, signal_p_fit
        
    
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
        for j in range(self.q):
            if j > 0:
                X0 =  self.__find_x_init(tf[j-1])

            input = input_init[:self.m]
            input = p.create(input)

            
            (_, PV[j], _) = signal.lsim2(tf[j], input, T, X0=X0, atol=atol) 
            input_init = input_init[self.m:]
        
            min_PV = np.copy(min(PV[j]))
            if min_PV < 0:
                for i in range(0, len(PV[j])):
                    PV[j][i] = PV[j][i] + abs(min_PV)

        return PV
    
    
    def __find_x_init(self, tf):

    
        """
        This method calculates the state-vector from a long -1 drive signal. 
        Must call before sending / receiving signals to / from transfer function 
        model
        Args:
        - tf = transfer function
        Returns:
        - X0 = system's state-vector result for steady state
        """
        U = np.array([-1.0] * 480)
        T = np.linspace(0, 40e-9, 480)
        (_, _, xout) = signal.lsim2(tf, U=U, T=T, X0=None, atol=1e-13)
        X0 = xout[-1]

        return X0

    
    def get_cost(self, p):
    
        fitness = np.zeros(self.q)
        
        PV_OL = self.__getTransferFunctionOutput(self.sim_model, p, self.t2, self.X0)
        for i in range(self.q):
            fitness[i] = signalprocessing.cost(self.t2, 
                                            PV_OL[i], 
                                            cost_function_label=self.cost_f, 
                                            st_importance_factor=self.st_importance_factor, 
                                            SP=self.SP[i]).costEval

        return np.sum(fitness)



class cpso_sk:

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
                x,
                x_value,
                pbest,
                pbest_value,
                gbest,
                v,
                c1_min,
                c1_max,
                c2_min,
                c2_max,
                w_init,
                step = 'soa'):
        
        self.n = n

        self.m = m

        self.q = q

        self.m_c = self.m * self.q
          
        self.sim_model = sim_model
        
        self.t2 = t2
        
        self.X0 = X0
        
        self.cost_f = cost_f
        
        self.st_importance_factor = st_importance_factor
        
        self.SP = SP

        self.step = step

        self.w_init = w_init

        self.w = np.ones(self.n) * w_init
        
        self.c1 = np.ones(self.n) * 0.2
        
        self.c2 = np.ones(self.n) * 0.2

        self.c1_min = c1_min

        self.c1_max = c1_max

        self.c2_min = c2_min

        self.c2_max = c2_max

        self.x = x

        self.pbest = pbest

        self.x_value = x_value

        self.pbest_value = pbest_value

        self.v = v

        self.rel_improv = np.zeros(self.n)

        global context

        context = np.copy(gbest)
    
    
    def partition(self):

        global context

        if self.step == 'soa':


            context_cost = self.get_cost(context)


            for j in range(self.n):

                tmp = np.copy(self.x[j])
                
                self.x[j] = np.copy(context)

                for q in range(self.q):

                    for g in range(q * (self.m) + (q + 1) * self.m):

                        self.x[j, g] = tmp[g]


                    self.rel_improv[j] = (self.pbest_value[j] - self.x_value[j]) \
                        / (self.pbest_value[j] + self.x_value[j]) 
                    
                    self.w[j] = self.w_init + ( (self.w_final - self.w_init) * \
                        ((math.exp(self.rel_improv[j]) - 1) / (math.exp(self.rel_improv[j]) + 1)) ) 
                    
                    self.c1[j] = ((self.c1_min + self.c1_max)/2) + ((self.c1_max - self.c1_min)/2) + \
                        (math.exp(- self.rel_improv[j]) - 1) / (math.exp(-self.rel_improv[j]) + 1) 
                    
                    self.c2[j] = ((self.c2_min + self.c2_max)/2) + ((self.c2_max - self.c2_min)/2) + \
                        (math.exp(- self.rel_improv[j]) - 1) / (math.exp( - self.rel_improv[j]) + 1)
                    
                    
                    for g in range(q * (self.m) + (q + 1) * self.m):
                            self.v[j, g] = ((self.w[j] * self.v[j, g]) + (self.c1[j] * random.uniform(0, 1) \
                                * (self.pbest[j, g] - self.x[j, g]) + (self.c2[j] * \
                                    random.uniform(0, 1) * (context[g] - self.x[j,g]))))

                    
                    for g in range(q * (self.m) + (q + 1) * self.m):
                        
                        self.x[j, q] = self.x[j, q] + self.v[j, q]

                    
                    self.x_value[j] = self.get_cost(self.x[j, :])

                    
                    if self.x_value[j] < self.pbest_value[j]:

                        self.pbest_value[j] = self.x_value[j]

                        for g in range(q * (self.m) + (q + 1) * self.m):
                                self.pbest[j, g] = self.x[j, g]
                    

                    if self.x_value[j] < context_cost:

                        context_cost = self.x_value[j]

                        for g in range(q * (self.m) + (q + 1) * self.m):

                            context[g] = self.x[j, g]
                
            return context, context_cost

        else:

            pass

    
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
        for j in range(self.q):
            if j > 0:
                X0 =  self.__find_x_init(tf[j-1])

            input = input_init[:self.m]
            input = p.create(input)

            
            (_, PV[j], _) = signal.lsim2(tf[j], input, T, X0=X0, atol=atol) 
            input_init = input_init[self.m:]
        
            min_PV = np.copy(min(PV[j]))
            if min_PV < 0:
                for i in range(0, len(PV[j])):
                    PV[j][i] = PV[j][i] + abs(min_PV)

        return PV
    
    
    def __find_x_init(self, tf):

    
        """
        This method calculates the state-vector from a long -1 drive signal. 
        Must call before sending / receiving signals to / from transfer function 
        model
        Args:
        - tf = transfer function
        Returns:
        - X0 = system's state-vector result for steady state
        """
        U = np.array([-1.0] * 480)
        T = np.linspace(0, 40e-9, 480)
        (_, _, xout) = signal.lsim2(tf, U=U, T=T, X0=None, atol=1e-13)
        X0 = xout[-1]

        return X0

    
    def get_cost(self, p):
    
        fitness = np.zeros(self.q)
        
        PV_OL = self.__getTransferFunctionOutput(self.sim_model, p, self.t2, self.X0)
        for i in range(self.q):
            fitness[i] = signalprocessing.cost(self.t2, 
                                            PV_OL[i], 
                                            cost_function_label=self.cost_f, 
                                            st_importance_factor=self.st_importance_factor, 
                                            SP=self.SP[i]).costEval

        return np.sum(fitness)
