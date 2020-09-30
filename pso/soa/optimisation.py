import sys, os
import pyvisa as visa

# make modules importable from anywhere
# sys.path.append(r'C:\Users\Christopher\OneDrive - University College London\ipes_cdt\phd_project\projects\soa_driving\code\soa_driving\optimisation\python\\')
from soa import devices, signalprocessing, analyse, distort_tf

import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

import multiprocessing
import pickle

from scipy import signal



class PSO:
    """
    Optimises parameters using particle swarm optimisation (PSO) algorithm
    """

    def __init__(self, 
                 t, 
                 init_OP, 
                 n, 
                 iter_max, 
                 rep_max, 
                 init_v_f, 
                 max_v_f, 
                 w_init=0.9, 
                 w_final=None, 
                 c1=0.2, c2=0.2, 
                 adapt_accel=True, 
                 sect_to_optimise='whole_signal', 
                 areas_to_suppress='None', 
                 off_suppress_f=0.2, 
                 on_suppress_f=0.8, 
                 embed_init_signal=True, 
                 path_to_embedded_signal=None, 
                 directory='Unknown', 
                 cost_f='Unknown', 
                 st_importance_factor=None, 
                 sim_model=None, 
                 awg=None, 
                 osc=None, 
                 awg_res=8, 
                 min_val=-1.0, 
                 max_val=1.0, 
                 record_extra_info=False, 
                 linux=True,
                 SP=None):
        """
        Initialise pso parameters

        Args:
        - t (list of floats): time array for signals
        - init_OP (array of floats): start OP from which to begin optimisation
        - n (int): number of particles
        - iter_max (int): max number of pso iterations
        - rep_max (int): number of times to repeat PSO
        - init_v_f (float): factor by which to multiply initial positions by to 
        get initial velocity for first iteration
        - max_v_f (float): factor by which to multiply max param values by to get 
        max velocity for each iteration
        - w_init (float): initial inertial weight value (0 <= w < 1) e.g. 0.9
        - w_final (float): final intertia weight value e.g. 0.5 (0 <= w < 1)
        - c1 (float): acceleration constant (0 <= c1 <= 1)
        - c2 (float): pso acceleration constant (0 <= c2 <= 1)
        - adapt_accel (boolean): whether want to use adaptive acceleration i.e. 
        adaptively adjust w, c1 and c2
        - sect_to_optimise (str): specifies which part of signal to optimise 
        (enter as string from list of options. Must be 1 of: 'whole_signal'
        - areas_to_suppress (str): specify which (if any) areas of signal to 
        restrict in terms of what value they particles can take. Must be 1 of: 
        'None', 'start_centre', 'start_centre_end', 'pisic_shape'
        - off_suppress_f (float): factor by which to multiply initial guess by 
        when in 'off' state and add to this to get space constraint
        - on_suppress_f (float): factor by which to multiply initial guess by when 
        in 'on' state to get on constraing for PSO particles
        - embed_init_signal (boolean): specify whether want to embed the initial 
        signal amongst the initialised particle positions so that don't ever 
        optimise towards something that's worse than initial signal
        - path_to_embedded_signal (str): give full path to where signal .csv file 
        of signal you want to embed is saved. If path_to_embedded_signal = None 
        and embed_init_signal = True, will embed init_OP amongst the initialised 
        particle positions
        - directory (str): directory in which we're working (will save files here)
        - cost_f (str): cost function to use to evaluate performance. Must be 1
        of: 'mSE', 'st', 'mSE+st', 's_mse+st', 'mse+st+os', 'zlpc'
        - st_importance_factor (float): factor by which want settling time to be 
        larger/smaller than mean squared error. Only (and must) specify if using 
        s_mse+st cost_f
        - sim_model (obj): simulation model to use if using simulation. When model 
        is passed a list of inputs, it should be able to simulate list of outputs
        - awg (obj): arbitrary waveform generator object if using experiment
        - osc (obj): oscilliscope object if using experiment
        - awg_res (int): resolution (bits) of awg used to generate signal
        - min_val (float): min voltage value awg can take
        - max_val (float): max voltage value awg can take
        - record_extra_info (boolean) = If want to plot extra info about PSO. 
        - linux (bool): If True, file dirs have forward slash. If false, have backslash
        - SP (list of floats): target SP PSO should try to achieve
        """
        if linux:
            self.slash = '/'
        else:
            self.slash = '\\'

        self.t = t
        self.init_OP = init_OP
        self.n = n
        self.iter_max = iter_max
        self.rep_max = rep_max
        self.init_v_f = init_v_f
        self.max_v_f = max_v_f
        self.w_init = w_init
        self.w_final = w_final
        self.c1 = c1
        self.c2 = c2
        self.adapt_accel = adapt_accel
        self.sect_to_optimise = sect_to_optimise
        self.areas_to_suppress = areas_to_suppress
        self.off_suppress_f = off_suppress_f
        self.on_suppress_f = on_suppress_f
        self.embed_init_signal = embed_init_signal
        self.path_to_embedded_signal = path_to_embedded_signal
        self.directory = directory
        self.cost_f = cost_f
        self.st_importance_factor = st_importance_factor
        self.sim_model = sim_model
        self.awg = awg
        self.osc = osc
        self.awg_res = awg_res
        self.min_val = min_val
        self.max_val = max_val
        self.SP = SP
        self.record_extra_info = record_extra_info
        self.num_points = len(self.init_OP)
        if sim_model == None and awg == None or \
           sim_model == None and osc == None or \
           sim_model != None and awg != None and osc != None:
           message = 'Must either use simulation or experimental env to perform \
                      pso in. sim_model, awg and osc cannot all be None /not None.'
           sys.exit(message)
        if self.adapt_accel == True and w_final == None:
            sys.exit('Must specifiy w_final value if using adaptive accel.')

        # get init output
        if self.sim_model != None:
            self.X0 = self.__find_x_init(self.sim_model) 
            self.init_PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                            self.init_OP, 
                                                            self.t, 
                                                            self.X0) 
        else:
            self.init_PV = self.__getSoaOutput(self.init_OP) 

        # init params
        self.K_index, self.K = self.__getSectionToOptimise() # opt indices
        self.m = len(self.K) # num params to optimise
        self.num_points = int(len(self.t)) # num points in signal
        self.curr_iter = 0 
        self.x = np.zeros((self.n, self.m)) # current pop position array
        self.x_value = np.zeros(self.n) # fitness vals of positions
        self.pbest_value = np.copy(self.x_value) # best local fitness vals
        self.min_cost_index = np.argmin(self.pbest_value) # index best fitness
        self.gbest = np.copy(self.K) # global best positions
        self.gbest_cost = self.pbest_value[self.min_cost_index] # global best val
        self.awg_step_size = (self.max_val - self.min_val) / (2**self.awg_res)
        if self.SP is None:
            self.SP = analyse.ResponseMeasurements(self.init_PV, self.t).sp.sp 
        else:
            self.SP = SP

        # set particle position and velocity boundaries
        self.LB = np.zeros(self.m) # lower bound on particle positions
        self.UB = np.zeros(self.m) # upper bound on particle positions
        for g in range(0, self.m):
            self.LB[g] = self.min_val
            self.UB[g] = self.max_val
        self.v_LB = np.zeros(self.m) # lower bound on particle velocities
        self.v_UB = np.zeros(self.m) # upper bound on particle velocities
        for g in range(0, self.m):
            self.v_UB[g] = self.UB[g] * self.max_v_f
            self.v_LB[g] = self.v_UB[g] * (-1)

        # initialise particle positions
        for g in range(0, self.m):
            if self.embed_init_signal == True:
                if self.path_to_embedded_signal == None:
                    self.x[0, g] = self.init_OP[g] # embed init OP at start
                else:
                    init_sig = self.__getInitialDrivingSignalGuess(self.path_to_embedded_signal)
                    self.x[0, g] = init_sig[g]
            else:
                self.x[0, g] = self.LB[g] + ( random.uniform(0, 1) * (self.UB[g] - self.LB[g]) )
                self.x[0, :] = self.__discretiseParticlePosition(self.x[0, :])
                self.x[0, :] = self.__suppressAreasOfSignal(self.x[0, :])
                self.x[0, :] = self.__discretiseParticlePosition(self.x[0, :])
                self.x[0, :] = self.__suppressAreasOfSignal(self.x[0, :])
                self.x[0, :] = self.__discretiseParticlePosition(self.x[0, :])
        for j in range(1, self.n):
            for g in range(0, self.m):
                self.x[j, g] = self.LB[g] + \
                    (random.uniform(0,1) * (self.UB[g] - self.LB[g]))
            self.x[j, :] = self.__discretiseParticlePosition(self.x[j, :]) 
            self.x[j, :] = self.__suppressAreasOfSignal(self.x[j, :]) 
            self.x[j, :] = self.__discretiseParticlePosition(self.x[j, :]) 
            self.x[j, :] = self.__suppressAreasOfSignal(self.x[j, :]) 
            self.x[j, :] = self.__discretiseParticlePosition(self.x[j, :]) 

        # initialise particle velocities
        self.v = np.zeros((self.n, self.m))
        for j in range(0, self.n):
            for g in range(0, self.m):
                self.v[j, g] = self.init_v_f * self.x[j, g]

        self.pbest = np.copy(self.x) 
        self.iter_gbest_reached = [0] # for plotting
    
        # set up dirs and load data if needed
        self.path_to_data = self.directory + self.slash + 'data' + self.slash # path to save data
        self.pso_dir_name = "n_" + str(self.n) + \
                            "_mxvl_" + str(self.max_val) + \
                            "_mnvl_" + str(self.min_val) + \
                            "_ivf_" + str(self.init_v_f) + \
                            "_mvf_" + str(self.max_v_f) + \
                            "_irmx_" + str(self.iter_max) + \
                            "_rm_" + str(self.rep_max) + \
                            "_cstf_" + str(self.cost_f) + \
                            '_sif_' + str(self.st_importance_factor) + \
                            "_wi_" + str(self.w_init) + \
                            "_wf_" + str(self.w_final) + \
                            "_c1_" + str(self.c1) + \
                            "_c2_" + str(self.c2) + \
                            '_aptacl_' + str(self.adapt_accel) + \
                            '_offsf_' + str(self.off_suppress_f) + \
                            '_onsf_' + str(self.on_suppress_f) + \
                            '_emb_' + str(self.embed_init_signal) + self.slash
        self.path_to_pso_data = self.path_to_data + self.pso_dir_name
        if os.path.exists(self.path_to_data) == False:
            os.mkdir(self.path_to_data)
        if os.path.exists(self.path_to_pso_data) == False:
            os.mkdir(self.path_to_pso_data) 
        if os.path.exists(self.path_to_pso_data + 'curr_pos.csv') == True:
            print('Programme detected a curr_pos.csv file in pso data directory ' + \
                str(self.path_to_pso_data) + \
                '. Continue from the last particle swarm position(s) saved?')
            ans = input("y or n: ")
            while ans != 'y' or ans != 'n':
                ans = input('y or n')
            if ans == 'n':
                pass
            elif ans == 'y':
                message = 'Please check your folders in the path ' + \
                    str(self.path_to_data) + \
                    '. Please enter the full path of the folder from which you \
                        want to load your saved data. To find the path, open the \
                        folder and right click on any file saved inside it, \
                        click on properties and copy-past the path. This programme \
                        requires you to manually enter this path because you might \
                        have multiple iteration extension folders to choose from \
                        as a starting point.'
                print(message)
                self.path_to_pso_data = input('Enter path of folder data is saved\
                     in (NOT path to specific file): ') + self.slash
                while os.path.exists(self.path_to_pso_data) == False:
                    print('Entered path not found. Please enter a valid path to \
                        your saved data')
                    self.path_to_pso_data = input('Enter path of folder data is \
                        saved in (NOT path to specific file): ') + self.slash
                self.iter_max, \
                self.path_to_pso_data, \
                self.x, \
                self.pbest, \
                self.x_value, \
                self.pbest_value, \
                self.min_cost_index, \
                self.gbest, \
                self.gbest_cost, \
                self.curr_iter = self.__loadPsoData() 
                os.mkdir(self.path_to_pso_data)
        
        else:
            # evaluate init particle positions
            print('Initialising PSO...')
            start_time = time.time()
            self.x_value = self.__evaluateParticlePositions(np.copy(self.x), 
                                                            curr_iter=self.curr_iter, 
                                                            plot=True) 
            self.pbest_value = np.copy(self.x_value) # store local best vals
            end_time = time.time()
            time_all_particles = end_time - start_time
            print("Time to evaluate " + str(self.n) + \
                " particles == time per generation: " + str(time_all_particles) + \
                " s | " + str(time_all_particles/60) + " mins | " + \
                str((time_all_particles/60)/60) + " hrs")
            time_all_generations = time_all_particles * self.iter_max
            print("Time to evaluate " + str(self.iter_max) + \
                " generations == estimated time to algorithm completion: " + \
                str(time_all_generations) + " s | " + str(time_all_generations/60) \
                + " mins | " + str((time_all_generations/60)/60) + " hrs")
            print("Time to complete all " + str(self.rep_max) + " PSO repetitions: " + \
                str(time_all_generations*self.rep_max) + " s | " + \
                str((time_all_generations*self.rep_max)/60) + " mins | " + \
                str(((time_all_generations*self.rep_max/60))/60) + " hrs")
            
            # init global cost history for plotting
            self.gbest_cost_history = [] 
            self.min_cost_index = np.argmin(self.pbest_value)
            for g in range(0, self.m):
                self.gbest[g] = self.pbest[self.min_cost_index, g] 
            self.gbest_cost = self.pbest_value[self.min_cost_index] 
            self.gbest_cost_history = np.append([self.gbest_cost_history], 
                                                [self.gbest_cost]) 

            print('Costs: ' + str(self.pbest_value))
            print('Best cost: ' + str(self.gbest_cost))

            # init analysis vals
            self.rt_st_os_analysis = [self.__analyseSignal(self.gbest, curr_iter=0)] 

            if self.record_extra_info == True:
                # record init particle positions and pbest
                self.__savePsoData(np.copy(self.x), 
                                   np.copy(self.x_value), 
                                   self.curr_iter, 
                                   np.copy(self.pbest), 
                                   self.gbest_cost_history, 
                                   [0], 
                                   self.rt_st_os_analysis)


        # run pso algorithm
        self.__runPsoAlgorithm() 

    def __analyseSignal(self, OP, curr_iter):
        """
        This method analyses a signal to get its rise time, settling time and 
        overshoot. Will also save input and output data

        Args:
        - OP = driving signal whose resultant output we want to analyse
        - curr_iter = current iteration. Use this for saving.

        Returns:
        - [rt, st, os] = table with rise time, settling time and overshoot of 
        output signal
        """

        # GET OUTPUT
        if self.sim_model != None:
            PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                  OP, 
                                                  self.t, 
                                                  self.X0) 
        else:
            PV = self.__getSoaOutput(OP) 

        OP_df = pd.DataFrame(OP)
        OP_df.to_csv(self.path_to_pso_data + "OP_itr" + str(curr_iter) + ".csv", 
                     index=None, 
                     header=False)
        PV_df = pd.DataFrame(PV)
        PV_df.to_csv(self.path_to_pso_data + "PV_itr" + str(curr_iter) + ".csv", 
                     index=None, 
                     header=False)

        responseMeasurementsObject = analyse.ResponseMeasurements(PV, self.t) 

        rt = responseMeasurementsObject.riseTime
        st = responseMeasurementsObject.settlingTime
        os = responseMeasurementsObject.overshoot
        st_index = responseMeasurementsObject.settlingTimeIndex

        return [rt, st, os, st_index]


    def __getInitialDrivingSignalGuess(self, path_to_sig):
        '''
        This method loads a previously defined csv with the driving signal that 
        we want to use as our initial guess to embed amongst the initialised 
        particle positions

        Args:
        - path_to_sig = path to signal that want to use 

        Returns:
        - init_sig = initial guess for signal to embed amongst particles
        '''
        init_sig_array = pd.read_csv(path_to_sig, header=None).values 
        init_sig = np.zeros(len(init_sig_array))

        for i in range(0, len(init_sig_array)):
            init_sig[i] = float(init_sig_array[i]) 

        return init_sig

    def __savePsoData(self, 
                      x, 
                      x_value, 
                      curr_iter, 
                      pbest, 
                      gbest_cost_history, 
                      iter_gbest_reached, 
                      rt_st_os_analysis):
        """
        This method is for saving pso progress so it can later be loaded

        Args:
        - x = particle positions to save
        - x_value = fitness values for current positions
        - pbest = personal historic best position for each particle
        - curr_iter = current iteration reached so far
        - gbest_cost_history = history of gbest reached
        - iter_gbest_reached = corresponding iteration new gbest was found
        - rt_st_os_analysis = hsitory of rise time, settling time and overshoot
        throughout pso algoithm

        Returns:
        - 
        """

        iter_gbest_reached_df = pd.DataFrame(iter_gbest_reached)
        gbest_cost_history_df = pd.DataFrame(gbest_cost_history)
        rt_st_os_analysis_df = pd.DataFrame(rt_st_os_analysis)
        curr_pos_df = pd.DataFrame(x)
        curr_pos_df.to_csv(self.path_to_pso_data + 'curr_pos.csv', 
                           index=None, 
                           header=False)
        curr_pos_val_df = pd.DataFrame(x_value)
        curr_pos_val_df.to_csv(self.path_to_pso_data + 'curr_pos_values.csv', 
                               index=None, 
                               header=False)
        iter_gbest_reached_df.to_csv(self.path_to_pso_data + "iter_gbest_reached.csv", 
                                     index=None, 
                                     header=False)
        gbest_cost_history_df.to_csv(self.path_to_pso_data + "gbest_cost_history.csv", 
                                     index=None, 
                                     header=False)
        rt_st_os_analysis_df.to_csv(self.path_to_pso_data + 'rt_st_os_analysis.csv', 
                                    index=None, 
                                    header=False)
        f = open(self.path_to_pso_data + 'curr_iter.csv', 'w')
        f.write(str(curr_iter))

        if self.record_extra_info == True:
            # save x and pbest for each particle
            curr_pbest_df = pd.DataFrame(x)
            curr_pbest_df.to_csv(self.path_to_pso_data + 'curr_inputs_' \
                + str(curr_iter) + '.csv', 
                                 index=None,   
                                 header=False)
            curr_x_df = pd.DataFrame(pbest)
            curr_x_df.to_csv(self.path_to_pso_data + 'pbest_pos_' \
                + str(curr_iter) + '.csv', 
                             index=None, 
                             header=False)


    def __loadPsoData(self):
        """
        This method loads previously saved pso data

        Args:
        - 

        Returns:
        - iter_max = max number of pso iterations
        - path_to_pso_data = path to where pso data will now be stored
        - x = particle positions
        - pbest = local particle position values
        - x_value = fitness of particle positions
        - pbest_value = local particle position best fitnesses
        - min_cost_index = particle index of fittest particle
        - gbest = global best particle
        - gbest_cost = global best particle fitness
        """
        # init params
        iter_max = 0
        path_to_pso_data = ''
        curr_iter = 0 
        x = np.zeros(self.n) 
        x_value = np.zeros(self.n) 
        min_cost_index = 0 
        gbest = np.zeros(len(self.K)) 

        # get where saved data is
        self.path_to_pso_data = input('Enter folder path of data: ') + self.slash
        while os.path.exists(self.path_to_pso_data) == False:
            print('Entered path not found. Enter a valid path to your saved data')
            path_to_pso_data = input('Enter folder path of data: ') + self.slash
        curr_iter_array = pd.read_csv(path_to_pso_data + "curr_iter.csv", 
                                      header=None).values 
        curr_iter = curr_iter_array[0]
        curr_iter = curr_iter[0]

        # update
        print('How many more generations would you like to do?')
        extend_iterations_by = int(input('Number of extra generations (integer): '))
        while type(extend_iterations_by) != int:
            extend_iterations_by = int(input('Number of extra generations (integer): '))
        iter_max = curr_iter + extend_iterations_by
        pso_dir_name = "n_" + str(self.n) + \
                       "_mxvl_" + str(self.max_val) + \
                       "_mnvl_" + str(self.min_val) + \
                       "_ivf_" + str(self.init_v_f) + \
                       "_mvf_" + str(self.max_v_f) + \
                       "_irmx_" + str(self.iter_max) + \
                       "_rm_" + str(self.rep_max) + \
                       "_cstf_" + str(self.cost_f) + \
                       '_sif_' + str(self.st_importance_factor) + \
                       "_wi_" + str(self.w_init) + \
                       "_wf_" + str(self.w_final) + \
                       "_c1_" + str(self.c1) + \
                       "_c2_" + str(self.c2) + \
                       '_aptacl_' + str(self.adapt_accel) + \
                       '_offsf_' + str(self.off_suppress_f) + \
                       '_onsf_' + str(self.on_suppress_f) + \
                       '_embd_' + str(self.embed_init_signal) + self.slash
        path_to_pso_data = self.path_to_data + pso_dir_name

        # load positions
        x_df = pd.read_csv(path_to_pso_data + "curr_pos.csv", 
                           header=None) 
        x_array = x_df.values
        for i in range(0, self.num_points):
            x[i] = float(x_array[i]) 
        pbest = np.copy(x)

        # load position vals
        x_value_df = pd.read_csv(path_to_pso_data + "curr_pos_values.csv", 
                                 header=None) 
        x_value_array = x_value_df.values
        for i in range(0, self.n):
            x_value[i] = float(x_value_array[i]) 
        pbest_value = np.copy(self.x_value)
        min_cost_index = np.argmin(pbest_value)
        for g in range(0, len(self.K)):
            gbest[g] = pbest[min_cost_index, g] 
        gbest_cost = pbest_value[min_cost_index]

        data = (iter_max, 
                path_to_pso_data, 
                x, 
                pbest, 
                x_value, 
                pbest_value,
                min_cost_index, 
                gbest, 
                gbest_cost, 
                curr_iter)

        return data


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
        (_, PV, _) = signal.lsim2(tf, U, T, X0=X0, atol=atol)

        # ensure lower point of signal >=0 (can occur for sims), otherwise
        # will destroy st, os and rt analysis
        min_PV = np.copy(min(PV))
        if min_PV < 0:
            for i in range(0, len(PV)):
                PV[i] = PV[i] + abs(min_PV)

        return PV


    def __getSoaOutput(self, OP, channel=1):
        """
        This method sends a drive signal to the SOA and gets an soa output

        Args:
        - OP = signal to send to AWG which will be used to drive SOA
        - channel = channel osc reads soa output on

        Returns:
        - PV = soa output
        """

        if type(OP[1]) == np.ndarray:
            # must convert to float as osc only takes list of floats
            flattened_OP = [val for sublist in OP for val in sublist]
        else:
            # no need to convert
            flattened_OP = OP

        self.awg.send_waveform(flattened_OP, suppress_messages=True)
        time.sleep(3)
        PV = self.osc.measurement(channel=channel)

        return PV

    def __getSectionToOptimise(self):
        """
        This method sets the section of the signal that we want to optimise

        Args:
        - 

        Returns:
        - K_index = column vector of particle indices
        - K = column vector of particle values
        """

        if self.sect_to_optimise == 'whole_signal':
            start_opt_index = 0
            end_opt_index = int(self.num_points) 
        
        # create index and value arrays of points (particles K) to optimise
        K_index = np.asarray(list(range(start_opt_index, end_opt_index+1)))
        K = np.asarray([self.init_OP[start_opt_index:end_opt_index]])

        # transpose to column vectors 
        K_index = K_index.transpose()
        K = K.transpose()

        return K_index, K

    def __discretiseParticlePosition(self, part_pos):
        """
        This method discretises a drive signal/particle positions so that they 
        can be read by awg. This also reduces the size of the pso algorithm
        search space of particle positions, which will speed up convergence

        Args:
        - part_pos = drive signal / particle position

        Returns: 
        - discretised particle position
        """

        for g in range(0, len(part_pos)):
            if part_pos[g] % self.awg_step_size != 0:
                # value not allowed therefore must discretise
                part_pos[g] = int(part_pos[g] / self.awg_step_size) \
                    * self.awg_step_size 
        
        return part_pos

    def __suppressAreasOfSignal(self, part_pos):
        """
        This method suppresses (or doesn't) various areas of the drive signal 
        from being able to take certain values depending on what user has specified

        Args:
        - part_pos = drive signal / particle position

        Returns:
        - suppressed (or not suppressed) particle position
        """

        if self.areas_to_suppress == 'None':
            # no suppression
            pass

        elif self.areas_to_suppress == 'start_centre':
            # get params
            start_drive = self.init_OP[0] 
            end_drive = self.init_OP[-1] 
            max_drive = np.amax(self.init_OP) 
            for i in range(0, len(self.init_OP)):
                if self.init_OP[i] > start_drive:
                    # get particle index signal turns on
                    K_index_signal_on = int(i - 0.5*(len(self.init_OP) - len(self.K))) 
                    break
            for i in range(int(len(self.init_OP)-1), K_index_signal_on, -1):
                if self.init_OP[i] > end_drive:
                    # get particle index signal turns off
                    K_index_signal_off = int(i - 0.5*(len(self.init_OP) - len(self.K))) 

            # suppress start and centre
            for index in range(0, K_index_signal_on):
                if part_pos[index] > start_drive + abs(self.off_suppress_f*start_drive):
                        # suppress start of signal
                        part_pos[index] = start_drive + abs(self.off_suppress_f*start_drive)
            for index in range(K_index_signal_on, len(self.K)):
                if part_pos[index] < max_drive - abs(self.on_suppress_f*max_drive):
                    # suppress centre  to end of signal
                    part_pos[index] = max_drive - abs(self.on_suppress_f*max_drive)
        
        elif self.areas_to_suppress == 'start_centre_end':
            # get params
            start_drive = self.init_OP[0] 
            max_drive = np.amax(self.init_OP) 
            for i in range(0, len(self.init_OP)):
                if self.init_OP[i] > start_drive:
                    K_index_signal_on = int(i - 0.5*(len(self.init_OP) - len(self.K))) 
                    break

            # suppress start, centre and end
            for index in range(0, K_index_signal_on):
                if part_pos[index] > start_drive + abs(self.off_suppress_f*start_drive):
                    # suppress start of signal
                    part_pos[index] = start_drive + abs(self.off_suppress_f*start_drive)
            for index in range(K_index_signal_off, int(len(self.K))):
                if part_pos[index] > end_drive + abs(self.off_suppress_f*start_drive):
                    # suppress end of signal
                    part_pos[index] = end_drive + abs(self.off_suppress_f*start_drive)
            for index in range(K_index_signal_on, K_index_signal_off):
                if part_pos[index] < max_drive - abs(self.on_suppress_f*max_drive):
                    # suppress centre of signal
                    part_pos[index] = max_drive - abs(self.on_suppress_f*max_drive)

        elif self.areas_to_suppress == 'pisic_shape':
            # suppress start and most of centre except for leading edge of 
            # drive signal, as this is an optional pisic area for PSO to fill
            pisic_length_factor = 0.1

            # get params
            start_drive = self.init_OP[0] 
            end_drive = self.init_OP[-1] 
            max_drive = np.amax(self.init_OP) 
            for i in range(0, len(self.init_OP)):
                if self.init_OP[i] > start_drive:
                    K_index_signal_on = int(i - 0.5*(len(self.init_OP) - len(self.K))) 
                    break
            for i in range(int(len(self.init_OP)-1), K_index_signal_on, -1):
                if self.init_OP[i] > end_drive:
                    K_index_signal_off = int(i - 0.5*(len(self.init_OP) - len(self.K)))

            # suppress start and centre
            for index in range(0, K_index_signal_on):
                if part_pos[index] > start_drive + abs(self.off_suppress_f*start_drive):
                    # suppress start of signal
                    part_pos[index] = start_drive + abs(self.off_suppress_f*start_drive)

            for index in range(K_index_signal_on + int((len(self.K)*pisic_length_factor)), 
                               len(self.K)):
                if part_pos[index] > self.init_OP[-1]:
                    # suppress centre to below the step/initial input (except 
                    # pisic area, whose length is defined by pisic_length_factor)
                    part_pos[index] = self.init_OP[-1]


        return part_pos

    def __evaluateParticlePositions(self, particles, curr_iter=None, plot=False):
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

        if self.record_extra_info == True:
            curr_outputs = np.zeros((self.n, self.m)) 

        if plot == True and curr_iter == None:
            sys.exit('method requires arg curr_iter if want to plot')

        if plot == True:
            plt.figure(1)
            plt.figure(2)
        x_value = np.zeros(self.n) # int current particle fitnesses/costs storage
        for j in range(0, self.n): 
            particle = particles[j, :] 
            OP = np.copy(self.init_OP) 
            particleIndex = 0
            for signalIndex in range(self.K_index[0], self.K_index[-1]):
                OP[signalIndex] = particle[particleIndex]
                particleIndex += 1 

            if self.sim_model != None:
                PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                      OP, 
                                                      self.t, 
                                                      self.X0) 
            else:
                PV = self.__getSoaOutput(OP) 

            x_value[j] = signalprocessing.cost(self.t, 
                                               PV, 
                                               cost_function_label=self.cost_f, 
                                               st_importance_factor=self.st_importance_factor, 
                                               SP=self.SP).costEval 

            if self.record_extra_info == True:
                # store particle output
                curr_outputs[j, :] = PV
            
            if plot == True:
                plt.figure(1) 
                plt.plot(self.t, PV, c='b') 
                plt.figure(2)
                plt.plot(self.t, OP, c='r')

        if plot == True:
        # get best fitness for analysis
            min_cost_index = np.argmin(x_value)       
            if self.sim_model != None:
                best_PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                           particles[min_cost_index,:], 
                                                           self.t, 
                                                           self.X0) 
            else:
                best_PV = self.__getSoaOutput(particles[min_cost_index, :])      

        
        if plot == True:
            # finalise and save plot
            plt.figure(1)
            plt.plot(self.t, self.SP, c='g', label='Target SP')
            plt.plot(self.t, self.init_PV, c='r', label='Initial Output')
            plt.plot(self.t, best_PV, c='c', label='Best fitness')
            st_index = analyse.ResponseMeasurements(best_PV, self.t).settlingTimeIndex
            plt.plot(self.t[st_index], 
                     best_PV[st_index], 
                     marker='x', 
                     markersize=6, 
                     color="red", 
                     label='Settling Point')
            plt.legend(loc='lower right')
            plt.title('PSO-Optimised Output Signals After ' + str(curr_iter) + \
                ' Generations')
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
        
        return x_value 
                
    def __runPsoAlgorithm(self):
        """
        This method runs the pso algorithm

        Args:
        - 

        Returns:
        - 
        """
        # INITIALISE COPIES OF PARAMETERS WE'RE CHANGING
        # v = np.copy(self.v)
        # x = np.copy(self.x)
        # x_value = np.copy(self.x_value)
        # pbest = np.copy(self.pbest)
        # pbest_value = np.copy(self.pbest_value)
        # gbest = np.copy(self.gbest)
        # curr_iter = np.copy(self.curr_iter)
        # gbest_cost = np.copy(self.gbest_cost)
        # gbest_cost_history = np.copy(self.gbest_cost_history)
        # iter_gbest_reached = np.copy(self.iter_gbest_reached)

        print('################### RUNNING PSO ALGORITHM #####################')

        v = np.copy(self.v)
        x = np.copy(self.x)
        x_value = np.copy(self.x_value)
        pbest = np.copy(self.pbest)
        pbest_value = np.copy(self.pbest_value)
        gbest = np.copy(self.gbest)
        curr_iter = self.curr_iter+1
        gbest_cost = np.copy(self.gbest_cost)
        gbest_cost_history = np.copy(self.gbest_cost_history)
        rt_st_os_analysis = np.copy(self.rt_st_os_analysis)
        iter_gbest_reached = np.copy(self.iter_gbest_reached)
        meta_path_to_pso_data = self.path_to_pso_data

        # # run thru pso multiple times
        curr_rep = 1 
        while curr_rep <= self.rep_max:
            self.path_to_pso_data = self.path_to_pso_data + 'rep' + \
                str(curr_rep) + self.slash
            os.mkdir(self.path_to_pso_data) 
            for g in range(0, self.m):
                if self.embed_init_signal == True:
                    x[0, g] = gbest[g] # embed signal guess

            w = np.ones(self.n) * self.w_init
            c1 = np.ones(self.n) * self.c1
            c2 = np.ones(self.n) * self.c2

            if self.adapt_accel == True:
                rel_improv = np.zeros(self.n)
                c1_max = 2.5 
                c2_max = 2.5 
                c1_min = 0.1
                c2_min = 0.1

            pc_marker = int(0.05*self.iter_max) # for plotting/saving
            if pc_marker == 0:
                pc_marker = 1 

            while curr_iter <= self.iter_max:

                if self.adapt_accel == True:
                    for j in range(0, self.n):
                        # update particle vals
                        rel_improv[j] = (pbest_value[j] - x_value[j]) \
                            / (pbest_value[j] + x_value[j]) 
                        w[j] = self.w_init + ( (self.w_final - self.w_init) * \
                            ((math.exp(rel_improv[j]) - 1) / (math.exp(rel_improv[j]) + 1)) ) 
                        c1[j] = ((c1_min + c1_max)/2) + ((c1_max - c1_min)/2) + \
                            (math.exp(-rel_improv[j]) - 1) / (math.exp(-rel_improv[j]) + 1) 
                        c2[j] = ((c2_min + c2_max)/2) + ((c2_max - c2_min)/2) + \
                            (math.exp(-rel_improv[j]) - 1) / (math.exp(-rel_improv[j]) + 1) 
                
                # update particle velocities
                for j in range(0, self.n):
                    for g in range(0, self.m):
                        v[j, g] = (w[j] * v[j, g]) + (c1[j] * random.uniform(0, 1) \
                            * (pbest[j, g] - x[j, g]) + (c2[j] * \
                                random.uniform(0, 1) * (gbest[g] - x[j,g])))

                # handle velocity boundary violations
                for j in range(0, self.n):
                    for g in range(0, self.m):
                        if v[j, g] > self.v_UB[g]:
                            v[j, g] = self.v_UB[g]
                        if v[j, g] < self.v_LB[g]:
                            v[j, g] = self.v_LB[g]
                
                # update particle positions
                for j in range(0, self.n):
                    x[j, :] = x[j, :] + v[j, :]
                
                # handle position boundary violations
                for j in range(0, self.n):
                    for g in range(0, self.m):
                        if x[j, g] < self.LB[g]:
                            x[j, g] = self.LB[g]
                        elif x[j, g] > self.UB[g]:
                            x[j, g] = self.UB[g]

                # descretise
                for j in range(0, self.n):
                    x[j, :] = self.__discretiseParticlePosition(x[j, :])

                # suppress
                for j in range(0, self.n):
                    x[j, :] = self.__suppressAreasOfSignal(x[j, :])

                # eval particle positions
                if curr_iter % pc_marker == 0 or curr_iter == self.iter_max:
                    # plot/save
                    x_value = self.__evaluateParticlePositions(x, 
                                                               curr_iter=curr_iter, 
                                                               plot=True)
                else:
                    x_value = self.__evaluateParticlePositions(x, 
                                                               curr_iter=curr_iter, 
                                                               plot=False)
                
                # update local best particle positions & fitness vals
                for j in range(0, self.n):
                    if x_value[j] < pbest_value[j]:
                        pbest_value[j] = x_value[j] 
                        for g in range(0, self.m):
                            pbest[j, g] = x[j, g] 
                
                # update global best particle positions & history
                min_cost_index = np.argmin(pbest_value)
                if pbest_value[min_cost_index] < gbest_cost_history[-1]:
                    for g in range(0, self.m):
                        gbest[g] = pbest[min_cost_index, g]
                    rt_st_os_analysis = np.vstack((rt_st_os_analysis, 
                                                   self.__analyseSignal(gbest, 
                                                                        curr_iter)))
                    gbest_cost = pbest_value[min_cost_index]
                    gbest_cost_history = np.append([gbest_cost_history], [gbest_cost])
                    iter_gbest_reached = np.append([iter_gbest_reached], [curr_iter])
                cost_reduction = ((gbest_cost_history[0] - gbest_cost) \
                    / gbest_cost_history[0])*100

                print('Reduced cost by ' + str(cost_reduction) + '% so far')

                self.__savePsoData(x, 
                                   x_value, 
                                   curr_iter, 
                                   pbest, 
                                   gbest_cost_history, 
                                   iter_gbest_reached, 
                                   rt_st_os_analysis) 
                print('Num iterations completed: ' + str(curr_iter) + ' / ' + str(self.iter_max))
                curr_iter += 1 

            # ensure cost and analysis table has final gbest val
            gbest_cost_history = np.append([gbest_cost_history], [gbest_cost])
            self.rt_st_os_analysis = np.vstack((rt_st_os_analysis, 
                                                self.__analyseSignal(gbest, 
                                                                     curr_iter)))
            iter_gbest_reached = np.append([iter_gbest_reached],   
                                           [self.iter_max])

            self.__getPsoPerformance(gbest, 
                                     iter_gbest_reached, 
                                     gbest_cost_history, 
                                     self.rt_st_os_analysis)

            print('Num PSO repetitions completed: ' + str(curr_rep) + ' / ' + str(self.rep_max))
            # reset for next pso run
            v = np.copy(self.v)
            x = np.copy(self.x)
            x_value = np.copy(self.x_value)
            pbest = np.copy(self.pbest)
            pbest_value = np.copy(self.pbest_value)
            gbest_cost_history = []
            gbest_cost_history = np.append([gbest_cost_history], [gbest_cost])
            rt_st_os_analysis = [self.__analyseSignal(gbest, curr_iter)]
            iter_gbest_reached = np.copy(self.iter_gbest_reached)
            self.path_to_pso_data = meta_path_to_pso_data 
            curr_iter = 1

            curr_rep += 1 
            

        print('################### FINISHED PSO ALGORITHM ##################')


    def __getPsoPerformance(self, 
                            gbest, 
                            iter_gbest_reached, 
                            gbest_cost_history, 
                            rt_st_os_analysis):
        """
        This method analyses, plots and saves the pso algorithm's performance

        Args:
        - gbest = best global particle position
        - iter_gbest_reached = array of iterations at which a new gbest was found
        - gbest_cost_history = array of the history of gbest values that were found

        Returns:
        - 
        """
        self.gbest = [i[0] for i in gbest]

        # get best particle position output signal
        if self.sim_model != None:
            self.gbest_PV = self.__getTransferFunctionOutput(self.sim_model, 
                                                             self.gbest, 
                                                             self.t, 
                                                             self.X0) 
        else:
            self.gbest_PV = self.__getSoaOutput(self.gbest) 
    
        # plot final output signal
        plt.figure()
        plt.plot(self.t, self.SP, c='g', label='Target SP')
        plt.plot(self.t, self.init_PV, c='r', label='Initial Output')
        plt.plot(self.t, self.gbest_PV, c='c', label='PSO-Optimised Output')
        st_index = int(rt_st_os_analysis[len(rt_st_os_analysis)-1, 3]) 
        plt.plot(self.t[st_index], 
                 self.gbest_PV[st_index], 
                 marker='x', 
                 markersize=6, 
                 color="red", 
                 label='Settling Point')
        plt.legend(loc='lower right')
        plt.title('Final PSO-Optimised Output Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.savefig(self.path_to_pso_data + 'final_output.png')  
        plt.close()

        # plot final driving signal
        plt.figure()
        plt.plot(self.t, self.init_OP, c='r', label='Initial Input')
        plt.plot(self.t, self.gbest, c='c', label='PSO-Optimised Input')
        plt.legend(loc='lower right')
        plt.title('Final PSO-Optimised Input Signal')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.savefig(self.path_to_pso_data + 'final_input.png')  
        plt.close()

        # plot learning curve
        plt.figure()
        plt.plot(iter_gbest_reached, gbest_cost_history)
        plt.title('PSO Algorithm MSE Learning Curve')
        plt.xlabel('No. Iterations')
        plt.ylabel('Cost ' + str(self.cost_f))
        plt.savefig(self.path_to_pso_data + 'final_learning_curve.png')  
        plt.close()

        # plot how rt, st and os varied with iterations
        max_rt, max_st, max_os = np.amax(rt_st_os_analysis[:,0]),\
                                 np.amax(rt_st_os_analysis[:,1]),\
                                 np.amax(rt_st_os_analysis[:,2]) 
        plt.figure()
        plt.plot(iter_gbest_reached, 
                 rt_st_os_analysis[:,0]/max_rt, 
                 label='Rise Time')
        plt.plot(iter_gbest_reached, 
                 rt_st_os_analysis[:,1]/max_st, 
                 label='Settling Time')
        plt.plot(iter_gbest_reached, 
                 rt_st_os_analysis[:,2]/max_os, 
                 label='Overshoot')
        plt.title('PSO Algorithm RT-ST-OS Learning Curve')
        plt.legend(loc='upper right')
        plt.xlabel('No. Iterations')
        plt.ylabel('Normalised RT/ST/OS')
        plt.savefig(self.path_to_pso_data + 'rtstos_learning_curve.png')  
        plt.close()


        # save key data
        t_df = pd.DataFrame(self.t) 
        init_OP_df = pd.DataFrame(self.init_OP) #
        OP_df = pd.DataFrame(self.gbest) 
        SP_df = pd.DataFrame(self.SP) 
        init_PV_df = pd.DataFrame(self.init_PV) 
        PV_df = pd.DataFrame(self.gbest_PV) 
        iter_gbest_reached_df = pd.DataFrame(iter_gbest_reached) 
        gbest_cost_history_df = pd.DataFrame(gbest_cost_history) 
        rt_st_os_analysis_df = pd.DataFrame(rt_st_os_analysis) 

        t_df.to_csv(self.path_to_pso_data + "time.csv", 
                    index = None, 
                    header=False)
        init_OP_df.to_csv(self.path_to_pso_data + "initial_OP.csv", 
                          index = None, 
                          header=False)
        OP_df.to_csv(self.path_to_pso_data + "optimised_OP.csv", 
                     index = None, 
                     header=False)
        SP_df.to_csv(self.path_to_pso_data + "SP.csv", 
                     index = None, 
                     header=False)
        init_PV_df.to_csv(self.path_to_pso_data + "initial_PV.csv", 
                          index = None, 
                          header=False)
        PV_df.to_csv(self.path_to_pso_data + "optimised_PV.csv", 
                     index = None, 
                     header=False)
        iter_gbest_reached_df.to_csv(self.path_to_pso_data + "iter_gbest_reached.csv", 
                                     index = None, 
                                     header=False)
        gbest_cost_history_df.to_csv(self.path_to_pso_data + "gbest_cost_history.csv", 
                                     index = None, 
                                     header=False)
        rt_st_os_analysis_df.to_csv(self.path_to_pso_data + "rt_st_os_analysis.csv", 
                                    index = None, 
                                    header=False)

        

def run_test(directory_for_run, 
             tf_for_run, 
             t, 
             init_OP, 
             n, 
             iter_max, 
             rep_max, 
             init_v_f, 
             max_v_f, 
             w_init, 
             w_final, 
             adapt_accel, 
             areas_to_suppress, 
             on_suppress_f, 
             embed_init_signal, 
             path_to_embedded_signal, 
             cost_f, 
             st_importance_factor, 
             record_extra_info, 
             linux,
             sp, 
             pso_objs):
    '''
    This function defines the test we want to run for each job in a list of
    python multiprocessing jobs. It is not essential to use this function
    to run a loop of different PSO runs, but this will execute the PSO runs in
    parallel and therefore significantly speed up your experiments.
    '''

    psoObject = PSO(t, 
                    init_OP, 
                    n, 
                    iter_max, 
                    rep_max, 
                    init_v_f, 
                    max_v_f, 
                    w_init=w_init, 
                    w_final=w_final, 
                    adapt_accel=True, 
                    areas_to_suppress='pisic_shape', 
                    on_suppress_f=on_suppress_f, 
                    embed_init_signal=True, 
                    path_to_embedded_signal=None, 
                    directory=directory_for_run, 
                    cost_f=cost_f, 
                    st_importance_factor=st_importance_factor, 
                    sim_model=tf_for_run, 
                    record_extra_info=True, 
                    SP=sp)
    
    pso_objs.append(psoObject)

































if __name__ == '__main__':
    '''
    Below is an example implementation of the above PSO code. Note that there 
    are 2 main modes of using this PSO implementation:

    1) Simulation (using a transfer function that simulates SOAs)
    2) Experimental 

    To use the experimental setup, you will need all the same equipment, modules,
    specific GPIB addresses etc. that were used in the Connet lab in UCL's
    EEE Robert's building (contact the Optical Networks Group for more info).
    Users outside of ONG will need to write code to interface with their
    own equipment.

    To use the simulation (i.e. the transfer function), the user should not need
    to write any code themselves. Simply changing the below 'directory' variable
    to point this programme to where to store data should be sufficient. 

    The below code runs a PSO simulation, where PSO is optimising 10 different
    SOA transfer functions in parallel. Users can play around with the PSO
    hyperparameters to control PSO performance, convergence properties, 
    run time etc., and can also distort the transfer function by adjusting 
    the distortion coefficients or even implement their own transfer functions
    to simulate their custom SOAs. By optimising different transfer functions,
    users will be able to see how well PSO is generalising to different SOAs.

    While this code is not the 'cleanest', we have tried to insert clear comments
    so that a user wishing to delve deeper into the use of this PSO implementation 
    (beyond running simple transfer function simulations) can follow the logic 
    and implement the same functionality in their own programmes. 

    As a general rule-of-thumb, increasing n (the number of particles) is a 
    reliable way to improve PSO performance and find more optimal solutions.
    '''

    import soa



    # set dir to save data
    directory = os.path.dirname(soa.__file__)+'/../data/'

    # specify whether running simulation or experiment
    sim = True

    # specify if using linux (or mac) (for backslash or forward slash dirs)
    linux = True

    num_points = 240
    time_start = 0.0
    time_stop = 20e-9 
    t = np.linspace(time_start,time_stop,num_points)
    init_OP = np.zeros(num_points)

    # config pso params
    n = 3 
    iter_max = 150
    rep_max = 1
    max_v_f = 0.05
    init_v_f = max_v_f
    cost_f = 'mSE'
    st_importance_factor = None
    w_init = 0.9
    w_final = 0.5
    on_suppress_f = 2.0

    if sim == True:
        # define transfer function(s)
        # N.B. init_OP low must be -1 for tf
        init_OP[:int(0.25*num_points)],init_OP[int(0.25*num_points):] = -1, 0.5
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

        # set up simulation(s) you want to run
        init_PV = distort_tf.getTransferFunctionOutput(tf,init_OP,t)
        sp = analyse.ResponseMeasurements(init_PV, t).sp.sp
        tfs, _ = distort_tf.gen_tfs(num_facs=[1.0,1.2,1.4], 
                                    a0_facs=[0.8],
                                    a1_facs=[0.7,0.8,1.2],
                                    a2_facs=[1.05,1.1,1.2],
                                    all_combos=False)

        pso_objs = multiprocessing.Manager().list()
        jobs = []
        test_nums = [test+1 for test in range(len(tfs))]
        direcs = [directory + '/test_{}'.format(test_num) for test_num in test_nums]
        for tf, direc in zip(tfs, direcs):
            if os.path.exists(direc) == False:
                os.mkdir(direc)
            p = multiprocessing.Process(target=run_test, 
                                        args=(direc, 
                                              tf, 
                                              t, 
                                              init_OP, 
                                              n, 
                                              iter_max, 
                                              rep_max, 
                                              init_v_f, 
                                              max_v_f, 
                                              w_init, 
                                              w_final, 
                                              True, 
                                              'pisic_shape', 
                                              on_suppress_f, 
                                              True, 
                                              None, 
                                              cost_f, 
                                              st_importance_factor, 
                                              True, 
                                              linux,
                                              sp, 
                                              pso_objs,))
            
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()

        # plot composite graph
        pso_objs = list(pso_objs)
        plt.figure()
        plt.plot(t, sp, color='green')
        for pso_obj in pso_objs:
            plt.plot(t, pso_obj.gbest_PV)
        plt.show()


        # pickle data
        PIK = directory + '/pickle.dat'
        data = pso_objs
        with open(PIK, 'wb') as f:
            pickle.dump(data, f)





    else:
        # set up experiment(s) you want to run
        directory = r"C:\Users\onglab\Desktop\SOA_project\Chris\pso_no_fall_test_09012020" 
        awg = devices.TektronixAWG7122B("GPIB1::1::INSTR")
        osc = devices.Agilent86100C("GPIB1::7::INSTR")
        osc.set_acquire(average=True, count=30, points=num_points)
        osc.set_timebase(position=4.2e-8, range_=time_stop-time_start)

        init_OP[:60], init_OP[60:] = -0.5, 0.5 

        psoObject = pso(t, 
                        init_OP, 
                        n, 
                        iter_max, 
                        rep_max, 
                        init_v_f, 
                        max_v_f, 
                        w_init=w_init, 
                        w_final=w_final, 
                        adapt_accel=True, 
                        areas_to_suppress='start_centre', 
                        on_suppress_f=on_suppress_f, 
                        embed_init_signal=True, 
                        path_to_embedded_signal=None, 
                        directory=directory, 
                        cost_f=cost_f, 
                        st_importance_factor=st_importance_factor, 
                        awg=awg, 
                        osc=osc) 
