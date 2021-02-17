import sys, os

from soa import devices, get_fopdt_params, analyse

import pyvisa as visa
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import soa.analyse

class generateSignal:
    """
    Args:
    - num_points = number of points you want in signal 
    - t_start = time to start signal period
    - t_stop = time to stop signal period
    - save_signal = True or False for saving generated signal
    - directory = where to save generated signal

    """

    def __init__(self, 
                 num_points=240, 
                 t_start=0.0, 
                 t_stop=20e-9, 
                 save_signal=True, 
                 directory=None):
        self.num_points = num_points
        self.t_start = t_start
        self.t_stop = t_stop
        self.save_signal = save_signal
        self.directory = directory + "\\"
        
        self.t = np.linspace(self.t_start, self.t_stop, self.num_points) 
        self.signal = np.zeros(len(self.t))

        if save_signal == True:
            t_df = pd.DataFrame(self.t)
            t_df.to_csv(self.directory + "time.csv", index = None, header=False)

    def square(self, off_v=-0.5, on_v=0.5, duty_cycle=0.5, signal_name='square'):
        """
        Generates square driving signal

        Args:
        - off_v = voltage in off state
        - on_v = voltage in on state 
        - duty_cycle = fraction of time signal should be on for
        - signal_name = name you want to save signal data under
        """
        self.signal = np.zeros(len(self.t)) 

        num_points_on = int(duty_cycle * len(self.t)) 
        num_points_off = int((1-duty_cycle) * len(self.t)) 

        self.signal[0:int(num_points_off/2)] = off_v
        self.signal[int(num_points_off/2):int(num_points_off/2)+num_points_on] = on_v
        self.signal[int(num_points_off/2)+num_points_on:] = off_v

        self.__checkSignal() # check signal within acceptable range
        
        if self.save_signal == True:
            self.__saveSignal(signal_name)

        return self.signal

    def pisic(self, 
              off_v=-0.5, 
              on_v=0.5, 
              impulse_v=0.2, 
              duty_impulse=0.05, 
              duty_cycle=0.5, 
              signal_name='pisic'):
        """
        Generates pisic driving signal

        Args:
        - off_v = voltage in off state
        - on_v = voltage in on state 
        - impulse_v = pre-impulse voltage (added to on_v at signal leading edge)
        - duty_impulse = fraction of time period pre impulse be should on for
        - duty_cycle = fraction of time period signal should be on for
        - signal_name = name you want to save signal data under

        """
        self.signal = np.zeros(len(self.t)) 

        # gen square
        num_points_on = int(duty_cycle * len(self.t)) 
        num_points_off = int((1-duty_cycle) * len(self.t)) 
        self.signal[0:int(num_points_off/2)] = off_v
        self.signal[int(num_points_off/2):int(num_points_off/2)+num_points_on] = on_v
        self.signal[int(num_points_off/2)+num_points_on:] = off_v

        # add pre-impulse to leading edge
        num_points_pisic = int(duty_impulse * len(self.t))
        self.signal[int(num_points_off/2):int(num_points_off/2)+num_points_pisic] = on_v + impulse_v

        self.__checkSignal() # check signal within acceptable range
        
        if self.save_signal == True:
            self.__saveSignal(signal_name)

        return self.signal

    def misic(self, 
              off_v=-0.5, 
              on_v=0.5, 
              impulse_v=0.2, 
              duty_cycle=0.5, 
              num_misic_bits=100, 
              bit_rate=240e-12, 
              signal_name='misic1'):
        """
        Generates misic driving signal

        Args:
        - off_v = voltage in off state
        - on_v = voltage in on state 
        - impulse_v = voltage of misic impulses
        - duty_cycle = fraction of time period signal is on
        - num_misic_bits = total number of bits in misic signal (paper used 100)
        - bit_rate = (enter in ps i.e. units of e-12) rate at which misic 
        impulses/bits are repeated (in paper, misic pulses repeated every 
        240 ps == 4.16 Gb/s)
        - signal_name = name you want to save signal data under. Can be misic1, 
        misic2, misic3, misic4, misic5 or misic6
        """
        self.signal = np.zeros(len(self.t)) 

        # gen square
        num_points_on = int(duty_cycle * len(self.t)) 
        num_points_off = int((1-duty_cycle) * len(self.t)) 
        self.signal[0:int(num_points_off/2)] = off_v
        self.signal[int(num_points_off/2):int(num_points_off/2)+num_points_on] = on_v
        self.signal[int(num_points_off/2)+num_points_on:] = off_v

        # define misic params     
        while num_misic_bits % 100 != 0:
            print('WARNING: Please enter number of misic bits as a multiple of \
                100 (e.g. 200).')
            num_misic_bits = input('Num misic bits: ')

        # from paper, 6% of bits were for pre impulse (section A)
        num_bits_sectA = int(0.06*num_misic_bits)
        # from paper, 8% of bits were for middle section (section B)
        num_bits_sectB = int(0.08*num_misic_bits)
        # from paper, 86% of bits were from end section (section C)
        num_bits_sectC = int(0.86*num_misic_bits)

        time_period = self.t_stop - self.t_start 
        time_per_index = time_period / self.num_points 
        # calc number of indices (to nearest integer) in signal that make up one 
        # misic bit N.B. Dividing bit rate by 3 since that is what seems to have 
        # been done in misic paper....
        indices_per_bit = int((bit_rate/3) / time_per_index)
        if indices_per_bit < 1:
            indices_per_bit = 1 

        # gen misic bit sequence
        while signal_name != 'misic1' and \
            signal_name != 'misic2' and \
            signal_name != 'misic3' and \
            signal_name != 'misic4' and \
            signal_name != 'misic5' and \
            signal_name != 'misic6':
            print('WARNING: Must enter a valid misic format. Please enter \
                misic1, misic2, misic3, misic4, misic5 or misic6.')
            signal_name = input('Misic format: ')
        
        if signal_name == 'misic1':
            sectA_bit_pattern = [1,1,1,1,1,1]
            sectB_bit_pattern = [1,1,0,1,0,1,0,1]
            sectC_bit_pattern = [0,1,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)
        
        elif signal_name == 'misic2':
            sectA_bit_pattern = [1,1,1,1,1,1]
            sectB_bit_pattern = [0,1,1,0,1,1,0,1]
            sectC_bit_pattern = [0,1,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)

        elif signal_name == 'misic3':
            sectA_bit_pattern = [1,1,1,1,1,1]
            sectB_bit_pattern = [0,1,0,1,0,1,0,1]
            sectC_bit_pattern = [0,1,0,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)

        elif signal_name == 'misic4':
            sectA_bit_pattern = [0,1,0,1,0,1]
            sectB_bit_pattern = [0,1,0,1,0,1,1,1]
            sectC_bit_pattern = [0,1,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)

        elif signal_name == 'misic5':
            sectA_bit_pattern = [0,1,1,0,1,1]
            sectB_bit_pattern = [0,1,0,1,0,1,1,1]
            sectC_bit_pattern = [0,1,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)


        elif signal_name == 'misic6':
            sectA_bit_pattern = [1,1,1,1,1,1]
            sectB_bit_pattern = [1,1,1,1,1,1,1,1]
            sectC_bit_pattern = [1,1,1]
            misic_bits = self.__generateMisicBitSequence(sectA_bit_pattern, 
                                                         sectB_bit_pattern, 
                                                         sectC_bit_pattern, 
                                                         num_bits_sectA, 
                                                         num_bits_sectB, 
                                                         num_bits_sectC, 
                                                         signal_name)

        # convert misic bits into specified bit rate
        misic_impulses = np.zeros(indices_per_bit * num_misic_bits) 
        for bit in range(0, int(len(misic_bits))):
            counter = 0 
            for index in range(0, indices_per_bit):
                misic_impulses[bit+counter] = on_v + (misic_bits[bit] * impulse_v)
                counter += 1 

        # add misic impulses to square
        self.signal[int(num_points_off/2):int(num_points_off/2)+int(len(misic_impulses))] = misic_impulses[:36]

        self.__checkSignal() # check signal within acceptable range
        
        if self.save_signal == True:
            self.__saveSignal(signal_name)

        return self.signal

    def __plotSignal(self, signal_name):
        """
        Plots and displays drive signal

        Args:
        - signal_name = name you want to save signal data under
        """
        plt.figure()
        plt.plot(self.t, self.signal, label='Generated signal')
        plt.legend(loc='lower right')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.title(signal_name)
    
    def __saveSignal(self, signal_name):
        """
        Saves signal as csv and as image
        """
        # SAVE IMAGE
        self.__plotSignal(signal_name)
        plt.savefig(self.directory + signal_name + ".png")
        plt.close()

        # SAVE CSV
        signal_df = pd.DataFrame(self.signal)
        signal_df.to_csv(self.directory + signal_name + ".csv", 
                         index = None, 
                         header=False)

    def __checkSignal(self):
        """
        Checks if signal is within acceptable range for awg
        """
        max_v = 1.0 # max allowed drive voltage
        min_v = -1.0 # min allowed drive voltage

        for index in range(0, self.num_points):
            if self.signal[index] > max_v:
                self.signal[index] = max_v
        for index in range(0, self.num_points):
            if self.signal[index] <min_v:
                self.signal[index] = min_v

    def __generateMisicBitSequence(self, 
                                   sectA_bit_pattern, 
                                   sectB_bit_pattern, 
                                   sectC_bit_pattern, 
                                   num_bits_sectA, 
                                   num_bits_sectB, 
                                   num_bits_sectC, 
                                   signal_name):
        """
        Function generates misic bit sequence

        Args:
        - sectA_bit_pattern = bit pattern for section A (e.g. [1,1,1,1,1,1])
        - sectB_bit_pattern = e.g. [1,1,0,1,0,1,0,1]
        - sectC_bit_patter = e.g. [0,1,1]
        """
        sectA_bits = np.zeros(num_bits_sectA)
        sectB_bits = np.zeros(num_bits_sectB)
        sectC_bits = np.zeros(num_bits_sectC)

        # sect a
        sectA_pattern_length = int(len(sectA_bit_pattern))
        for sectA_index in range(0, num_bits_sectA, sectA_pattern_length):
            i = 0 
            for pattern_index in range(0, sectA_pattern_length):
                sectA_bits[sectA_index+i]  = sectA_bit_pattern[pattern_index]
                i += 1 

        # sect b
        sectB_pattern_length = int(len(sectB_bit_pattern))
        for sectB_index in range(0, num_bits_sectB, sectB_pattern_length):
            i = 0 
            for pattern_index in range(0, sectB_pattern_length):
                sectB_bits[sectB_index+i]  = sectB_bit_pattern[pattern_index]
                i += 1 

        # sect c
        sectC_pattern_length = int(len(sectC_bit_pattern))
        if signal_name == 'misic2':
            # special case
            for sectC_index in range(1, num_bits_sectC, sectC_pattern_length):
                i = 0 
                for pattern_index in range(1, sectC_pattern_length):
                    sectC_bits[sectC_index+i]  = sectC_bit_pattern[pattern_index]
                    i += 1 
                    if sectC_index+i >= num_bits_sectC:
                        # reached end
                        break
            sectC_bits[0] = 1
            sectC_bits[-3:] = [1,1,0]

        elif signal_name == 'misic4':
            # special case
            for sectC_index in range(1, num_bits_sectC, sectC_pattern_length):
                i = 0 
                for pattern_index in range(1, sectC_pattern_length):
                    sectC_bits[sectC_index+i]  = sectC_bit_pattern[pattern_index]
                    i += 1 
                    if sectC_index+i >= num_bits_sectC:
                        # reached end
                        break
            sectC_bits[0] = 1
            sectC_bits[-3:] = [0,1,1]

        elif signal_name == 'misic5':
            # special case
            for sectC_index in range(2, num_bits_sectC, sectC_pattern_length):
                i = 0 
                for pattern_index in range(1, sectC_pattern_length):
                    sectC_bits[sectC_index+i]  = sectC_bit_pattern[pattern_index]
                    i += 1 
                    if sectC_index+i >= num_bits_sectC:
                        # reached end
                        break
            sectC_bits[0] = 1
            sectC_bits[1] = 1

        else:
            # normal case
            for sectC_index in range(0, num_bits_sectC, sectC_pattern_length):
                i = 0 
                for pattern_index in range(0, sectC_pattern_length):
                    sectC_bits[sectC_index+i]  = sectC_bit_pattern[pattern_index]
                    i += 1 
                    if sectC_index+i >= num_bits_sectC:
                        # reached end
                        break

        # return misic bit stream
        misic_bits = np.zeros(len(sectA_bits) + len(sectB_bits) + len(sectC_bits))
        for i in range(0, len(sectA_bits)):
            misic_bits[i] = sectA_bits[i]
        for i in range(len(sectA_bits), len(sectA_bits)+len(sectB_bits)):
            misic_bits[i] = sectB_bits[i-len(sectA_bits)]
        for i in range(len(sectA_bits)+len(sectB_bits), len(misic_bits)):
            misic_bits[i] = sectC_bits[i-(len(sectA_bits)+len(sectB_bits))]

        return misic_bits

class soaOutput:
    """
    Class for getting and analysing soa output

    Args:
    - awg = awg object
    - osc = osc object
    - num_points = number of points in signal
    - time_start = time signal period starts
    - time_stop = time signal period stops
    - save_siganl = if want to save soa output signal
    - directory = where to save data#
    """
    
    def __init__(self, 
                 awg, 
                 osc, 
                 num_points=240, 
                 t_start=0.0, 
                 t_stop=20e-9, 
                 save_signal=True, 
                 directory=None):
        self.awg = awg
        self.osc = osc
        self.num_points = num_points
        self.t_start = t_start
        self.t_stop = t_stop
        self.save_signal = save_signal
        self.directory = directory + "\\"

        self.t = np.linspace(self.t_start, 
                             self.t_stop, 
                             self.num_points)
        self.signal = np.zeros(int(len(self.t))) 

    def getSoaOutput(self, drive_signal, signal_name='Unknown'):
        """
        This method sends a drive signal to the SOA and gets an soa output

        Args:
        - drive_signal = signal you want to drive soa with
        - signal_name = name under which you want to save soa output data

        Returns:
        - soa_output = output of soa
        """
        self.signal = np.zeros(int(len(self.t))) 

        self.awg.send_waveform(drive_signal, suppress_messages=True)
        time.sleep(3)
        self.signal = self.osc.measurement(channel=1)

        if self.save_signal == True:
            self.__saveSignal(signal_name)

        return self.signal

    def __plotSignal(self, signal_name):
        """
        Plots and displays drive signal

        Args:
        - signal_name = name you want to save signal data under
        """
        plt.figure()
        plt.plot(self.t, self.signal, label='SOA output')
        plt.legend(loc='lower right')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(signal_name)

    def __saveSignal(self, signal_name):
        """
        Saves signal as csv and as image
        """
        # SAVE IMAGE
        self.__plotSignal(signal_name)
        plt.savefig(self.directory + signal_name + ".png")
        plt.close()

        # SAVE CSV
        signal_df = pd.DataFrame(self.signal)
        signal_df.to_csv(self.directory + signal_name + ".csv", 
                         index = None, 
                         header=False)


class cost:
    """
    This class evaluates the cost of a signal output PV given some target output 
    SP. The higher the cost, the poorer the signal performed. There are various 
    ways to evaluate cost, each of which value different aspects of the signal 
    quality in different ways

    Parameters to enter:
    - PV = process variable i.e. an array of signal amplitudes
    - SP = set point i.e. an array of target signal amplitudes

    Parameters to return:
    - meanSquaredError = the sum of the difference between PV and SP squared
    - zweeLeePerformanceCriterion = weighted sum of the stepinfo information 
    (rise time, settling time, overshoot and steady state error). Set beta value 
    to determine how cost should be evaluated. Beta < 0.7 prioritises quick rise 
    time and settling time, Beta > 0.7 prioritises low overshoot and steady state 
    error

    """

    def __init__(self, 
                 t, 
                 PV=None, 
                 cost_function_label=None, 
                 st_importance_factor=None, 
                 SP=None):
        self.t = t
        self.PV = PV
        self.cost_function_label = cost_function_label 
        # set factor by which want settling time to be larger/smaller than mean 
        # squared error. Only (and must) specify if using s_mse+st cost_f:
        self.st_importance_factor = st_importance_factor 
        self.SP = SP 
        # num points at end of signal output to average over to find steady state:
        self.n_steady_state = 5 

        # find steady state and off indices
        if self.PV[-1] <= (2*np.mean(self.PV[:5])):
            # signal has fallen at end
            for i in range(int(len(self.PV)/2), int(len(self.PV))):
                if self.PV[i] >= (2*np.mean(self.PV[:5])):
                    self.index_signal_off = i
                    break
                else:
                    # signal never settles
                    self.index_signal_off = int(len(self.PV))
            # go back e.g. 10 points and use that to set steady state
            start_index = self.index_signal_off-10
            end_index = self.index_signal_off-5
            self.y_ss = np.mean(self.PV[start_index:end_index]) # steady stae

            # find more accurately where index turned off
            for i in range(int(len(self.PV)-1), 0, -1):
                if self.PV[i] >= (0.95*self.y_ss):
                    # signal is within 5% of steady state
                    self.index_signal_off = i
                    break
        else:
            # signal has not fallen to 0
            self.y_ss = np.mean(PV[-self.n_steady_state:]) 

        self.num_points = int(len(self.PV))
        self.measurements = get_fopdt_params.ResponseMeasurements(self.PV, 
                                                                  self.t)
        self.index_signal_on = self.measurements.inflectionTimeIndex 

        if self.cost_function_label == 'mSE':
            if self.SP is None:
                print('ERROR: To evaluate MSE, need to provide a set point to \
                    cost object. Please provide a set point (in the form of a \
                        step function)')
            self.costEval = self.__getMeanSquaredError(self.SP)

        elif self.cost_function_label == 'st':
            responseMeasurementsObject = analyse.ResponseMeasurements(self.PV, 
                                                                      self.t, 
                                                                      SP=self.SP)
            self.settlingTime = responseMeasurementsObject.settlingTime
            self.settlingTimeCost = self.settlingTime

            self.costEval = self.settlingTimeCost 
            
        elif cost_function_label == 'mSE+st':
            self.mseCost = self.__getMeanSquaredError(self.SP)

            responseMeasurementsObject = analyse.ResponseMeasurements(self.PV, 
                                                                      self.t, 
                                                                      SP=self.SP)
            self.settlingTime = responseMeasurementsObject.settlingTime
            self.settlingTimeCost = self.settlingTime
            
            self.costEval = self.mseCost + self.settlingTimeCost 

        elif cost_function_label == 's_mse+st':
            self.mse = self.__getMeanSquaredError(self.SP)
            responseMeasurementsObject = analyse.ResponseMeasurements(self.PV, 
                                                                      self.t, 
                                                                      SP=self.SP)
            self.settlingTime = responseMeasurementsObject.settlingTime
            self.settlingTimeCost = self.st_importance_factor * self.settlingTime

            # scale st to some importance factor 
            self.costEval = (self.settlingTimeCost * self.mse) / (self.settlingTimeCost + self.mse)

        elif cost_function_label == 'mse+st+os':
            self.mseCost = self.__getMeanSquaredError(self.SP) * 10**3
            responseMeasurementsObject = analyse.ResponseMeasurements(self.PV, 
                                                                      self.t, 
                                                                      SP=self.SP)
            self.settlingTime = responseMeasurementsObject.settlingTime
            self.overShoot = abs(responseMeasurementsObject.overshoot) * 10**(-10)
            
            self.costEval = self.mseCost + self.settlingTime + self.overShoot 
            



        elif cost_function_label == 'zlpc':
            beta = 0.9 # set beta parameter
            responseMeasurementsObject = analyse.ResponseMeasurements(self.PV, 
                                                                      self.t, 
                                                                      SP=self.SP)
            self.overShoot = responseMeasurementsObject.overshoot
            self.riseTime = responseMeasurementsObject.riseTime
            self.settlingTime = responseMeasurementsObject.settlingTime
            self.mse = self.__getMeanSquaredError(self.SP)

            self.costEval = ((1-math.exp(-beta)) * \
                (self.overShoot + (self.mse))) + ((math.exp(-beta)) * \
                    (self.settlingTime - self.riseTime))


        else:
            print('Must specify cost function to use!')


    def getSetPoint(self):
        """
        This method gets the set point of a given signal

        Args:
        - PV = signal to use to get set point

        Returns:
        - set point / target signal
        """

        SP = analyse.ResponseMeasurements(self.PV, self.t).sp.sp

        return SP

    def __getMeanSquaredError(self, SP):
        """
        This function calcs the mean squared error for a given signal

        Args:
        - SP = set point / target signal to use

        Returns:
        - mean squared error
        """

        # GET MSE
        error = np.subtract(self.PV, SP) 
        error_squared = np.square(error) 
        meanSquaredError = np.mean(error_squared) 

        return meanSquaredError





