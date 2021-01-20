import numpy as np
import matplotlib.pyplot as plt

class ResponseMeasurements:
    def __init__(self,signal,t,gradientPoints=8,percentage=5,hopSize=1,SP=None):
        """
        60 points: gradientPoints=8
        240 points: gradientPoints=8
        1350 points: gradientPoints=50
        """
        self.signal = signal
        self.t = t
        self.gradientPoints = gradientPoints
        self.percentage = percentage/100
        self.hopSize = hopSize
        self.dt = abs(self.t[len(self.t)-1] - self.t[len(self.t)-2])
        self.SP = SP # (optional) provide SP to compare signal to

        # num points from end to start measuring high ss (approx 30 for 240 pt signal):
        self.start_measurement_index = int(0.125 * len(self.signal)) 
        # num points from end to stop measuring high ss (approx 20 for 240 pt signal):
        self.end_measurement_index = int(0.08333 * len(self.signal)) 

        self.__getMeasurements()

    def __getMeasurements(self):
        self.__getInflectionTimeIndex()
        self.__getSettlingMaxIndex()
        self.__getSettlingTimeIndex()
        self.__getSSHighValue()
        self.__getSSLowValue()
        self.__getSP()
        self.__getRiseTime()
        self.__getOvershoot()

    def __getSP(self):
        self.sp = SetPoint(self)

    def __getInflectionTimeIndex(self):
        if self.SP is None:
            grad = []
            n = len(self.signal)        #length of signal
            for i in range(n):          
                if i + self.gradientPoints < n:     
                    grad.append(abs(self.signal[i] \
                        - self.signal[i+self.gradientPoints]))
                else:
                    break
            self.inflectionTimeIndex = np.argmax(grad)
        else:
            # SP is provided
            for index in range(0, len(self.SP)):
                if self.SP[index] > self.SP[0]:
                    self.inflectionTimeIndex = index
                    break

    def __getSettlingMaxIndex(self):
        self.settlingMaxIndex = np.argmax(self.signal)

    def __getSettlingTimeIndex(self):
        if self.SP is None:
            # don't have reference SP 
            ss_high = self.signal[self.settlingMaxIndex] 
            n = len(self.signal)
            signal_end = np.mean(self.signal[len(self.signal)-\
                self.start_measurement_index:len(self.signal)-\
                    self.end_measurement_index]) 

            for i in range(0, len(self.signal)-self.start_measurement_index):
                if np.max(abs(np.asarray(self.signal[i:len(self.signal)-\
                    self.end_measurement_index]) - \
                        signal_end)) <= self.percentage * signal_end:
                    self.settlingTimeIndex = i
                    self.settlingTime = (i-self.inflectionTimeIndex)*abs(self.t[0] -\
                         self.t[1]) 
                    break
                else:
                    self.settlingTimeIndex = len(self.signal) \
                        - self.start_measurement_index
                    self.settlingTime = self.t[-1] # signal never settles 

            if self.settlingTime < 0:
                # signal is unrecognisable/not a step therefore cannot have settled
                self.settlingTimeIndex = len(self.signal) \
                    - self.start_measurement_index
                self.settlingTime = self.t[-1] # signal never settles

        else: 
            # SP is provided
            n = len(self.signal)
            signal_end = self.SP[-1]

            for i in range(0, len(self.signal)-self.start_measurement_index):
                if np.max(abs(np.asarray(self.signal[i:len(self.signal)-\
                    self.end_measurement_index]) \
                        - signal_end)) <= self.percentage * signal_end:
                    self.settlingTimeIndex = i
                    self.settlingTime = (i-self.inflectionTimeIndex)*abs(self.t[0] \
                        - self.t[1]) 
                    break

                else:
                    self.settlingTimeIndex = len(self.signal) \
                        - self.start_measurement_index
                    self.settlingTime = self.t[-1] # signal never settles 

            if self.settlingTime < 0:
                # signal is unrecognisable/not a step therefore cannot have settled
                self.settlingTimeIndex = len(self.signal) \
                    - self.start_measurement_index
                self.settlingTime = self.t[-1] # signal never settles


 

    def __getSSHighValue(self):
        
        if self.SP is None:
            self.SSHighValue = np.mean(self.signal[len(self.signal)\
                -self.start_measurement_index:len(self.signal)\
                    -self.end_measurement_index]) 

        else:
            self.SSHighValue = self.SP[-1]

 

    def __getSSLowValue(self):

        if self.SP is None:
            self.SSLowValue = np.mean(self.signal[:self.inflectionTimeIndex])
 
        else:
            self.SSLowValue = self.SP[0]


    def __getRiseTime(self):
        
        off_set = self.sp.sp[0] # get amount signal is offset from 0 by

        i=self.inflectionTimeIndex
        prev_diff = abs((self.signal.copy()[i]-off_set) \
            - (((self.SSHighValue-off_set)*0.1)))
        curr_diff = abs((self.signal.copy()[i+1]-off_set) \
            - (((self.SSHighValue-off_set)*0.1)))
        while curr_diff < prev_diff:
            prev_diff = abs((self.signal.copy()[i]-off_set) \
                - (((self.SSHighValue-off_set)*0.1)))
            curr_diff = abs((self.signal.copy()[i+1]-off_set) \
                - (((self.SSHighValue-off_set)*0.1)))
            i += 1
            if i == len(self.signal)-2:
                break
        self.idxTen = i-1

        i=self.inflectionTimeIndex
        prev_diff = abs((self.signal.copy()[i]-off_set) \
            - (((self.SSHighValue-off_set)*0.9)))
        curr_diff = abs((self.signal.copy()[i+1]-off_set) \
            - (((self.SSHighValue-off_set)*0.9)))
        while curr_diff < prev_diff:
            prev_diff = abs((self.signal.copy()[i]-off_set) \
                - (((self.SSHighValue-off_set)*0.9)))
            curr_diff = abs((self.signal.copy()[i+1]-off_set) \
                - (((self.SSHighValue-off_set)*0.9)))
            i += 1
            if i == len(self.signal)-2:
                break
        self.idxNinety = i-1
        timeTen = self.t[self.idxTen]
        timeNinety = self.t[self.idxNinety]
        self.riseTime = (timeNinety - timeTen)

        if self.idxTen == self.idxNinety:
            # signal never rose
            self.riseTime = 1000 

    def __getOvershoot(self):
        self.overshoot = abs(float((self.signal[self.settlingMaxIndex] \
            - self.SSHighValue)/self.SSHighValue))


class SetPoint:
    def __init__(self,response):
        self.response = response
        self.__getSetPoint()

    def __getSetPoint(self):
        self.sp = np.zeros(len(self.response.signal))
        self.inflectionTimeIndex = self.response.inflectionTimeIndex
        self.sp[:self.inflectionTimeIndex] = self.response.SSLowValue
        self.sp[self.inflectionTimeIndex:] = self.response.SSHighValue