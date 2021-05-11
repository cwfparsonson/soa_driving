import numpy as np
import matplotlib.pyplot as plt

class ResponseMeasurements:

    def __init__(self, signal, t, gradientPoints=1, percentage=1, hopSize = 1):

        self.signal = signal
        self.t = t
        self.gradientPoints = gradientPoints
        self.percentage = percentage/100
        self.hopSize = hopSize
        self.dt = abs(self.t[len(self.t)-1] - self.t[len(self.t)-2])

        self.__getMeasurements()

    def __getMeasurements(self):

        self.__getInflectionTimeIndex()

    def __getInflectionTimeIndex(self):

        grad = []
        n = len(self.signal)
        for i in range(n):
            if i + self.gradientPoints < n:
                grad.append(abs(self.signal[i] - self.signal[i+self.gradientPoints]))
            else:
                break

        self.inflectionTimeIndex = np.argmax(grad)


