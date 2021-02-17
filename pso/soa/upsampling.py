import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import sys
import pandas as pd
from soa import analyse

class ups:
    
    # each upsampling instance can be set to have different points
    def __init__(self, points):
        self.points = points
    

    # main method of upsampling class
    def create(self, *args):
    # *args: list as input     
        input = np.array(*args)
        
        if input.size != self.points:
            n = self.points // input.size
            
            new_input = np.repeat(input, n)
            
            return new_input
        
        else:
            return input