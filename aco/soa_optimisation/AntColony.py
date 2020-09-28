import numpy as np
# from utilz import *
import os
from Ant import Ant
import pandas as pd
from threading import Thread
import time
import random as rnd
#TODO: ADD NON-REPEAT OPTION AND TEST/CONSTRUCT GRAPH THAT IS INHERENTLY NON-REPEAT

class AntColony:

    def __init__(self, \
                ACOStateFilePath=None, \
                graph=None, \
                numberOfAnts=None, \
                generations=None, \
                metricFunction=None, \
                evaporationConstant=0.5, \
                pheremoneExponent=1.0, \
                heuristicExponent=1.0, \
                pheremoneMatrix=None, \
                heuristicMatrix=None, \
                Q=None, \
                
                repeatNode=True, \

                #TODO: change this to be a set of nodes optionally
                universalStartingNode=None, \

                explorationProbability=None, \

                onlyBestAntUpdate=False, \
                
                #TODO: default to number of nodes
                movesPerGeneration=None, \

                #TODO: consider either json or just save object directly   
                saveFilePath=None, \
                ):
                
        if ACOStateFilePath is not None:#metric function
            self.ACOStateFilePath = ACOStateFilePath
            self.function = metricFunction
            self.onlyBestAntUpdate = onlyBestAntUpdate
            self.explorationProbability = explorationProbability
            self.__initializeFromACOStateFile()
        
        else:
            
            self.start = -1
            self.graph = graph
            self.numberOfAnts = numberOfAnts
            self.generations = generations

            self.function = metricFunction

            self.evaporationConstant = evaporationConstant
            self.pheremoneExponent = pheremoneExponent
            self.heuristicExponent = heuristicExponent

            if pheremoneMatrix is not None:
                self.pheremoneMatrix = pheremoneMatrix
            else:
                self.pheremoneMatrix = (self.graph != 0).astype(float)

            if heuristicMatrix is not None:
                self.heuristicMatrix = heuristicMatrix
            else:
                self.heuristicMatrix = np.ones(self.graph.shape)

            self.Q = Q

            self.repeatNode = repeatNode
            self.universalStartingNode = universalStartingNode

            self.explorationProbability = explorationProbability

            self.onlyBestAntUpdate = onlyBestAntUpdate
            
            #make sure that not taking more moves than allowed if ants cannot re-visit nodes
            if self.repeatNode == False and movesPerGeneration >= len(self.graph):
                self.movesPerGeneration = len(self.graph) - 1
            else:
                self.movesPerGeneration = movesPerGeneration

            self.saveFilePath = saveFilePath
            if self.saveFilePath is not None:
                if not os.path.isdir(self.saveFilePath):
                    os.mkdir(self.saveFilePath)

            self.bestPath = None
            self.paths = []

            self.bestScore = None
            self.bestScoreHistory = []
            self.scores = []

    def _saveProgress(self,generation):

        np.savetxt('{}/bestScoreHistory.csv'.format(self.saveFilePath),self.bestScoreHistory)

    def _getProbabilityMatrix(self):#TODO: make a uniform matrix for if exploration is encouraged

        pheremoneExpMat = np.power(self.pheremoneMatrix,self.pheremoneExponent)
        heuristicExpMat = np.power(self.heuristicMatrix,self.heuristicExponent)

        productMat = pheremoneExpMat * heuristicExpMat
        productSum = np.sum(np.sum(productMat))

        self.probabilityMatrix = productMat / productSum

    def _generateAnts(self):

        self.ants = []
        for _ in range(self.numberOfAnts):
            if self.universalStartingNode is not None:#add list/not list option here for collection of starting nodes
                self.ants.append(Ant(self.universalStartingNode,self.repeatNode))
            else:
                i = np.random.choice(list(range(len(self.graph))))
                j = np.random.choice(list(range(len(self.graph))))
                self.ants.append(Ant((i,j),self.repeatNode))

    def _moveAntsAlongPath(self):#TODO: add a probability option for exploration
        i = len(self.ants)
        for ant in self.ants:
            i-=1
            print(i)
            if self.explorationProbability is not None:
                if rnd.randint(0,100)/100 <= self.explorationProbability:
                    probMat = np.array((self.graph != 0),dtype='float')
                else:
                    probMat = self.probabilityMatrix
            else:
                probMat = self.probabilityMatrix
            for _ in range(self.movesPerGeneration):
                ant.move(probMat)
    
    def _getScores(self):

        for ant in self.ants:
            ant.score = self.function(ant.path)
            self.scores.append(ant.score)
            self.paths.append(ant.path)
        
        bestIdx = np.argmin(self.scores)

        if self.bestScore is None or (self.scores[bestIdx] < self.bestScore and self.bestScore is not None):
            self.bestScore = self.scores[bestIdx]
            self.bestPath = self.paths[bestIdx]
        
        self.bestScoreHistory.append(self.bestScore)
        
        self.scores = []
        self.paths = []

    def _updatePheremones(self):

        if self.Q == None:
            self.Q = self.bestScore

        self.pheremoneMatrix *= (1 - self.evaporationConstant)

        if self.onlyBestAntUpdate:
            idx = np.argmin([ant.score for ant in self.ants])
            path = self.ants[idx].path
            score = self.ants[idx].score
            for stop in path:
                self.pheremoneMatrix[stop[0]][stop[1]] += self.Q/score
        else:
            for ant in self.ants:
                for stop in ant.path:
                    self.pheremoneMatrix[stop[0]][stop[1]] += self.Q/ant.score
            
        self.Q = self.bestScore


    def optimize(self):
        
        for i in range(self.start+1,self.generations):
            tm = time.time()
            print('Generation {}'.format(i))
            self.()
            print('generate ants: {}'.format(time.time() - tm))
            self.()
            print('probability matrix: {}'.format(time.time() - tm))
            self.()
            print('move ants: {}'.format(time.time() - tm))
            self.()
            print('get scores: {}'.format(time.time() - tm))
            self.()
            print('update pheremones: {}'.format(time.time() - tm))
            if self.saveFilePath is not None:
                if i%10 == 0 or i == self.generations - 1:
                    print('Saving...')
                    thread = Thread(target=self._saveProgress, args=([i]))
                    thread.start()
                    thread.join()
                    print('Saved')
            print('save: {}'.format(time.time() - tm))

                

        

