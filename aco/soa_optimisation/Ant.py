import numpy as np
import time
class Ant:

    def __init__(self,start,repeatNode=False):

        self.score = 0
        self.repeatNode = repeatNode
        self.visitedNodes = [start[1]]
        self.setPosition(start)
        # print('position at init: {}'.format(self.currentPosition))
        self.clearPath()

    def move(self,probabilityMatrix):
        # probabilityMatrix = probabilityMatrix.copy()
        probabilityVector = probabilityMatrix[self.currentPosition[1]][:].copy()
        # print('position before move: {}'.format(self.currentPosition))
        # print('prob vector before change: {}'.format(probabilityVector))
        if not self.repeatNode:
            probabilityVector[self.visitedNodes] = 0.0
        # print('prob vector after change: {}'.format(probabilityVector))
        probabilityVector /= sum(probabilityVector)

        newPosition = (self.currentPosition[1],np.random.choice(list(range(len(probabilityVector))), \
                                        p=probabilityVector))
        # print('position after move: {}\n\n'.format(newPosition))
        self.updatePath(newPosition)
        self.visitedNodes.append(newPosition[1])
    
    def clearPath(self):

        self.path = []

    def setPosition(self,position):

        self.currentPosition = position

    def updatePath(self,newPosition):

        self.path.append(newPosition)
        self.currentPosition = newPosition