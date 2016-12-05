#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:01:17 2016

@author: tyagi
"""
import six.moves.cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
#import pdb



def errorPlot(errorDict):
    #plot the validation and train errors
    savedTrain = errorDict["train"]
    savedTrainArray = np.asarray(savedTrain)
    x = np.asarray(range(0,len(savedTrainArray)))
    plt.clf()
    plt.plot(x,savedTrainArray)
    savedValid = errorDict["valid"]
    savedValidArray = np.asarray(savedValid)
    x = np.asarray(range(0,len(savedValidArray)))
    
    plt.plot(x,savedValidArray)
#    savedTest = errorDict["test"]
#    savedTestArray = np.asarray(savedTest)
#    x = np.asarray(range(0,len(savedTestArray))) 
#    plt.xlabel('Epochs')
#    plt.ylabel('Error %')
#    plt.plot(x,savedTestArray)  
    plt.legend(['Train Error', 'Validation Error'])
    plt.title('Errors vs Epochs')
    plt.savefig('plots/Error.png')
    plt.show()


def configPlot(configDict):
    #plot the validation and train errors
    alphaList = configDict["alpha"]
    betaList = configDict["beta"]   
    gammaList = configDict["gamma"]
    alphaArray = np.asarray(alphaList)
    betaArray = np.asarray(betaList)
    gammaArray = np.asarray(gammaList)
    plt.clf()
    x = np.asarray(range(0,len(alphaArray)))
    plt.plot(x,alphaArray)
    plt.plot(x,betaArray)
    plt.plot(x,gammaArray)
    plt.xlabel('Epochs')
    plt.ylabel('Parameters')
    plt.legend(['Alpha','Beta','Gamma'])
    plt.title('Parameters')
    plt.savefig('plots/Parameters.png')
    plt.show()
    
def plotGraphs():
   pass

if __name__ == '__main__':
#    errorPlot()
    plotGraphs()