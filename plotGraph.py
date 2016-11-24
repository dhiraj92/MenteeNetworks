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
    pass

def configPlot(configDict):
    #plot the validation and train errors
    pass
    
def plotGraphs():

    ## plot training errors vs epochs    
#    savedParams = pickle.load(open('Error_dicr.pkl'))
#    savedTrain = savedParams["train"]
#    savedTrainArray = np.asarray(savedTrain)
#    x = np.asarray(range(0,len(savedTrainArray)))
#    plt.xlabel('Epochs')
#    plt.ylabel('Training Error %')
#    plt.title('Training Errors vs Epochs')
#    plt.plot(x,savedTrainArray)
#    plt.savefig('TrainingError.png')
#    plt.show()
#    
#    ## plot validation error
#    savedValid = savedParams["valid"]
#    savedValidArray = np.asarray(savedValid)
#    x = np.asarray(range(0,len(savedValidArray)))
#    plt.xlabel('Epochs')
#    plt.ylabel('Validation Error %')
#    plt.title('Validation Errors vs Epochs')
#    plt.plot(x,savedValidArray)
#    plt.savefig('ValidationError.png')
#    plt.show()
#    
#    ## plot test error
#    savedTest = savedParams["test"]
#    savedTestArray = np.asarray(savedTest)
#    x = np.asarray(range(0,len(savedTestArray))) 
#    plt.xlabel('Epochs')
#    plt.ylabel('Test Error %')
#    plt.title('Test Errors vs Epochs')
#    plt.plot(x,savedTestArray)    
#    plt.savefig('TestError.png')
#    plt.show()
    
    
    ## code to plot all three in one graph
    savedParams = pickle.load(open('Error_dicr.pkl'))
    savedTrain = savedParams["train"]
    savedTrainArray = np.asarray(savedTrain)
    x = np.asarray(range(0,len(savedTrainArray)))
    plt.xlabel('Epochs')
    plt.ylabel('Error %')
    plt.title('Errors vs Epochs')
    plt.plot(x,savedTrainArray)
    
    ## plot validation error
    savedValid = savedParams["valid"]
    savedValidArray = np.asarray(savedValid)
    x = np.asarray(range(0,len(savedValidArray)))
    plt.plot(x,savedValidArray)
    
#    ## plot test error
#    savedTest = savedParams["test"]
#    savedTestArray = np.asarray(savedTest)
#    x = np.asarray(range(0,len(savedTestArray)))
#    plt.plot(x,savedTestArray)
#    plt.legend(['Training Error','Validation Error', 'Test Error'],loc='top right')
#    plt.savefig('Error.png')
#    plt.show()

if __name__ == '__main__':
    plotGraphs()