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
   
    plt.plot(x,savedTrainArray)
    savedValid = errorDict["valid"]
    savedValidArray = np.asarray(savedValid)
    x = np.asarray(range(0,len(savedValidArray)))
    
    plt.plot(x,savedValidArray)
    savedTest = errorDict["test"]
    savedTestArray = np.asarray(savedTest)
    x = np.asarray(range(0,len(savedTestArray))) 
    plt.xlabel('Epochs')
    plt.ylabel('Error %')
    plt.plot(x,savedTestArray)  
    plt.legend(['Train Error', 'Validation Error', 'Test Error'])
    plt.title('Errors vs Epochs')
    plt.savefig('Error.png')
    plt.show()


def configPlot(configDict):
    #plot the validation and train errors
    alphaList = configDict["alpha"]
    betaList = configDict["beta"]   
    gammaList = configDict["gamma"]
    alphaArray = np.asarray(alphaList)
    betaArray = np.asarray(betaList)
    gammaArray = np.asarray(gammaList)
    
    x = np.asarray(range(0,len(alphaArray)))
    plt.plot(x,alphaArray)
    plt.plot(x,betaArray)
    plt.plot(x,gammaArray)
    plt.xlabel('Epochs')
    plt.ylabel('Parameters')
    plt.legend(['Alpha','Beta','Gamma'])
    plt.title('Parameters')
    plt.savefig('Parameters.png')
    plt.show()
    
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
#    errorPlot()
    plotGraphs()