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

def plotGraphs():
    ## plot training errors vs epochs
    savedParams = pickle.load(open('Error_dicr.pkl'))
    savedTrain = savedParams["train"]
    savedTrainArray = np.asarray(savedTrain)
    x = np.asarray(range(0,len(savedTrainArray)))
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.title('Errors vs Epochs')
    plt.plot(x,savedTrainArray)
    
    ## plot validation error
    savedValid = savedParams["valid"]
    savedValidArray = np.asarray(savedValid)
    plt.plot(x,savedValidArray)

    ## plot test error
    savedTest = savedParams["test"]
    savedTestArray = np.asarray(savedTest)
    plt.plot(x,savedTestArray)
        
    plt.legend(['Training Error','Validation Error','Test Error'],loc='upper right')
    plt.savefig('Error.png')
    plt.show()
    
    ## plot validation errors vs epochs
#    savedValid = savedParams["valid"]
#    savedValidArray = np.asarray(savedValid)
#    x = np.asarray(range(0,len(savedValidArray)))
#    plt.xlabel('Epochs')
#    plt.ylabel('Training Error')
#    plt.title('Training error vs Epochs')
#    plt.plot(x,savedValidArray)
#    plt.savefig('validationError.png')
#    plt.show()
    
    
    

if __name__ == '__main__':
    plotGraphs()