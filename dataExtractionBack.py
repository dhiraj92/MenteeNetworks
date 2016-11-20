# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:19:43 2016

@author: Dhiraj
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 13:46:39 2016

@author: Dhiraj
"""
import numpy

import gzip, cPickle


my_x = []
my_y = []
with open('data/mnist_all_rotation_normalized_float_train_valid.amat', 'r') as f:
    for line in f:
        line = line.strip('\n')
        my_list = line.split('   ') # replace with your own separator instead
        my_x.append(my_list[1:-1]) # omitting identifier in [0] and target in [-1]
        my_y.append(my_list[-1])
train_set_x = numpy.array(my_x[:10000], dtype='float64')
train_set_y = numpy.array(my_y[:10000], dtype='float64')
val_set_x = numpy.array(my_x[10000:], dtype='float64')
val_set_y = numpy.array(my_y[10000:], dtype='float64')
my_x = []
my_y = []
with open('data/mnist_all_rotation_normalized_float_test.amat', 'r') as f:
    for line in f:
        line = line.strip('\n')
        my_list = line.split('   ') # replace with your own separator instead
        my_x.append(my_list[1:-1]) # omitting identifier in [0] and target in [-1]
        my_y.append(my_list[-1])
test_set_x = numpy.array(my_x, dtype='float64')
test_set_y = numpy.array(my_y, dtype='float64')

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open('data/rotateImg.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()
