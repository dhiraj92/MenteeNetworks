# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 15:06:55 2016

@author: Dhiraj
"""

from __future__ import print_function
import thenoMLP

__docformat__ = 'restructedtext en'
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:29:43 2016

@author: Dhiraj
"""





import six.moves.cPickle as pickle
import os
import sys
import timeit
import pdb
import numpy

import theano
import theano.tensor as T


from logistic_sgd import LogisticRegression, load_data


# start-snippet-1
class TemperatureSoftmax(object):
    def __init__(self, temperature=0.1):
        self.temperature = temperature
    
    def softmax(self, x):
        if self.temperature != 1:
            e_x = T.exp(x / self.temperature)
            return (e_x ) / (e_x.sum(axis=-1).dimshuffle(0, 'x') )
        else:
            return theano.tensor.nnet.softmax(x)

class MentorNetwork(object):
        def __init__(self, input):        
            activation=T.tanh
    #        dataset='mnist.pkl.gz'
    #        datasets = load_data(dataset)
    #        test_set_x, test_set_y = datasets[2]
    #        test_set_x = test_set_x.get_value()
    #        test_set_y = test_set_y.eval()            
            self.Params = pickle.load(open('best_model_mlp_params.pkl'))[0]
            self.Wh = theano.shared(value = self.Params[0].eval(), name='Wh', borrow=True)
            self.bh = theano.shared(value = self.Params[1].eval(), name='bh', borrow=True)
            self.W = theano.shared(value = self.Params[2].eval(), name='W', borrow=True)
            self.b = theano.shared(value = self.Params[3].eval(), name='b', borrow=True)
            self.lin_output = T.dot(input, self.Wh) + self.bh
            self.output = (
                self.lin_output if activation is None
                else activation(self.lin_output)
                )
            
            temp = 1
            temperature_softmax = TemperatureSoftmax(temperature=0.1)
            self.p_y_given_x = temperature_softmax.softmax((T.dot(self.output, self.W) + self.b))
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        
    
    # symbolic description of how to compute prediction as class whose
    # probability is maximal
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

def menteeHiddenLoss(mentorLayer,menteeLayer):
    squareDiff = T.mean(T.pow(mentorLayer-menteeLayer,2))
    squareRootDiff = T.sqrt(squareDiff)
    return squareRootDiff
# start-snippet-2
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.MentorNetwork = MentorNetwork(            
            input=input      
        )


        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.hiddenError = (
                T.sqrt(T.mean(T.pow(self.MentorNetwork.output-self.hiddenLayer.output,2)))
        )
        self.softError = (
                T.sqrt(T.mean(T.pow(self.MentorNetwork.p_y_given_x-self.logRegressionLayer.p_y_given_x,2)))
        )

        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input



def test_mlp(learning_rate=0.01, L1_reg=0.001, L2_reg=0.0001, n_epochs=10,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    #mentorHiddenOut = T.matrix('mentorHiddenOut')
    #mentorSoftOut = T.matrix('mentorSoftOut')

    rng = numpy.random.RandomState(1234)
    #import pdb;pdb.set_trace()
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,        
        n_out=10       
        
    )
    

     

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    alpha = theano.shared(value = 1.0, name='alpha', borrow=True)
    beta = theano.shared(value = 1.0, name='beta', borrow=True)
    gamma = theano.shared(value = 1.0, name='gamma', borrow=True)
    
    
    cost = (
        classifier.negative_log_likelihood(y)
#        + L1_reg * classifier.L1
#        + L2_reg * classifier.L2_sqr
    )    
    
    costHidden = (
        classifier.hiddenError

    )
    
    costSoft = (
        classifier.softError
    )
    
    totalCost = (
        alpha*cost +
        beta*costHidden +
        gamma*costSoft
    )    
    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(totalCost, param) for param in classifier.params]
    #gparamsSoft = [T.grad(costSoft, param) for param in classifier.params]
    #gparamsHidden = [T.grad(costHidden, param) for param in classifier.params]    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * (gparam ))
        for param,gparam in zip(classifier.params, gparams)
    ]
    
    train_model = theano.function(
        inputs=[index],
        outputs=totalCost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]

        }
    )

    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`

    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    finalParams = [param for param in classifier.params]
    minibatch_avg_cost_avg = []
    errorDict = {}
    errorDict['train'] = list()
    errorDict['valid'] = list()
    errorDict['test'] = list()
    configDict = dict()
    configDict['alpha'] = []
    configDict['beta'] = []
    configDict['gamma'] = []
    while (epoch < n_epochs) and (not done_looping):
        print(alpha.get_value())
        incAplha = T.dscalar('a')
        incBeta = T.dscalar('b')
        incGamma = T.dscalar('g')
        hyperUpdatesAlph = theano.function([incAplha], alpha, updates=[(alpha, alpha+incAplha)])
        hyperUpdatesBeta = theano.function([incBeta], beta, updates=[(beta, beta+incBeta)])
        hyperUpdatesGamma = theano.function([incGamma], gamma, updates=[(gamma, gamma+incGamma)])
        if epoch < 5:        
            hyperUpdatesAlph(0.1)
            hyperUpdatesGamma(-0.01)        
            hyperUpdatesBeta(-0.02)
        else :
            hyperUpdatesAlph(-0.1)
            hyperUpdatesGamma(-0.01)        
            hyperUpdatesBeta(-0.02)
        
        #pdb.set_trace()
        configDict['alpha'].append(alpha.get_value().tolist())
        configDict['beta'].append(beta.get_value().tolist())
        configDict['gamma'].append(gamma.get_value().tolist())
        print(alpha.get_value(),beta.get_value(),gamma.get_value())
        
        epoch = epoch + 1
        #print (finalParams[0].eval())
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost_avg.append(train_model(minibatch_index))
            
            #print("Training error",minibatch_avg_cost)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print(iter)

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                errorDict['valid'].append(this_validation_loss*100)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                this_training_loss = numpy.mean(minibatch_avg_cost_avg)
                errorDict['train'].append(this_training_loss*100)

                print(
                    'epoch %i, minibatch %i/%i, training error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_training_loss * 100.
                    )
                )
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    errorDict['test'].append(test_score*100)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
                    
                    #pdb.set_trace()
                    
            if patience <= iter:
                done_looping = True
                break

    #pdb.set_trace()


    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
           
    #plot these two to understand how config and errorDict change 
    return errorDict,configDict
    
def predict(input):
    n_in=28 * 28
    n_hidden=500
    n_out=10
    
    #test_set_x = test_set_x.get_value()
    #test_set_y = test_set_y.eval()
    #test_set_x = input
    rng = numpy.random.RandomState(1234)
    #x = T.matrix('x')

    # Declare MLP classifier
    classifier = thenoMLP.MLP(
        rng=rng,
        input=input,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )
    
    # load the saved model
    classifier.params, classifier.logRegressionLayer.y_pred, classifier.logRegressionLayer.p_y_given_x,classifier.input,classifier.logRegressionLayer.input = pickle.load(open('best_model_mlp_params.pkl'))
    #pdb.set_trace()
    predict_model = theano.function(
        inputs=[classifier.hiddenLayer.input],
        outputs=classifier.hiddenLayer.output)
    #pdb.set_trace()

    hidden_act = predict_model(input)
    
    print("Hidden outpusts:", hidden_act)
    print( "hidden output shape", hidden_act.shape)
    #pdb.set_trace()
    logRegressionLayer = LogisticRegression(
            input=classifier.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

    predict_model_final = theano.function(
       inputs=[classifier.logRegressionLayer.input],
        outputs=classifier.logRegressionLayer.p_y_given_x)
    
    #print("Expected values: ", test_set_y)
    predicted_values = predict_model_final(hidden_act)
    print("Predicted values:", predicted_values)
    mentloss = logRegressionLayer.menteeLoss(hidden_act,hidden_act)
    print (mentloss)
    return hidden_act,predicted_values
    #pdb.set_trace()
import plotGraph
if __name__ == '__main__':
    errorDict,configDict = test_mlp(dataset="data/mnist.pkl.gz")
    plotGraph.errorPlot(errorDict)
    plotGraph.configPlot(configDict)
    #plot these two dicts 
    

