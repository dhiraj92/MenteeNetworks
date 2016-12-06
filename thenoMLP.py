from __future__ import print_function

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
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

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


# start-snippet-2
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):


        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
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
        



def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=200,
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

    rng = numpy.random.RandomState(1234)
    #import pdb;pdb.set_trace()
    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10,
        
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
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

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
  
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
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
    while (epoch < n_epochs) and (not done_looping):
        
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
                    with open('best_model_mlp_params.pkl', 'wb') as f:
                        pickle.dump((classifier.params, classifier.logRegressionLayer.y_pred, classifier.logRegressionLayer.p_y_given_x,
                 classifier.input,classifier.logRegressionLayer.input), f)

            if patience <= iter:
                done_looping = True
                break
    #pdb.set_trace()
    with open('Error_dicr.pkl', 'wb') as f:
        pickle.dump(errorDict, f)           
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return errorDict
    
def predict(dataset, n_hidden, n_in, n_out):
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')

    # Declare MLP classifier
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )
    
    # load the saved model
    classifier.params, classifier.logRegressionLayer.y_pred, classifier.logRegressionLayer.p_y_given_x,classifier.input,classifier.logRegressionLayer.input = pickle.load(open('best_model_mlp_params.pkl'))
    #pdb.set_trace()
    classifier.params = [theano.shared(param.eval()) for param in classifier.params]
    predict_model = theano.function(
        inputs=[classifier.hiddenLayer.input],
        outputs=classifier.hiddenLayer.output)
    hidden_act = predict_model(test_set_x[:200])
    print("Hidden outputs:", hidden_act)
    print( "hidden output shape", hidden_act.shape)
    #pdb.set_trace()
    classifier.logRegressionLayer.W = theano.shared(classifier.params[2].eval())
    classifier.logRegressionLayer.b = theano.shared(classifier.params[3].eval())
    classifier.logRegressionLayer.params = [theano.shared(param.eval()) for param in classifier.params[2:4]]
    predict_model_final = theano.function(
       inputs=[classifier.hiddenLayer.output],
        outputs=classifier.logRegressionLayer.y_pred)
    
    print("Expected values: ", test_set_y[:200])
    predicted_values = predict_model_final(hidden_act)
    print("Predicted values:", predicted_values)

    #pdb.set_trace()

def predictW():
    activation=T.tanh
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    test_set_y = test_set_y.eval()
    Params = pickle.load(open('best_model_mlp_params.pkl'))[0]
    Wh = theano.shared(value = Params[0].eval(), name='Wh', borrow=True)
    bh = theano.shared(value = Params[1].eval(), name='bh', borrow=True)
    W = theano.shared(value = Params[2].eval(), name='W', borrow=True)
    b = theano.shared(value = Params[3].eval(), name='b', borrow=True)
    lin_output = T.dot(test_set_x[:10], Wh) + bh
    output = (
        lin_output if activation is None
        else activation(lin_output)
        )
    
    temp = 1
    p_y_given_x = T.nnet.softmax((T.dot(output, W) + b)/temp)
    
    # symbolic description of how to compute prediction as class whose
    # probability is maximal
    y_pred = T.argmax(p_y_given_x, axis=1)
    print ("predicted",y_pred.eval())
    print ("excepted",test_set_y[:10])
    pdb.set_trace()
    # end-snippet-1


if __name__ == '__main__':
#    predictW() 
#   data folder has other datasets too
    errorDict = test_mlp(dataset="data/mnist.pkl.gz")
    #params = test_mlp()
    n_in=28 * 28
    n_hidden=500
    n_out=10
    dataset='mnist.pkl.gz'
    
    #predictW()
    #predict(dataset,n_hidden,n_in,n_out)
    #params = test_mlp()
