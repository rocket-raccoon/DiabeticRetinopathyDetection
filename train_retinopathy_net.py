################################################################################
#  A big problem we have is that the data is too big to be loaded into shared  #
#  memory.  So this script is kind of a work around to that.  Instead, what    #
#  we do here is divide the data into "big" batches.  These batches are        #
#  iteratively loaded into shared memory.  We can then do SGD using            #
#  minibatches on these individual big batches.                                #
################################################################################

#Dependencies
import os
import sys
import time
import numpy
import numpy as np
import theano
import theano.tensor as T
import settings

from theano.ifelse import ifelse
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from relu import relu
from dropout import dropout_neurons_from_layer
from convolutional_mlp import LeNetConvPooLayer

#This function loads our dataset into a shared variable for theano
#Assumes there is (x, y) matrix pair pickled
def load_batch(dataset):

    # Load the dataset
    f = open(dataset, 'rb')
    x_set, y_set = cPickle.load(f)
    f.close()

    #Create shared variables out of it
    x_set = x_set.reshape(x_set.shape[0], settings.PIXELS_PER_IMAGE)
    shared_x = theano.shared(numpy.asarray(x_set, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(y_set, dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')

#This function is responsible for training our neural network
#Every SGD step will use this to update our parameters
def get_train_func(index, cost, updates, x, y, use_dropout,
                   train_set_x, train_set_y, batch_size):
    train_func = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index*train_batch_size: (index+1)*train_batch_size],
            y: train_set_y[index*train_batch_size: (index+1)*train_batch_size],
            use_dropout: 1.0
        }
    )
    return train_func

#This function will just compute the error rate on same set of data
#We will use this to see training and test error so we can decide things like
#when to stop training, which parameters to use, etc.
def get_err_func(index, error_func, x, y, use_dropout,
                 datax, datay, batch_size):
    test_model = theano.function(
        [index],
        error_func,
        givens={
            x: datax[index*batch_size: (index + 1)*batch_size],
            y: datay[index*batch_size: (index + 1)*batch_size],
            use_dropout: 0.0
        }
    )
    return test_model

#This is a handy helper function that will return a list of all the updates
#that need to be passed to the training function
def get_updates(m_params, params, grads, learning_rate, momentum):
    momentum_updates = [
        (m_param_i, momentum * m_param_i - learning_rate * grad_i)
        for m_param_i, grad_i in zip(m_params, grads)
    ]
    regular_updates = [
        (param_i, param_i + momentum * m_param_i - learning_rate * grad_i)
        for param_i, m_param_i, grad_i in zip(params, m_params, grads)
    ]
    updates = regular_updates + momentum_updates
    return updates

#This is the meat of our application
#Will continuously train our neural network architecture until we see fit
#to stop it.
def train_neural_network(learning_rate = 0.05,
                         n_epochs = 200,
                         data_directory = settings.PROCESSED_TRAIN_DIR,
                         train_batch_size = 10,
                         test_batch_size = 10,
                         nkerns = [20,50],
                         momentum = 0.9,
                         dropout_rates = [0.95, 0.75, 0.75, 0.75]):

    #First, we load in the test batch.  This will always be in memory
    #There should only be one in there
    test_batch = [bb for bb in os.listdir(data_directory) of "test" in bb][0]
    test_set_x, test_set_y = load_batch(data_directory + "/" + test_batch)

    #Now, we get the paths to each of the big train batches
    #We will load these into memory on as as need basis unlike the test batch
    big_batches = [bb for bb in os.listdir(data_directory) if "train" in bb]
    big_batches = [data_directory + "/" + batch for batch in minibatches]

    #Setup our symbolic variables for theano
    index = T.lscalar('index')
    x = T.matrix('x')
    y = T.ivector('y')
    use_dropout = T.scalar('use_dropout')

    #Setup layer0
    rng = numpy.random.RandomState(23455)
    layer0_input = dropout_neurons_from_layer(rng, x, dropout_rates[0], use_dropout)
    layer0_input = layer0_input.reshape((train_batch_size, 1, 512, 512))
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(train_batch_size, 1, 512, 512),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )
    layer0.output = dropout_neurons_from_layer(rng, layer0.output, dropout_rates[1], use_dropout)

    #Setup layer 1
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(train_batch_size, nkerns[0], 254, 254),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )
    layer1.output = dropout_neurons_from_layer(rng, layer1.output, dropout_rates[2], use_dropout)

    #Setup layer 2
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 125 * 125,
        n_out=500,
        activation=relu
    )
    layer2.output = dropout_neurons_from_layer(rng, layer2.output, dropout_rates[3], use_dropout)

    #Setup layer 3
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=5)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # create a list of all momentum parameters, initialized to zero
    m_params = []
    for param in params:
        param_shape = param.get_value().shape
        data_type = param.dtype
        m_param = theano.shared(np.zeros(param_shape, dtype=data_type), borrow=True)
        m_params.append(m_param)

    updates = get_updates(m_params, params, grads, learning_rate, momentum)

    while(True):

        for big_batch in big_batches:

            #Load in a big batch as a shared variable
            train_set_x, train_set_y = load_batch(big_batch)

            #Setup our training and testing functions in theano
            train_func = get_train_func(index, cost, updates, x, y,
                                        train_set_x, train_set_y, use_dropout)
            test_err_func = get_err_func(index, layer3.errors(y), x, y, use_dropout,
                                         test_set_x, test_set_y, test_batch_size)
            train_err_func = get_err_func(index, layer3.errors(y), x, y, use_dropout,
                                          train_set_x, train_set_y, train_batch_size)

            #Run SGD over the big batch
            for minibatch_index in xrange(n_train_batches):

                #Take one training step
                minibatch_avg_cost = train_func(minibatch_index)
                itr += 1

                #Compute the test error
                test_losses = [test_err_func(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print "Mean Test Error: %f"%(test_score * 100)

                #Compute the training error
                train_loss = [train_err_func(i) for i in xrange(n_train_batches)]
                train_score = numpy.mean(train_loss)
                print "Mean Train Error: %f"%(train_score*100)

        epoch += 1
        if epoch > n_epochs:
            break

if __name__ == "__main__":
    train_retinopathy_net()


