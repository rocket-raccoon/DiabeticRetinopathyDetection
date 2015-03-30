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
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from relu import relu
from dropout import dropout_neurons_from_layer
from convolutional_mlp import LeNetConvPooLayer

def load_batch(minibatches, index):
    datasets = load_data(settings.PROCESSED_TRAIN_DIR + "/" + dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]



def train_neural_network(learning_rate = 0.05,
                         n_epochs = 200,
                         data_directory = settings.PROCESSED_TRAIN_DIR,
                         batch_size = settings.MINI_BATCH_SIZE * settings.TRAIN_PERC,
                         nkerns = [20,50],
                         momentum = 0.9,
                         dropout_rates = [0.95, 0.75, 0.75, 0.75]):

    #Due to memory constraints, we have to load the data in batches off disk
    #So, let's get the file paths of all the minibatches in the data directory
    minibatches = [batch for batch in os.listdir(data_directory)]
    minibatches = [data_directory + "/" + batch for batch in minibatches]

    #Setup our symbolic variables for theano
    index = T.lscalar('index')
    x = T.matrix('x')
    y = T.ivector('y')
    use_dropout = T.scalaer('use_dropout')
    rng = numpy.random.RandomState(23455)

    #Setup layer0
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

    #Create all the updates
    momentum_updates = [
        (m_param_i, momentum * m_param_i - learning_rate * grad_i)
        for m_param_i, grad_i in zip(m_params, grads)
    ]
    regular_updates = [
        (param_i, param_i + momentum * m_param_i - learning_rate * grad_i)
        for param_i, m_param_i, grad_i in zip(params, m_params, grads)
    ]
    updates = regular_updates + momentum_updates

    #Create out training and testing functions
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index*train_batch_size: (index+1)*train_batch_size],
            y: train_set_y[index*train_batch_size: (index+1)*train_batch_size],
            use_dropout: 1.0
        }
    )

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index*test_batch_size: (index + 1)*test_batch_size],
            y: test_set_y[index*test_batch_size: (index + 1)*test_batch_size],
            use_dropout: 0.0
        }
    )

    train_losses = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: train_set_x[index*train_batch_size: (index + 1)*train_batch_size],
            y: train_set_y[index*train_batch_size: (index + 1)*train_batch_size],
            use_dropout: 0.0
        }
    )

    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    epoch = 0

    while (True):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index
            cost_ij = train_model(minibatch_index)
            test_losses = [test_model(i) for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)
            train_loss = [train_losses(i) for i in xrange(n_train_batches)]
            train_score = numpy.mean(train_loss)
            print "Mean Test Score: %f"%(test_score * 100)
            print "Mean Train Score: %f"%(train_score * 100)
        if epoch > n_epochs:
            break


if __name__ == "__main__":
    train_retinopathy_net()


