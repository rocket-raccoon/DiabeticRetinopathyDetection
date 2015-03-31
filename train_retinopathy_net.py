################################################################################
#  A big problem we have is that the data is too big to be loaded into shared  #
#  memory.  So this script is kind of a work around to that.  Instead, what    #
#  we do here is divide the data into "big" batches.  These batches are        #
#  iteratively loaded into shared memory.  We can then do SGD using            #
#  minibatches on these individual big batches.                                #
################################################################################

#Dependencies
import os
import time
import numpy as np
import theano
import theano.tensor as T
import settings
import cPickle

from retinopathy_net import RetinopathyNet

#Saves the parameters of our architecture to disk
def save_params(params, output_dir, file_id):
    param_dict = {}
    counter = 0
    for param in params:
        param_dict["%i"%counter] = param.get_value()
        counter += 1
    output_file = output_dir + "/training_params_%i"%file_id
    np.savez(output_file, **param_dict)

#This function loads our dataset into a shared variable for theano
#Assumes there is (x, y) matrix pair pickled
def load_batch(dataset):

    # Load the dataset
    f = open(dataset, 'rb')
    x_set, y_set = cPickle.load(f)
    f.close()

    #Create shared variables out of it
    ppi = reduce(lambda x,y: x*y, x_set[0].shape)
    x_set = x_set.reshape(x_set.shape[0], ppi)
    shared_x = theano.shared(np.asarray(x_set, dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(np.asarray(y_set, dtype=theano.config.floatX),
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
            x: train_set_x[index*batch_size: (index+1)*batch_size],
            y: train_set_y[index*batch_size: (index+1)*batch_size],
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
def train_retinopathy_net(learning_rate = 0.01,
                          n_epochs = 200,
                          data_directory = settings.PROCESSED_TRAIN_DIR,
                          train_batch_size = 10,
                          test_batch_size = 10,
                          nkerns = [20,50],
                          momentum = 0.9,
                          dropout_rates = [0.95, 0.75, 0.75, 0.75],
                          param_save_freq = 1000):

    #First, we load in the test batch.  This will always be in memory
    #There should only be one in there
    print "Loading test batch into shared memory..."
    test_batch = [bb for bb in os.listdir(data_directory) if "test" in bb][0]
    test_set_x, test_set_y = load_batch(data_directory + "/" + test_batch)

    #Now, we get the paths to each of the big train batches
    #We will load these into memory on an as need basis unlike the test batch
    big_batches = [bb for bb in os.listdir(data_directory) if "train" in bb]
    big_batches = [data_directory + "/" + batch for batch in big_batches]

    #Setup our symbolic variables for theano
    index = T.lscalar('index')
    x = T.matrix('x')
    y = T.ivector('y')
    use_dropout = T.scalar('use_dropout')
    rng = np.random.RandomState(23455)

    #Instantiate our retinopathy net
    print "Building retinopathy network..."
    retinopathy_net = RetinopathyNet(rng, x, y, use_dropout,
                                     dropout_rates, train_batch_size, nkerns)
    cost = retinopathy_net.cost
    errors = retinopathy_net.errors
    params = retinopathy_net.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # create a list of all momentum parameters, initialized to zero
    m_params = []
    for param in params:
        param_shape = param.get_value().shape
        data_type = param.dtype
        m_param = theano.shared(np.zeros(param_shape, dtype=data_type), borrow=True)
        m_params.append(m_param)

    #Create our parameter updates
    updates = get_updates(m_params, params, grads, learning_rate, momentum)

    print "Training neural network..."

    #big_batches = [big_batches[0]]
    n_train_batches = test_set_x.get_value().shape[0] / train_batch_size
    n_test_batches = test_set_x.get_value().shape[0] / test_batch_size
    itr = 0
    epoch = 0
    save_counter = 0

    while(True):

        for big_batch in big_batches:

            #Load in a big batch as a shared variable
            train_set_x, train_set_y = load_batch(big_batch)

            #Setup our training and testing functions in theano
            train_func = get_train_func(index, cost, updates, x, y, use_dropout,
                                        train_set_x, train_set_y, train_batch_size)
            test_err_func = get_err_func(index, errors, x, y, use_dropout,
                                         test_set_x, test_set_y, test_batch_size)
            train_err_func = get_err_func(index, errors, x, y, use_dropout,
                                          train_set_x, train_set_y, train_batch_size)

            #Run SGD over the big batch
            for minibatch_index in xrange(n_train_batches):

                #Take one training step
                minibatch_avg_cost = train_func(minibatch_index)
                itr += 1

                #Compute the test error
                test_losses = [test_err_func(i) for i in xrange(n_test_batches)]
                test_score = np.mean(test_losses)
                print "Mean Test Error: %f"%(test_score * 100)

                #Compute the training error
                train_loss = [train_err_func(i) for i in xrange(n_train_batches)]
                train_score = np.mean(train_loss)
                print "Mean Train Error: %f"%(train_score*100)

                #Save params periodically
                if itr % param_save_freq == 0:
                    save_params(params, settings.PARAMS_DIR, save_counter)
                    save_counter += 1

        #Stop training if we exceed the max number of epochs allowed
        epoch += 1
        if epoch > n_epochs:
            break

if __name__ == "__main__":
    train_retinopathy_net()


