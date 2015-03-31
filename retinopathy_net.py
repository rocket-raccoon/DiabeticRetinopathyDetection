#Dependencies
from relu import relu
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from dropout import dropout_neurons_from_layer
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class RetinopathyNet(object):

    def __init__(self,
                 rng,
                 x,
                 y,
                 use_dropout,
                 dropout_rates,
                 train_batch_size,
                 nkerns):

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

        #Relevant functions and attributes for this class
        self.cost = layer3.negative_log_likelihood(y)
        self.errors = layer3.errors(y)
        self.params = layer0.params + layer1.params + layer2.params + layer3.params

