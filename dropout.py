#Dependencies
import theano.tensor as T
import theano
from theano.ifelse import ifelse

#p is the probability of keeping a neuron active
#Dropout neurons from a layer according to this probability p
def dropout_neurons_from_layer(rng, layer, p, use_dropout):
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1,p=p, size=layer.shape) / p
    dropout_layer = layer * T.cast(mask, theano.config.floatX)
    output = ifelse(T.eq(use_dropout, 1), dropout_layer, layer)
    return output

