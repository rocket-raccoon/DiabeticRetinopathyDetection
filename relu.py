#Dependencies
from theano import tensor as T

#Rectified Linear Unit, should work better than sigmoid or tanh
def relu(x):
    return T.switch(x < 0, 0, x)


