import numpy as np
from math import tanh, exp, e

ALPHA = 0.00001

ACTIVATIONS = {\
        'tanh':(tanh,lambda x:1.-tanh(x)**2),\
        'tanh+ax':(lambda x:tanh(x)+ALPHA*x,lambda x:1.-tanh(x)**2+ALPHA),\
        'logistic':(lambda x:1/(1+exp(-x)),lambda x:1-1/(1+exp(-x))**2)\
        }

_logistic = np.vectorize(ACTIVATIONS['logistic'][0])
def COMPRESS(w):
    w = _logistic(w)
    mn = w.min()
    mx = w.max()
    w -= mn
    w /= mx-mn
    return w

def init_weight(rows, cols, gaus=1., rand=1.):
    weights = np.mat(np.zeros((rows, cols)), dtype=np.float64)
    sigma = 1/(2*rows)
    for i in range(rows):
        for j in range(cols):
            weights[i,j] = exp(-((i*(cols-1)-j*(rows-1))/(cols*rows))**2/(2*sigma**2))
    for i in range(rows):
        weights[i,:] /= weights[i,:].sum()
    weights *= gaus
    weights += (np.random.rand(rows, cols)*2-1)*(rand/cols)
    return weights

class ANN:
    def __init__(self, layers, activation='tanh', learning_rate=1., gaus_scale=1., rand_scale=1.):
        self.__layers = tuple(layers)
        self.__dimensions = []
        l = iter(self.__layers)
        h = next(l)
        for n in l:
            self.__dimensions.append((n,h))
            h = n
        self.__dimensions = tuple(self.__dimensions)
        self.__weights = tuple((init_weight(rows,cols, gaus=gaus_scale, rand=rand_scale),\
                np.mat(np.zeros(rows),dtype=np.float64).T) \
                for rows,cols in self.__dimensions)
        self.__activation = ACTIVATIONS[activation]
        self.__derivative = np.vectorize(self.__activation[1])
        self.__activation = np.vectorize(self.__activation[0])
        self.__learning_rate = learning_rate
    def forward(self, val):
        val = np.mat(val).T
        for weight,bias in self.__weights:
            val = self.__activation(weight*val+bias)
        return np.array(val).flatten()
    def change_learning_rate(self, factor):
        self.__learning_rate *= factor
    def train(self, inp, out, training_weight=1.):
        inp = np.mat(inp).T
        out = np.mat(out).T
        deriv = []
        val = inp
        vals = [val]
        # forward calculation of activations and derivatives
        for weight,bias in self.__weights:
            val = weight*val
            val += bias
            deriv.append(self.__derivative(val))
            vals.append(self.__activation(val))
        deriv = iter(reversed(deriv))
        weights = iter(reversed(self.__weights))
        errs = []
        errs.append(np.multiply(vals[-1]-out, next(deriv)))
        # backwards propagation of errors
        for (w,b),d in zip(weights, deriv):
            errs.append(np.multiply(np.dot(w.T, errs[-1]), d))
        weights = iter(self.__weights)
        for (w,b),v,e in zip(\
                self.__weights,\
                vals, reversed(errs)):
            e *= self.__learning_rate*training_weight
            w -= e*v.T
            b -= e
        tmp = vals[-1]-out
        return np.dot(tmp[0].T,tmp[0])*.5*training_weight
    def get_layer_image(self, layer):
        return COMPRESS(self.__weights[layer][0])
    def get_layer_count(self):
        return len(self.__weights)

if __name__ == '__main__':
    from math import sin,cos
    from random import shuffle
    n = ANN((2,7,2), learning_rate=1., scale=0.00001)
    f=lambda i,j:(cos(i/20),sin(j/20)*cos(i/20))
    data = [((u,v),f(u,v)) for u,v in ((i/100,j/100) for i in range(100) for j in range(100))]
    def ssqd():
        print("avg SSqD: %g"%(sum(sum((a-b)**2 for a,b in zip(n.forward(x),y)) for x,y in data)/len(data)))
    ssqd()
    for i in range(10):
        print("Training pass %d"%i)
        shuffle(data)
        print(sum(n.train(inp,out) for inp,out in data))
        ssqd()

