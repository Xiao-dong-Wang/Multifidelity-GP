import autograd.numpy as np
import os
import string

def init_dataset(funct, num, bounds):
    dim = bounds.shape[0]
    num_low = num[0]
    num_high = num[1]
    low_x = np.random.uniform(-0.5, 0.5, (dim, num_low))
    high_x = np.random.uniform(-0.5, 0.5, (dim, num_high))

    dataset = {}
    dataset['low_x']    = low_x
    dataset['high_x']    = high_x
    dataset['low_y']    = funct[0](low_x, bounds)
    dataset['high_y']    = funct[1](high_x, bounds)
    return dataset

# bounds:   0 : 1
def test1_high(x, bounds):
    tmp = test1_low(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (x-np.sqrt(2))*tmp**2
    return ret.reshape(1, -1)

def test1_low(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = np.sin(8.0*np.pi*x)
    return ret.reshape(1, -1)

# bounds:   0 : 1
def test2_high(x, bounds):
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = (6.0*x - 2.0)**2 * np.sin(12.*x - 4.0)
    return ret.reshape(1, -1)

def test2_low(x, bounds):
    tmp = test2_high(x, bounds)
    mean = bounds.mean(axis=1)
    delta = bounds[:,1] - bounds[:,0]
    x = (x.T * delta + mean).T
    ret = 0.5*tmp + 10.0*(x-0.5) - 5.0
    return ret.reshape(1, -1)

def get_funct(funct):
    if funct == 'test1':
        return [test1_low, test1_high]
    elif funct == 'test2':
        return [test2_low, test2_high]
    else:
        return [test1_low, test1_high]

