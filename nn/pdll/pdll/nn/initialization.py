
from pdll.backend.executor import engine


def uniform(low, high, size):
    '''uniform
    '''
    return engine.np.random.uniform(low=low, high=high, size=size)


def ones(shape):
    '''ones
    '''
    return engine.np.ones(shape=shape)


def zeros(shape):
    '''ones
    '''
    return engine.np.zeros(shape=shape)

    


def uniform_(tensor, low, high):
    '''uniform
    '''
    tensor.storage[...] = uniform(low, high, tensor.shape)


