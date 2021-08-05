import numpy as np

import numba
from numba import jit



@jit(nopython=True, parallel=True, )
def func():
    pass



@jit(nopython=True, nogil=True)
def func():
    pass
