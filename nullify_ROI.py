from __future__ import division
import numpy as np
from pylab import *
import random
def nullify_ROI(nr):#nr donates feature nullification rate
    x = np.ones((784, 1), np.float32)
    fn = np.ceil(28 * 28 * nr / (18)).astype(np.int)
    print fn
    for i in range(18):#ROI is a 18*18 region of center of sample
        t1 = np.arange(140 + i * 28, 157 + i * 28)
        a = random.sample(t1, fn)
        for j in range(fn):
            x[a[j]] = 0

    return x