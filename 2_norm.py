from __future__ import division
import numpy as np
import random
A=np.ones((4,5),np.float)

mm=['RFN','FSFN', 'SWFN', 'Dropout']
for M in range(4):
    tmpmm=mm[M]
    zero_rate=0
    for N in range(5):
        zero_rate+=0.1
        TMP0 = 0
        for L in range(10000):
            if tmpmm == 'RFN':
                c = round(np.random.normal(np.floor(784 * zero_rate), 1)).as_integer_ratio()  # RFN
                a0 = np.arange(0, 784)
                #print c[0]
                a = random.sample(a0, c[0])
                x = np.ones((784, 1), np.float32)
                for i in range(len(a)):
                    b = a[i]
                    x[b] = 0
                x = np.reshape(x, (28, 28))


            elif tmpmm == 'FSFN':
                if zero_rate > 0.5:
                    x = np.zeros((784, 1), np.float32)
                    l = (len(str(zero_rate)) - 2)
                    lw = np.power(10, l)
                    space = np.floor(1 / (1 - zero_rate)).astype(np.int)
                    for i in range(int(np.floor(784 / lw))):
                        a = np.random.random_integers(i * lw, (i + 1) * lw - 1, 1)
                        forward = a
                        backward = a
                        tmpcount = 0
                        fn = round(lw * (1 - zero_rate))
                        while forward > i * lw:
                            x[forward] = 1
                            tmpcount += 1
                            forward = forward - space
                        while backward <= (i + 1) * lw:
                            x[backward] = 1
                            tmpcount += 1
                            if tmpcount == fn:
                                break
                            backward = backward + space
                    j = (i + 1) * lw
                    while j < 784:
                        x[j] = 1
                        j = j + space
                else:
                    x = np.ones((784, 1), np.float32)
                    l = (len(str(zero_rate)) - 2)
                    lw = np.power(10, l)
                    fn = int(round(lw * zero_rate))
                    space = np.floor(1 / zero_rate).astype(np.int)
                    for i in range(int(np.floor(784 / lw))):
                        a = np.random.random_integers(i * lw, (i + 1) * lw, 1)
                        forward = a
                        backward = a
                        tmpcount = 0
                        while forward >= i * lw:
                            x[forward] = 0
                            tmpcount += 1
                            if tmpcount == fn:
                                break
                            forward = forward - space
                        while backward <= (i + 1) * lw:
                            x[backward] = 0
                            tmpcount += 1
                            if tmpcount == fn:
                                break
                            backward = backward + space
                    j = (i + 1) * lw
                    while j < 784:
                        x[j] = 0
                        j = j + space
                x = np.reshape(x, ( 28, 28))

            elif tmpmm == 'SWFN':
                if zero_rate > 0.5:
                    x = np.zeros((784, 1), np.float32)
                    l = (len(str(zero_rate)) - 2)
                    lw = np.power(10, l)
                    space = np.floor(1 / (1 - zero_rate)).astype(np.int)
                    for i in range(int(np.floor(784 / lw))):
                        nf = int((1 - zero_rate) * lw)
                        a0 = np.arange(i * lw, (i + 1) * lw - 1)
                        a = random.sample(a0, nf)
                        for k in range(nf):
                            x[a[k]] = 1
                    j = (i + 1) * lw
                    while j < 784:
                        x[j] = 1
                        j = j + space
                else:
                    x = np.ones((784, 1), np.float32)
                    l = (len(str(zero_rate)) - 2)
                    lw = np.power(10, l)
                    space = np.floor(1 / (zero_rate)).astype(np.int)
                    for i in range(int(np.floor(784 / lw))):
                        nf = int(zero_rate * lw)
                        a0 = np.arange(i * lw, (i + 1) * lw - 1)
                        a = random.sample(a0, nf)
                        for k in range(nf):
                            x[a[k]] = 0
                    j = (i + 1) * lw
                    while j < 784:
                        x[j] = 0
                        j = j + space
                x = np.reshape(x, (28, 28))


            else:
                x = np.ones((784, 1), np.float32)  # dropout feature nullification
                for i in range(x.shape[0]):
                    x[i] = np.random.binomial(1, (1 - zero_rate), 1)
                x = np.reshape(x, (28, 28))
            TMP0 = TMP0+np.linalg.norm(x, ord=2)
            print TMP0
        print('\n'*2)
        A[M][N]=TMP0/10000



print A
