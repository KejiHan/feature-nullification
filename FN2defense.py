#when call main annoate train phase
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import random
import copy
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=True, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../tmp', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=1, shuffle=False, **kwargs)

from ori_main import Net, optimizer
#from mnist_fconvolution import optimizer
model=Net()
model=torch.load('/home/hankeji/Desktop/RFN/tra_con/mnist.pkl')
model.cuda()
optimizer = optimizer

mm=['RFN', 'FSFN', 'SWFN', 'Dropout']
for j in range(4):
    zero_rate = 0.5
    tmpmm = mm[j]
    for k in range(4):
        zero_rate = zero_rate + 0.1
        alpha = 0
        acc = []
        acc = np.asarray(acc, np.float16)
        for i in range(31):  # alpha in range(0,0.3)
            cadv = 0
            cori = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                correct = 0
                correct0 = 0
                print alpha

                if tmpmm == 'RFN':
                    c = round(np.random.normal(np.floor(784 * zero_rate), 1)).as_integer_ratio()  # RFN
                    a0 = np.arange(0, 784)
                    print c[0]
                    a = random.sample(a0, c[0])
                    x = np.ones((data.size()[0] * 784, 1), np.float32)
                    for i in range(len(a)):
                        b = a[i]
                        x[b] = 0
                    x = np.reshape(x, (1, 1, 28, 28))
                    x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
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
                    x = np.reshape(x, (1, 1, 28, 28))
                    x = Variable(torch.from_numpy(x), requires_grad=True).cuda()
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
                    x = np.reshape(x, (1, 1, 28, 28))
                    x = Variable(torch.from_numpy(x), requires_grad=True).cuda()

                else:#Dropout
                    x = np.ones((784, 1), np.float32)  # dropout feature nullification
                    for i in range(x.shape[0]):
                        x[i] = np.random.binomial(1, (1 - zero_rate), 1)
                    x = np.reshape(x, (1, 1, 28, 28))
                    x = Variable(torch.from_numpy(x), requires_grad=True).cuda()

                data = torch.FloatTensor(data)
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data, requires_grad=True), Variable(target)
                data1 = torch.mul(data, x)
                output = model(data1)
                loss = F.nll_loss(output, target)
                optimizer.zero_grad()
                loss.backward()
                # optimizer.step()#don't updata model
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).cpu().sum()
                cori += pred.eq(target.data).cpu().sum()
                print correct / data.size()[0]
                data0 = data + torch.mul(torch.sign(data.grad), alpha)  # generate adversarial samples

                '''
                attain FN samples
                '''
                '''
                if batch_idx==0:
                    #advs = Variable(torch.randn(1, 1, 28, 28))
                    advs1 = Variable(torch.randn(1, 1, 28, 28))
                    #advs=data1
                    advs1=data0
                elif batch_idx<3:
                    #advs=torch.cat((advs,data1),0)
                    advs1=torch.cat((advs1,data0),0)
                else:
                    #np.save('/home/hankeji/Desktop/MNIST_FN&ADV/SWFN/FN/'+'SWFN_0.5.npy',advs.cpu().data.numpy())
                    np.save('/home/hankeji/Desktop/MNIST_FN&ADV/ORI/ADV/' + str(alpha) + '_ORI.npy',advs1.cpu().data.numpy())
                    break
                '''

                '''
                tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Scale(100),
                ]) 
                from PIL.Image import Image
                if batch_idx%1==0:
                    a=data0
                    a1=data
                    a1=a1.view(3,32,32)
                    a=a.view(3,32,32)
                    a0=a.cpu().data
                    a1=a1.cpu().data
                    a0=tf(a0)
                    Image.show(a0)
                if batch_idx%1==0:
                    a=data0
                    a1=data
                    a1=a1.view(-1,28,28)
                    a=a.view(-1,28,28)
                    a0=a.cpu().data.numpy()
                    a1=a1.cpu().data.numpy()
                    #print a0.shape
                    cv2.imshow('kk',a0[0])
                    cv2.imshow('kk1', a1[0])
                    cv2.waitKey(100)
                '''
                output0 = model(data0)
                pred0 = output0.data.max(1)[1]
                correct0 += pred0.eq(target.data).cpu().sum()
                cadv += pred0.eq(target.data).cpu().sum()
                print correct0 / data.size()[0]
                print ("\n" * 2)
            # torch.save(model, '/home/hankeji/Desktop/RFN/adv_train.pkl')
            ori = cori / len(test_loader.dataset)
            adv = cadv / len(test_loader.dataset)
            acc = np.append(acc, alpha)
            acc = np.append(acc, ori)
            acc = np.append(acc, adv)
            alpha = alpha + 0.01
            print ("ori accracy is {:.4f} ").format(ori)
            print ("adv accracy is {:.4f} ").format(adv)
        acc = np.reshape(acc, (-1, 3))
        np.save('/home/hankeji/Desktop/mnist_' + tmpmm + '_' + str(zero_rate) + '.npy', acc)

