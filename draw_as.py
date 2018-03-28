import numpy as np
import cv2
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import copy
def bgr_to_rgb(img_bgr):
    # r,g,b=cv2.split(img_bgr)
    img_rgb = np.zeros(img_bgr.shape, img_bgr.dtype)
    img_rgb[:, :, 0] = img_bgr[:, :, 2]
    img_rgb[:, :, 1] = img_bgr[:, :, 1]
    img_rgb[:, :, 2] = img_bgr[:, :, 0]
    return img_rgb


#r,g,b=cv2.split(img_bgr)
'''
img_rgb = np.zeros(img_bgr.shape,img_bgr.dtype)
img_rgb[:,:,0] = img_bgr[:,:,2]
img_rgb[:,:,1] = img_bgr[:,:,1]
img_rgb[:,:,2] = img_bgr[:,:,0]
'''
#img_bgr1 = cv2.imread('/home/hankeji/Desktop/cat.jpg')
rootpt='/home/hankeji/Desktop/'
a=['RFN','RFN','FSFN']
b=['0.1','0.2']
'''
plt.subplot(121),plt.imshow(img_bgr)
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_rgb)
plt.xticks([]),plt.yticks([])
plt.show()

'''
'''
def str0(tmp):
    return str(tmp)
c=0
for i in range(3):
    for j in range(2):
        joint=rootpt+b[j]+a[i]+'_0.5.npy'
        tmp=str0(c)
        tmp=np.load(joint)
        tmp=np.reshape(tmp,(28,28))
        tmp0=Variable(torch.from_numpy(tmp))
        if c == 0:
            torchcat=tmp0
        else:
            torchcat=torch.cat((torchcat, tmp0),0)
        c+=1
'''
torchcat=np.load('/home/hankeji/Desktop/MNIST_FN&ADV/DROPOUT/ADV/0DROPOUT_0.5.npy')
torchcat1=np.load('/home/hankeji/Desktop/MNIST_FN&ADV/DROPOUT/ADV/0.1DROPOUT_0.5.npy')
torchcat2=np.load('/home/hankeji/Desktop/MNIST_FN&ADV/DROPOUT/ADV/0.3DROPOUT_0.5.npy')
#torchcat3=np.load('/home/hankeji/Desktop/MNIST_FN&ADV/SWFN/FN/SWFN_0.5.npy')
print torchcat1.shape
torchcat=np.reshape(torchcat,(10,28,28))
torchcat1=np.reshape(torchcat1,(10,28,28))
torchcat2=np.reshape(torchcat2,(10,28,28))
#torchcat3=np.reshape(torchcat2,(9,28,28))
plt.subplot(331),plt.imshow(torchcat[0,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(334),plt.imshow(torchcat[1,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(337),plt.imshow(torchcat[2,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(332),plt.imshow(torchcat1[0,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(335),plt.imshow(torchcat1[1,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(338),plt.imshow(torchcat1[2,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(333),plt.imshow(torchcat2[0,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(336),plt.imshow(torchcat2[1,:,:])
plt.xticks([]),plt.yticks([])
plt.subplot(339),plt.imshow(torchcat2[2,:,:])
plt.xticks([]),plt.yticks([])
plt.show()
