#-*-coding:utf-8-*-
import numpy as np
from pylab import *
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
acc0=np.load('/home/hankeji/Desktop/mnist_PROT_REGION.npy')
acc1=np.load('/home/hankeji/Desktop/mnist_without_PROT_REGION.npy')
acc4=np.load('/home/hankeji/Desktop/RFN/mnist_ORI.npy')

plot01,=plt.plot(acc4[:,0],acc4[:,2],'c:',label='ori')
#plot1,=plt.plot(acc1[:,0],acc1[:,1],'m',label='WITHOUT_ori')
plot2,=plt.plot(acc1[:,0],acc1[:,2],'m.',label='WITHOUT')
#plot7,=plt.plot(acc0[:,0],acc0[:,1],'b',label='WITH_ori')
plot8,=plt.plot(acc0[:,0],acc0[:,2],'b.',label='WITH')
#plt.title('Defense Performance of FNs wiht protecting semantic region and without')
plt.legend(handles=[plot01, plot2, plot8])
plt.xlabel('phi(scale of gradient added to sample)')
plt.ylabel('accuracy of model')
ax=plt.axes()
plt.show()

