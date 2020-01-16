# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:25:43 2020

@author: Hector
"""

import attacks
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
from attacks import PGD, FGSM, ADV_interp
import VGG
from Net import Net
import sys
from tqdm import tqdm
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


EPOCH = 25
BATCH_SIZE = 32
DEPTH = 22
WIDEN_FACTOR = 10
DROPOUT = 0.3
CLASSES = 10
LEARNING_RATE = 1e-3
LOG_INTERVAL = 1
NB_MODELS = 1

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)
    
def target_transform(target):
    target_tensor = torch.zeros(10)
    target_tensor[target] = 1
    return target_tensor

transform = transforms.Compose([transforms.ToTensor()])
target_transform = target_transform

test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform, target_transform=target_transform)

samples = list(np.arange(0, 100, 10))

testset = torch.utils.data.Subset(test, samples)

test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

def Cross_Entropy(predicted, target):
     return -(target * predicted).sum(dim=1).mean()

criterion = Cross_Entropy

def test_model(model, test_loader, criterion, attacker=None, device="cpu"):

    model = model.to(device)
    model.eval()

    testing_accuracy = 0
    testing_CE = 0

    for iteration, (batch_x, batch_y) in enumerate(test_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
            
        if attacker is not None:
            batch_x, batch_y = attacker.attack(model, batch_x, batch_y)
        
        with torch.no_grad():
            batch_y_hat = model(batch_x, mode="prediction")
            loss = criterion(batch_y_hat, batch_y)
         
        current_accuracy = torch.eq(torch.max(batch_y_hat, 1)[1], torch.max(batch_y, 1)[1]).sum().item() / batch_y.shape[0]
        current_CE = loss.detach().cpu().item()

        testing_CE += current_CE
        testing_accuracy += current_accuracy

    #print("Testing Results - Avg accuracy: {:.2f} Avg loss: {:.2f}".format(testing_accuracy / len(test_loader), testing_CE / len(test_loader)))
    
    return testing_accuracy / len(test_loader), testing_CE / len(test_loader)

model_adv = Net()
model_PGD = Net()
model_natural = Net()
model_adv.load_state_dict(torch.load('models/adversarial_interpolation_0_epoch_0'))
model_PGD.load_state_dict(torch.load('models/PGD_0_epoch_0'))
model_natural.load_state_dict(torch.load('models/natural_0_epoch_0'))

data_natural = []
data_adv = []
data_PGD = []

for i in range(100):
    pgd = PGD(Cross_Entropy, iter=i)
    data_natural.append(test_model(model_natural, test_loader, criterion, attacker=pgd, device=device)[1])
    data_PGD.append(test_model(model_PGD, test_loader, criterion, attacker=pgd, device=device)[1])
    data_adv.append(test_model(model_adv, test_loader, criterion, attacker=pgd, device=device)[1])
    
plt.plot(data_natural, label='natural')
plt.plot(data_PGD, label='PGD')
plt.plot(data_adv, label='adv')
plt.legend()
plt.show()