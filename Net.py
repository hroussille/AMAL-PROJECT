# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 03:34:14 2020

@author: Hector
"""

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x, mode="prediction"):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        
        if mode == "features":
            return x
        
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)
 