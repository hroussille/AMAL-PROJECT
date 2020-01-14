# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:31:35 2020

@author: Hector
"""

import torchvision
import torch
from torch.utils.data import DataLoader
import numpy as np
import torchvision.transforms as transforms
import WideResNet
import copy

EPOCH = 10
BATCH_SIZE = 32
DEPTH = 22
WIDEN_FACTOR = 10
DROPOUT = 0.3
CLASSES = 10
LEARNING_RATE = 1e-2
LOG_INTERVAL = 1
NB_MODELS = 1

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
    

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
target_transform = target_transform
    
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform, target_transform=target_transform)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform, target_transform=target_transform)
    
train_loader = DataLoader(train, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)
    
def Cross_Entropy(predicted, target):
     return -(target * predicted).sum(dim=1).mean()
 
criterion = Cross_Entropy
    
def adversarial_interpolation(batch_x, batch_y, model, distance, num_classes=10, epsilon=1/32, epsilon_y=0.25, v_min=-1, v_max=1):
    
    inv_index = torch.arange(batch_x.size(0)-1, -1, -1).long()
    x_prime = batch_x[inv_index, :, :, :].detach()
    y_prime = batch_y[inv_index, :]
    
    batch_y = batch_y.float()
    y_prime = y_prime.float()
    
    x_init = batch_x.detach() + torch.zeros_like(batch_x).uniform_(-epsilon, epsilon)
    x_init.requires_grad = True
    
    adv_model = copy.deepcopy(model)
    adv_model.eval()

    adv_model.zero_grad()    
    loss_adv = distance(adv_model(x_init, mode="features"), adv_model(x_prime, mode="features")).mean()
    loss_adv.backward()
    
    x_tilde = x_init.data - epsilon * torch.sign(x_init.grad.data)
    x_tilde = torch.min(torch.max(x_tilde, batch_x - epsilon), batch_x + epsilon)
    x_tilde = torch.clamp(x_tilde, v_min, v_max)
    
    y_bar_prime = (1 - y_prime) / (num_classes - 1)
    
    y_tilde = (1 - epsilon_y) * batch_y + epsilon_y * y_bar_prime 
    
    return x_tilde.detach(), y_tilde.detach()

def PGD():
    pass

def test_model(model, test_loader, criterion, device="cpu"):
    
    model.to(device)
    model.eval()
    
    testing_accuracy = 0
    testing_CE = 0
    
    with torch.no_grad():
        
        for iteration, (batch_x, batch_y) in enumerate(test_loader):
                 
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
                    
            batch_y_hat = model(batch_x, mode="prediction")
                 
            loss = criterion(batch_y_hat, batch_y)
    
            current_accuracy = torch.eq(torch.max(batch_y_hat, 1)[1], torch.max(batch_y, 1)[1]).sum().item() / batch_y.shape[0]
            current_CE = loss.detach().cpu().item()
                    
            testing_CE += current_CE
            testing_accuracy += current_accuracy
         
    print("Testing Results - Avg accuracy: {:.2f} Avg loss: {:.2f}".format(testing_accuracy / len(test_loader), testing_CE / len(test_loader)))

def train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, distance=None, nb_epoch=10, mode="natural", device="cpu"):
    
    time_loss = []
    model.to(device)
    
    for epoch in range(nb_epoch):
        
         model.train()
         training_accuracy = 0
         training_CE = 0
         
         for iteration, (batch_x, batch_y) in enumerate(train_loader):
             optimizer.zero_grad()
             
             batch_x = batch_x.to(device)
             batch_y = batch_y.to(device)
             
             if mode == "adversarial_interpolation":
                 batch_x, batch_y = adversarial_interpolation(batch_x, batch_y, model, distance, num_classes=CLASSES)
                 
             elif mode == "PGD":
                 batch_x, batch_y = PGD(batch_x, batch_y, model)
                
             batch_y_hat = model(batch_x, mode="prediction")
             
             loss = criterion(batch_y_hat, batch_y)
             loss.backward()
             optimizer.step()
    
             current_accuracy = torch.eq(torch.max(batch_y_hat, 1)[1], torch.max(batch_y, 1)[1]).sum().item() / batch_y.shape[0]
             current_CE = loss.detach().cpu().item()
                
             training_CE += current_CE
             time_loss.append(current_CE)
             training_accuracy += current_accuracy
         
         print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(epoch, training_accuracy / len(train_loader), training_CE / len(train_loader)))
         
         test_model(model, test_loader, criterion, device=device)
         
         torch.save(model.state_dict(), checkpoint.format(epoch))
    
    return np.array(time_loss)

for i in range(NB_MODELS):
    print("Training natural model ({} / {}) : ".format(i + 1, NB_MODELS))
    checkpoint = "models/natural_" + str(i) + "_epoch_{}"
    model = WideResNet.WideResNet(DEPTH, WIDEN_FACTOR, DROPOUT, CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    time_loss = train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, distance=None, nb_epoch=EPOCH, mode="natural", device=device)
    model.eval()
    np.save("losses/natural_{}".format(i), time_loss)
    torch.save(model.state_dict(), "models/natural_{}".format(i))
    
    
for i in range(NB_MODELS):
    print("Training adversarial interpolation model ({} / {}) : ".format(i + 1, NB_MODELS))
    checkpoint = "models/adversarial_interpolation_" + str(i) + "_epoch_{}"
    model = WideResNet.WideResNet(DEPTH, WIDEN_FACTOR, DROPOUT, CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    time_loss = train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, distance=torch.nn.CosineSimilarity(), nb_epoch=EPOCH, mode="adversarial_interpolation", device=device)
    model.eval()
    np.save("losses/adversarial_interpolation_{}".format(i), time_loss)
    torch.save(model.state_dict(), "models/adversarial_interpolation_{}".format(i))