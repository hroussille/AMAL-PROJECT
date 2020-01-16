# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:55:15 2020

@author: Hector
"""

import torch
import numpy as np

class ADV_interp(object):
    def __init__(self, criterion, num_classes=10, epsilon=0.3, epsilon_y=0.5, v_min=0, v_max=1):
        self.criterion = criterion
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.epsilon_y = epsilon_y
        self.v_min = v_min
        self.v_max = v_max

    def attack(self, model, batch_x, batch_y):
        inv_index = torch.arange(batch_x.size(0)-1, -1, -1).long()
        x_prime = batch_x[inv_index, :, :, :].detach()
        y_prime = batch_y[inv_index, :]

        batch_y = batch_y.float()
        y_prime = y_prime.float()

        x_init = batch_x.detach() + torch.zeros_like(batch_x).uniform_(-self.epsilon, self.epsilon)
        x_init.requires_grad = True

        model.eval()

        model.zero_grad()
        loss_adv = self.criterion(model(x_init, mode="features"), model(x_prime, mode="features")).mean()
        loss_adv.backward()

        x_tilde = x_init.data - self.epsilon * torch.sign(x_init.grad.data)
        x_tilde = torch.min(torch.max(x_tilde, batch_x - self.epsilon), batch_x + self.epsilon)
        x_tilde = torch.clamp(x_tilde, self.v_min, self.v_max)

        y_bar_prime = (1 - y_prime) / (self.num_classes - 1)

        y_tilde = (1 - self.epsilon_y) * batch_y + self.epsilon_y * y_bar_prime

        return x_tilde.detach(), y_tilde.detach()
    
class PGD(object):
    def __init__(self, criterion, epsilon=0.3, iter=40, attack_lr=0.01, v_min=0, v_max=1, random_start=True):
        self.criterion = criterion
        self.epsilon = epsilon
        self.iter = iter
        self.attack_lr = attack_lr
        self.v_min = v_min
        self.v_max = v_max
        self.random_start = random_start

    def attack(self, model, batch_x, batch_y):
        model.eval()

        if self.random_start:
            batch_x_tilde = batch_x.detach() + torch.zeros_like(batch_x).uniform_(-self.epsilon, self.epsilon)
            batch_x_tilde = torch.clamp(batch_x_tilde, self.v_min, self.v_max)
        else:
            batch_x_tilde = batch_x.detach()

        for i in range(self.iter):
            
            batch_x_tilde.requires_grad = True
            
            batch_y_hat_tilde = model(batch_x_tilde, mode="prediction")
            
            loss = self.criterion(batch_y_hat_tilde, batch_y)
            
            loss.backward()

            grad = batch_x_tilde.grad.data

            batch_x_tilde = batch_x_tilde + self.attack_lr * torch.sign(grad)
            batch_x_tilde = torch.max(torch.min(batch_x_tilde, batch_x + self.epsilon), batch_x - self.epsilon)
            batch_x_tilde = torch.clamp(batch_x_tilde, self.v_min, self.v_max)
            batch_x_tilde = batch_x_tilde.detach()
        
        batch_x_tilde.requires_grad = True
        
        return batch_x_tilde, batch_y
    
class FGSM(object):
    def __init__(self, criterion, epsilon=8/255, v_min=0, v_max=1):
        self.epsilon = epsilon
        self.v_min = v_min
        self.v_max = v_max
        self.criterion = criterion
    
    def attack(self, model, batch_x, batch_y):
        model.eval()
        
        batch_y_hat = model(batch_x)
        loss = self.criterion(batch_y_hat, batch_y)
        loss.backward()
        
        sign_grads = torch.sign(batch_x.grad.data)
        batch_x_tilde = batch_x + self.epsilon * sign_grads
        batch_x_tilde = torch.clamp(batch_x_tilde, self.v_min. self.v_max)
        batch_x_tilde = batch_x_tilde.detach()
        batch_x_tilde.requires_grad = True
        
        return batch_x_tilde, batch_y
        
        
