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
import sys
from tqdm import tqdm
from tqdm import trange

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

class ADV_interp(object):
    def __init__(self, criterion, num_classes=10, epsilon=1/32, epsilon_y=0.25, v_min=-1, v_max=1):
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

"""
def adversarial_interpolation(batch_x, batch_y, model, criterion, num_classes=10, epsilon=1/32, epsilon_y=0.25, v_min=-1, v_max=1):

    inv_index = torch.arange(batch_x.size(0)-1, -1, -1).long()
    x_prime = batch_x[inv_index, :, :, :].detach()
    y_prime = batch_y[inv_index, :]

    batch_y = batch_y.float()
    y_prime = y_prime.float()

    x_init = batch_x.detach() + torch.zeros_like(batch_x).uniform_(-epsilon, epsilon)
    x_init.requires_grad = True

    model.eval()

    model.zero_grad()
    loss_adv = criterion(model(x_init, mode="features"), model(x_prime, mode="features")).mean()
    loss_adv.backward()

    x_tilde = x_init.data - epsilon * torch.sign(x_init.grad.data)
    x_tilde = torch.min(torch.max(x_tilde, batch_x - epsilon), batch_x + epsilon)
    x_tilde = torch.clamp(x_tilde, v_min, v_max)

    y_bar_prime = (1 - y_prime) / (num_classes - 1)

    y_tilde = (1 - epsilon_y) * batch_y + epsilon_y * y_bar_prime

    return x_tilde.detach(), y_tilde.detach()
"""


class PGD(object):
    def __init__(self, criterion, epsilon=3e-3, iter=5, attack_lr=1e-2, v_min=-1, v_max=1, random_start=True):
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

        batch_x = batch_x.cpu().numpy()

        for i in range(self.iter):
            batch_x_tilde.requires_grad = True
            batch_y_hat_tilde = model(batch_x_tilde)
            loss = self.criterion(batch_y_hat_tilde, batch_y)
            loss.backward()

            grad = batch_x_tilde.grad.data

            batch_x_tilde = batch_x_tilde + self.attack_lr * torch.sign(grad)
            batch_x_tilde = batch_x_tilde.cpu().detach().numpy()

            batch_x_tilde = np.clip(batch_x_tilde, batch_x - self.epsilon, batch_x + self.epsilon)
            batch_x_tilde = np.clip(batch_x_tilde, self.v_min, self.v_max)
            batch_x_tilde = torch.Tensor(batch_x_tilde)

        return batch_x_tilde, batch_y

"""
def PGD(batch_x, batch_y, model, criterion, epsilon=3e-3, iter=5, attack_lr=1e-2, v_min=-1, v_max=1, random_start=True):

    model.eval()

    if random_start:
        batch_x_tilde = batch_x.detach() + torch.zeros_like(batch_x).uniform_(-epsilon, epsilon)
        batch_x_tilde = torch.clamp(batch_x_tilde, v_min, v_max)
    else:
        batch_x_tilde = batch_x.detach()

    batch_x = batch_x.cpu().numpy()

    for i in range(iter):
        batch_x_tilde.requires_grad = True
        batch_y_hat_tilde = model(batch_x_tilde)
        loss = criterion(batch_y_hat_tilde, batch_y)
        loss.backward()

        grad = batch_x_tilde.grad.data

        batch_x_tilde = batch_x_tilde + attack_lr * torch.sign(grad)
        batch_x_tilde = batch_x_tilde.cpu().detach().numpy()

        batch_x_tilde = np.clip(batch_x_tilde, batch_x - epsilon, batch_x + epsilon)
        batch_x_tilde = np.clip(batch_x_tilde, v_min, v_max)
        batch_x_tilde = torch.Tensor(batch_x_tilde)

    return batch_x_tilde, batch_y
"""

def test_model(model, test_loader, criterion, device="cpu"):

    model.to(device)
    model.eval()

    testing_accuracy = 0
    testing_CE = 0

    with torch.no_grad():

        for iteration, (batch_x, batch_y) in tqdm(enumerate(test_loader)):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_y_hat = model(batch_x, mode="prediction")

            loss = criterion(batch_y_hat, batch_y)

            current_accuracy = torch.eq(torch.max(batch_y_hat, 1)[1], torch.max(batch_y, 1)[1]).sum().item() / batch_y.shape[0]
            current_CE = loss.detach().cpu().item()

            testing_CE += current_CE
            testing_accuracy += current_accuracy

    tqdm.write("Testing Results - Avg accuracy: {:.2f} Avg loss: {:.2f}".format(testing_accuracy / len(test_loader), testing_CE / len(test_loader)))

def train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, nb_epoch=10, attacker=None, device="cpu"):

    time_loss = []
    model.to(device)

    t_epoch = trange(nb_epoch, desc="Training", leave=True)
    for epoch in t_epoch:

        t_epoch.set_description("Training EPOCH {} / {}".format(epoch + 1,  nb_epoch))
        t_epoch.refresh() # to show immediately the update
        training_accuracy = 0
        training_CE = 0

        with tqdm(total = len(train_loader), file=sys.stdout) as pbar:
            for iteration, (batch_x, batch_y) in enumerate(train_loader):

                 batch_x = batch_x.to(device)
                 batch_y = batch_y.to(device)

                 if attacker is not None:
                     batch_x, batch_y = attacker.attack(model, batch_x, batch_y)

                 optimizer.zero_grad()
                 model.train()

                 batch_y_hat = model(batch_x, mode="prediction")

                 loss = criterion(batch_y_hat, batch_y)
                 loss.backward()
                 optimizer.step()

                 current_accuracy = torch.eq(torch.max(batch_y_hat, 1)[1], torch.max(batch_y, 1)[1]).sum().item() / batch_y.shape[0]
                 current_CE = loss.detach().cpu().item()

                 training_CE += current_CE
                 time_loss.append(current_CE)
                 training_accuracy += current_accuracy

                 pbar.update(1)
                 pbar.set_description("Iteration : {} Accuracy : {:0.2f} Loss : {:0.2f}".format(iteration, current_accuracy, current_CE))
                 pbar.refresh()

            tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:0.2f} Avg loss: {:0.2f}".format(epoch, training_accuracy / len(train_loader), training_CE / len(train_loader)))


            test_model(model, test_loader, criterion, device=device)

            torch.save(model.state_dict(), checkpoint.format(epoch))

    return np.array(time_loss)

t = trange(NB_MODELS, desc="PGD learning", leave=True)
for i in t:
    t.set_description("PGD learning model {} / {} :".format(i + 1, NB_MODELS))
    t.refresh()
    checkpoint = "models/PGD_" + str(i) + "_epoch_{}"
    model = WideResNet.WideResNet(DEPTH, WIDEN_FACTOR, DROPOUT, CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    time_loss = train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, nb_epoch=EPOCH, attacker=PGD(Cross_Entropy), device=device)
    model.eval()
    np.save("losses/PGD_{}".format(i), time_loss)
    torch.save(model.state_dict(), "models/PGD_{}".format(i))

t = trange(NB_MODELS, desc="Natural learning", leave=True)
for i in t:
    t.set_description("Natural learning model {} / {} :".format(i + 1, NB_MODELS))
    t.refresh()
    checkpoint = "models/natural_" + str(i) + "_epoch_{}"
    model = WideResNet.WideResNet(DEPTH, WIDEN_FACTOR, DROPOUT, CLASSES)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    time_loss = train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, nb_epoch=EPOCH, attack=None, device=device)
    model.eval()
    np.save("losses/natural_{}".format(i), time_loss)
    torch.save(model.state_dict(), "models/natural_{}".format(i))


t = trange(NB_MODELS, desc="Adversarial Interpolation learning", leave=True)
for i in t:
    t.set_description("Adversarial Interpolation learning model {} / {} :".format(i + 1, NB_MODELS))
    t.refresh()
    checkpoint = "models/adversarial_interpolation_" + str(i) + "_epoch_{}"
    model = WideResNet.WideResNet(DEPTH, WIDEN_FACTOR, DROPOUT, CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    time_loss = train_model(model, optimizer, train_loader, test_loader, criterion, checkpoint, nb_epoch=EPOCH, attacker=ADV_interp(torch.nn.CosineSimilarity()), device=device)
    model.eval()
    np.save("losses/adversarial_interpolation_{}".format(i), time_loss)
    torch.save(model.state_dict(), "models/adversarial_interpolation_{}".format(i))
