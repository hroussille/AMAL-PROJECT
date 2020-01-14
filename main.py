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
from attacks import PGD, ADV_interp
import WideResNet
import sys
from tqdm import tqdm
from tqdm import trange

EPOCH = 10
BATCH_SIZE = 32
DEPTH = 22
WIDEN_FACTOR = 10
DROPOUT = 0.3
CLASSES = 10
LEARNING_RATE = 1e-4
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

                # Clear possible gradients accumulated by attacker
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
                 
                 # Clear gradients accumulated by training
                 optimizer.zero_grad()

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
