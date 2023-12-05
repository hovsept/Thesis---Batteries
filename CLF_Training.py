# -*- SPMcoding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:03 2023

@author: Hovsep Touloujian
"""

import numpy as np
from SPM_Params import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.optimize
from functools import *

import torch
import torch.nn as nn
import torch.nn as F
from torchsummary import summary

n = p['nrn']+p['nrp']+p['n_sei']+4

class CLF(nn.Module):
    
    def __init__(self):
        super(CLF,self).__init__()
        
        self.linear1 = nn.Linear(n, 20,bias=False)
        self.linear2 = nn.Linear(20,10,bias=False)
        self.linear3 = nn.Linear(10,1,bias=False)
        
        self.linear4 = nn.Linear(n,10,bias=False)
        self.linear5 = nn.Linear(10,10,bias=False)
        self.linear6 = nn.Linear(10,1,bias=False)
        
    def forward(self,x):
        V = torch.square(self.linear1(x))
        V = torch.square(self.linear2(V))
        V = self.linear3(V)
        
        u = torch.relu(self.linear4(x))
        u = torch.relu(self.linear5(u))
        # u = 10*torch.sigmoid(0.005*self.linear6(u))-5
        u = 10*(2/torch.pi) * torch.atan(0.01*self.linear6(u))
        # u = self.linear6(u)
        
        return torch.vstack((V,u))
    
clf = CLF()

#Generate Samples in State Space
x_I = torch.transpose(torch.tensor(set_sample(2e-5, 0, 0.3, 0),dtype=torch.float32, requires_grad=True),0,1)
x_SnG = torch.transpose(torch.tensor(set_sample(2e-5, 0, 0.9, 0),dtype=torch.float32, requires_grad=True),0,1)
x_dS = torch.transpose(torch.tensor(set_sample(2e-5, 0, 0.9, 1),dtype=torch.float32, requires_grad=True),0,1)
for i in range(20):
    x_I = torch.vstack((x_I,torch.transpose(torch.tensor(set_sample(2e-5, 0, 0.3, 0),dtype=torch.float32, requires_grad=True),0,1)))
    x_SnG = torch.vstack((x_SnG,torch.transpose(torch.tensor(set_sample(2e-5, 0, 0.9, 0),dtype=torch.float32, requires_grad=True),0,1)))
    x_dS = torch.vstack((x_dS,torch.transpose(torch.tensor(set_sample(2e-5, 0, 1, 1),dtype=torch.float32, requires_grad=True),0,1)))

a = 1e-15
loss_fn = nn.LeakyReLU(a)
loss_k = []
dVf_max_k = []
lr = 1e-3
optimizer = torch.optim.RMSprop(clf.parameters(), lr=lr)
eps = 0.01
for t in tqdm(range(250)):
    L_I = torch.sum(loss_fn(clf(x_I)[::2] + eps))
    L_dS = torch.sum(loss_fn(-clf(x_dS)[::2] + eps))
    
    dVf = []
    L_SnG = 0
    for i in range(x_SnG.size()[0]):
        dV = torch.autograd.grad(clf(x_SnG[i])[0],x_SnG)[0][i]
        f = torch.tensor(f_SPM(x_SnG.detach().numpy()[i],clf(x_SnG[i])[1].detach().numpy(),p)[0],dtype=torch.float32)
        dVf.append(torch.matmul(dV,f).detach().numpy())
        L_SnG += loss_fn(torch.matmul(dV, f) + eps)
    
    loss = L_I + L_dS + L_SnG
    loss_k.append(loss.detach())
    dVf_max_k.append(max(dVf)[0])
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(loss_k)
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.grid()
plt.show()

plt.plot(dVf_max_k)
plt.ylabel('max(dV/dx * f(x,u))')
plt.xlabel('Iteration')
plt.grid()
plt.show()

print("x_I: ", np.max(clf(x_I)[::2].detach().numpy()))
print("x_dS: ", np.min(clf(x_dS)[::2].detach().numpy()))
print("x_S\G: ", dVf_max_k[-1])


