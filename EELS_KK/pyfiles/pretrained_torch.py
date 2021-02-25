#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:29:54 2021

@author: isabel
"""
#REVIEW PRETRAINED MODEL

import numpy as np
import random
import os
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MLP(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 10)
        self.linear2 = nn.Linear(10, 15)
        self.linear3 = nn.Linear(15, 5)
        self.output = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.output(x)
        return x


x_min = -0.5
x_max = 20
l = 1000
predict_x_np = np.linspace(x_min, x_max, l).reshape(l,1)
predict_x = torch.from_numpy(predict_x_np)
path_to_models = "train_004_on_I_1"
model = MLP(num_inputs=1, num_outputs=1)

files = np.loadtxt(path_to_models + "/costs.txt")
threshold_costs = 3

plt.figure()
plt.title("chi^2 dist of model")
plt.hist(files)

n_working_models = np.sum(files<threshold_costs)



im = im

eval_y = int(im.image_shape[0]/2)
eval_x = (im.image_shape[1] * np.array([0, 0.2, 0.4, 0.6, 0.8])).astype(int)

predictions = np.zeros((len(eval_x), n_working_models,  im.l))


predict_x_np = np.zeros((im.l,2))
predict_x_np[:,0] = im.deltaE
path_to_models = "train_004_on_I_1"
model = MLP(num_inputs=2, num_outputs=1)


for i in range(len(eval_x)):
    k=0
    plt.figure()
    y_data = im.get_pixel_signal(eval_y, eval_x[i])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))*im.ddeltaE*0.1
    predict_x_np[:,1] = I_i
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[i] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(i)))
                predictions[i,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[i,k,:]), color = "blue")
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
    plt.yscale("log")



for i in range(len(eval_x)):
    k=0
    plt.figure()
    y_data = im.get_pixel_signal(eval_y, eval_x[i])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))*im.ddeltaE*0.1
    predict_x_np[:,1] = I_i
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[i] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(i)))
                predictions[i,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[i,k,:]), color = "blue")
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
    plt.ylim(-500,1e4)
    plt.xlim(0,3)