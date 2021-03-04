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
from image_class import Spectral_image  

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

def scale(inp, ab):
    """
    min_inp = inp.min()
    max_inp = inp.max()
    
    outp = inp/(max_inp-min_inp) * (max_out-min_out)
    outp -= outp.min()
    outp += min_out
    
    return outp
    """
    
    return inp*ab[0] + ab[1]
    #pass

def find_scale_var(inp, min_out = 0.1, max_out=0.9):
    a = (max_out - min_out)/(inp.max()- inp.min())
    b = min_out - a*inp.min()
    return [a, b]



x_min = -0.5
x_max = 20
l = 1000
predict_x_np = np.linspace(x_min, x_max, l).reshape(l,1)
predict_x = torch.from_numpy(predict_x_np)
path_to_models = "train_004_on_I_scaled_5"
model = MLP(num_inputs=1, num_outputs=1)

files = np.loadtxt(path_to_models + "/costs.txt")
threshold_costs = 1

plt.figure()
plt.title("chi^2 dist of model")
plt.hist(files)

n_working_models = np.sum(files<threshold_costs)



#im = im
im = Spectral_image.load_data('../dmfiles/h-ws2_eels-SI_004.dm4')


ab_deltaE = find_scale_var(im.deltaE)
deltaE_scaled = scale(im.deltaE, ab_deltaE)

all_spectra = im.data
all_spectra[all_spectra<1] = 1
int_log_I = np.sum(np.log(all_spectra), axis=2).flatten()
ab_int_log_I = find_scale_var(int_log_I)
del all_spectra


eval_y = int(im.image_shape[0]/2)
eval_x = (im.image_shape[1] * np.array([0, 0.2, 0.4, 0.6, 0.8])).astype(int)

predictions = np.zeros((len(eval_x), n_working_models,  im.l))


predict_x_np = np.zeros((im.l,2))
predict_x_np[:,0] = deltaE_scaled
#path_to_models = "train_004_on_I_1"
model = MLP(num_inputs=2, num_outputs=1)


for x in range(len(eval_x)):
    k=0
    plt.figure()
    plt.title("results " + path_to_models)
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    y_data = im.get_pixel_signal(eval_y, eval_x[x])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))
    predict_x_np[:,1] = scale(I_i, ab_int_log_I)
    #print(predict_x_np)
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[j] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(j)))
                predictions[x,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[x,k,:]), color = "blue", alpha = 0.3)
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
    plt.yscale("log")



for i in range(len(eval_x)):
    k=0
    plt.figure()
    plt.title("results " + path_to_models)
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    y_data = im.get_pixel_signal(eval_y, eval_x[i])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))
    predict_x_np[:,1] = scale(I_i, ab_int_log_I)
    #print(predict_x_np)
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[j] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(i)))
                predictions[i,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[i,k,:]), color = "blue", alpha = 0.3)
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
    plt.ylim(-500,1e4)
    plt.xlim(0,3)

for i in range(len(eval_x)):
    k=0
    plt.figure()
    plt.title("results " + path_to_models)
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    y_data = im.get_pixel_signal(eval_y, eval_x[i])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))
    predict_x_np[:,1] = scale(I_i, ab_int_log_I)
    print(predict_x_np)
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[j] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(i)))
                predictions[i,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[i,k,:]), color = "blue", alpha = 0.1)
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
    plt.ylim(-500,1e4)
    plt.xlim(0,15)

for i in range(len(eval_x)):
    k=0
    plt.figure()
    plt.title("results " + path_to_models)
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    y_data = im.get_pixel_signal(eval_y, eval_x[i])
    y_data[y_data<1] = 1
    I_i = np.sum(np.log(y_data))
    predict_x_np[:,1] = scale(I_i, ab_int_log_I)
    print(predict_x_np)
    predict_x = torch.from_numpy(predict_x_np)
    
    for j in range(len(files)):
        if files[j] < threshold_costs:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(i)))
                predictions[i,k,:] = model(predict_x.float()).flatten()
            plt.plot(im.deltaE, np.exp(predictions[i,k,:]), color = "blue", alpha = 0.1)
            k+=1
    plt.plot(im.deltaE, y_data, color = "black")
