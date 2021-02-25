#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:24:31 2021

@author: isabel
"""
#TRAINING NN IN PYTORCH
import numpy as np
import random
import os
import scipy
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
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

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def loss_fn(output, target, error):
    loss = torch.mean(torch.square((output - target)/error))
    return loss


def MC_reps(data_avg, data_std, n_rep):
    n_full = len(data_avg)
    full_y_reps = np.zeros(shape=(n_full, n_rep))
    for i in range(n_rep):
        full_rep = np.random.normal(0, data_std)
        full_y_reps[:,i] = (data_avg + full_rep).reshape(n_full)
    return full_y_reps


def split_test_train(data, test_size=0.2):
    #TODO: to use if we do not use single complete spectra
    n_test = round(test_size*data.shape[1])
    train, test = torch.utils.data.random_split(data, [data.shape[1]-n_test, n_test])
    return train
    pass



def train_nn(image, n_rep = 100, n_epochs = 30000, path_to_model = "models", display_step = 1000):
    """training also on intensity, so only one model per image, instead of one model per cluster"""
    if hasattr(image, "name"):
        path_to_model = image.name + "_" + path_to_model
    
    if not os.path.exists(path_to_model):
        Path(path_to_model).mkdir(parents=True, exist_ok=True)
    else:
        ans = input("The directory " + path_to_model + " already exists, if there are trained models " +
                    "in this folder, they will be overwritten. Do you want to continue? \n"+
                    "yes [y], no [n], define new path[dp]\n")
        if ans[0] == 'n':
            return
        elif not ans[0] == 'y':
            path_to_model = input("Please define the new path: \n")
    
    spectra = image.get_cluster_spectra()
    
    if display_step is None:
        print_progress = False
    
    
    
    
    loss_test_reps = np.zeros(n_rep)
    n_data = image.l*image.n_clusters
    
    data_sigma = np.zeros((n_data,1))
    for cluster in range(image.n_clusters):
        ci_low = np.nanpercentile(spectra[cluster], 16, axis= 0)
        ci_high = np.nanpercentile(spectra[cluster], 84, axis= 0)
        data_sigma[cluster*image.l : (cluster+1)*image.l,0] = np.absolute(ci_high-ci_low)
    
    for i in range(n_rep):
        if print_progress: print("Started training on replica number {}".format(i) + ", at time ", datetime.now())
        data = np.zeros((n_data,1))
        data_x = np.zeros((n_data,2))
        
        for cluster in range(image.n_clusters):
            n_cluster = len(data[cluster])
            idx = random.randint(0,n_cluster-1)
            data[cluster*image.l : (cluster+1)*image.l,0] = np.log(spectra[cluster][idx])
            data_x[cluster*image.l : (cluster+1)*image.l,0] = image.deltaE
            data_x[cluster*image.l : (cluster+1)*image.l,1] = np.sum(np.log(spectra[cluster][idx]))
        
        
        

        model = MLP(num_inputs=2, num_outputs=1)
        model.apply(weight_reset)
        #optimizer = optim.RMSprop(model.parameters(), lr=6 * 1e-3, eps=1e-5, momentum=0.0, alpha = 0.9)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        """
        # TODO: rewrite to include pytorch directly, see pyfiles/train_nn.py
        full_y = full_y_reps[:, i].reshape(N_full, 1)
        train_x, train_y,train_sigma = full_x, full_y, full_sigma
        train_x = train_x.reshape(N_full, 1)
        train_y = train_y.reshape(N_full, 1)
        train_sigma = train_sigma.reshape(N_full, 1)
        """
        #full_y = full_y_reps[:, i].reshape(N_full,1)
        #train_x, test_x, train_y, test_y, train_sigma, test_sigma = \
        #    train_test_split(full_x, full_y, full_sigma, test_size=.2)
        
        
        #data = data.reshape(n_data, 1)
        
        train_x, test_x, train_y, test_y, train_sigma, test_sigma = train_test_split(data_x, data, data_sigma, test_size=0.2)
        
        N_test = len(test_x)
        N_train = len(train_x)
        
        test_x = test_x.reshape(N_test, 1)
        test_y = test_y.reshape(N_test, 2)
        train_x = train_x.reshape(N_train, 1)
        train_y = train_y.reshape(N_train, 1)
        train_sigma = train_sigma.reshape(N_train, 1)
        test_sigma = test_sigma.reshape(N_test, 1)
        
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        train_sigma = torch.from_numpy(train_sigma)
        
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        test_sigma = torch.from_numpy(test_sigma)
        
        # train_data_x, train_data_y, train_errors = get_batch(i)
        #loss_train = np.zeros(n_epochs)
        loss_test = np.zeros(n_epochs)
        min_loss_test = 1e6 #big number
        for epoch in range(1, n_epochs + 1):
            model.train()
            output = model(train_x.float())
            loss_train = loss_fn(output, train_y, train_sigma)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            #if epoch % display_step == 0:
            #    print('Rep {}, Epoch {}, Training loss {}'.format(i, epoch, loss_train))
            
            model.eval()
            with torch.no_grad():
                output_test = model(test_x.float())
                loss_test[epoch-1] = loss_fn(output_test, test_y, test_sigma)
                if epoch % display_step == 0 and print_progress:
                    print('Rep {}, Epoch {}, Training loss {}, Testing loss {}'.format(i, epoch, loss_train, loss_test[epoch-1]))
                if loss_test[epoch-1] < min_loss_test:
                    torch.save(model.state_dict(), path_to_model + "/nn_rep" + str(i))
                    loss_test_reps[i] = loss_test[epoch-1]
                    min_loss_test = loss_test_reps[i]
                    #iets met copy.deepcopy(model)
        np.savetxt(path_to_model+ "/costs.txt", loss_test_reps)
    
    
    
    
    
    
    
    
    #predict_x = np.concatenate((predict_x, np.vstack((image.deltaE,np.ones(image.l)*intensities[i])).T))
    
















#OLD
def binned_statistics(x,y, nbins, stats = None):
    """Find the mean, variance and number of counts within the bins described by ewd"""
    if stats is None:
        stats = []
        edges = None
    
    x = np.tile(x, len(y))
    y = y.flatten()
        
    
    
    #df_train, 
    cuts1, cuts2 = ewd(x,nbins)
    result = []
    if "mean" in stats:
        mean, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='mean', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='mean', bins=cuts2)
        result.append(mean)
    if "var" in stats:
        #var, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='std', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='std', bins=cuts2)
        low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
        high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)            
        var = high-low
        result.append(var)
    if "count" in stats:
        count, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='count', bins=cuts2)
        result.append(count)
    if "low" in stats:
        low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
        result.append(low)
    if "high" in stats:
        high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)
        result.append(high)
    if "mean2" in stats:
        mean2, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=get_mean, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=get_mean, bins=cuts2)
        result.append(mean2)
    
    return result, edges

def ewd(x, nbins):  
    """
    INPUT:
        x: 
        y:
        nbins: 
            
    OUTPUT:
        df_train:
        cuts1:
        cuts2:
    
    Apply Equal Width Discretization (EWD) to x and y data to determine variances
    """
    #TODO: I think everything that was here isn't needed?? since x is already sorted, and a 1D array
    #df_train = np.array(np.c_[x,y])
    cuts1, cuts2 = pd.cut(x, nbins, retbins=True)
    
    return cuts1, cuts2

def CI_high(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = np.array(data)
    n = len(a)
    b = np.sort(data)

    highest = np.int((1-(1-confidence)/2)*n)
    high_a = b[highest]
 
    return high_a

def CI_low(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values
    
    a = np.array(data)
    n = len(a)
    b = np.sort(data)
    lowest = np.int(((1-confidence)/2)*n)
    low_a = b[lowest]

    return low_a

def get_median(x,y,nbins):
    #df_train, 
    cuts1, cuts2 = ewd(x,y, nbins)
    median, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='median', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='median', bins=cuts2)
    return median

def get_mean(data):
    return np.mean(data)