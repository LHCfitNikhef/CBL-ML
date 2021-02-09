#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:20:45 2021

@author: isabel
TRAINING ZLP MODEL
"""
import numpy as np
import pandas as pd
import math
from copy import copy
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc, cm
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
import tensorflow.compat.v1 as tf


#TODO: change from binned statistics to eliminate hortizontal uncertainty?


#FROM LAURIEN
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



def gaussian(x, amp, cen, std):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    y = (amp) * np.exp(-(x-cen)**2 / (2*std**2))
    return y


def smooth_lau(x,window_len=10,window='hanning'):
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

    """
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    index = int(window_len/2)
    return y[(index-1):-(index)]

def smooth_im(self, window_len=10,window='hanning', keep_original = False):
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

    """
    #TODO: add comnparison
    window_len += (window_len+1)%2
    s=np.r_['-1', self.data[:,:,window_len-1:0:-1],self.data,self.data[:,:,-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    #y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len-1)*0.5)
    if keep_original:
        self.data_smooth = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,:,surplus_data:-surplus_data]
    else:
        self.data = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,:,surplus_data:-surplus_data]
    
    
    return #y[(window_len-1):-(window_len)]

def smooth(data, window_len=10,window='hanning', keep_original = False):
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

    """
    #TODO: add comnparison
    window_len += (window_len+1)%2
    s=np.r_['-1', data[:,window_len-1:0:-1],data,data[:,-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    #y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len-1)*0.5)
    return np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,surplus_data:-surplus_data]



def fun_clusters(clusters, function, **kwargs):
    #TODO
    pass

def smooth_clusters(image, clusters, window_len = None):
    smoothed_clusters = np.zeros((image.n_clusters), dtype = object)
    for i in range(image.n_clusters):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters

def derivative_clusters(image, clusters):
    dx = image.ddeltaE
    der_clusters = np.zeros((image.n_clusters), dtype = object)
    for i in range(image.n_clusters):
        der_clusters[i] = (clusters[i][:,1:]-clusters[i][:,:-1])/dx
    return der_clusters
    

def residuals(prediction, y, std):
    res = np.divide((prediction - y), std)
    return res

def make_model_lau(inputs, n_outputs):
    hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
    hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
    output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
    return output

def make_model(inputs, n_outputs):
    hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
    hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
    hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
    output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
    return output

def train_NN(image, spectra):#, vacuum_in):
    
    #reset tensorflow
    tf.get_default_graph
    tf.disable_eager_execution()
    #oud:
    wl1 = 50
    wl2 = 100
    
    #new??? #TODO
    wl1 = round(image.l/20)
    wl2 = wl1*2
    units_per_bin = 4
    nbins = round(image.l/units_per_bin)#150
    
    
    #filter out negatives and 0's
    for i  in range(image.n_clusters):
        spectra[i][spectra[i]<1] = 1
    
    
    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)
    #dE1s = find_clusters_dE1(image, smooth_dy_dx, spectra_smooth)
    
    
    dE1 = determine_dE1_new(image, smooth_dy_dx, spectra_smooth)#dE1s, dy_dx)
    
    
    #TODO: instead of the binned statistics, just use xth value to dischart -> neh says Juan    
    
    dE2 = determine_dE2_new(image, spectra_smooth, smooth_dy_dx)#[0], nbins, dE1)
    
    print("dE1 & dE2:", np.round(dE1,3), dE2)
    
    spectra_mean, spectra_var, cluster_intensities, deltaE = create_data(image, spectra, dE1, dE2, units_per_bin)
    
    x = tf.placeholder(["float", "float"], [None, 1], name="x")
    y = tf.placeholder("float", [None, 1], name="y")
    sigma = tf.placeholder("float", [None, 1], name="sigma")
    
    predictions = make_model(x,1)
    
    full_x = np.vstack((spectra_mean,cluster_intensities)).T
    full_y = spectra_mean # = df_train_full.drop_duplicates(subset = ['x']) # Only keep one copy per x-value
    full_sigma = spectra_var
    del spectra_mean, spectra_var
    
    
    #N_full = len(df_train_full['x'])
    
    #full_x = np.copy(df_train_full['x']).reshape(N_full,1)
    #full_y = np.copy(df_train_full['y']).reshape(N_full,1)
    #full_sigma = np.copy(df_train_full['sigma']).reshape(N_full,1)
    
    #N_pred = 3000
    #pred_min = -.5
    #pred_max = 20
    predict_x = image.deltaE #np.linspace(pred_min,pred_max,N_pred).reshape(N_pred,1)
    
    #print("Dataset is split into train subset (80%) and validation subset (20%)")
    
    
    #MONTE CARLO
    Nrep = 1000

    full_y_reps = np.zeros(shape=(N_full, Nrep))
    i=0
    while i < Nrep:
            full_rep = np.random.normal(0, full_sigma)
            full_y_reps[:,i] = (full_y + full_rep).reshape(N_full)
            i+=1 
            
    std_reps = np.std(full_y_reps, axis=1)
    mean_reps = np.mean(full_y_reps, axis=1)
    
    print('MC pseudo data has been created for 1000 replicas')
    
    
    N_train = int(.8 * N_full)
    N_test = int(.2 * N_full)
    
    
    
    
    
    
    
    pass

def find_dE1(image, dy_dx, y_smooth):
    #crossing
    #first positive derivative after dE=0:
    
    crossing = (dy_dx > 0)
    if not crossing.any():
        print("shouldn't get here")
        up = np.argmin(np.absolute(dy_dx)[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
    else:
        up = np.argmax(crossing[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
    pos_der = image.deltaE[up]
    return pos_der
    

def determine_dE1_new(image, dy_dx_clusters, y_smooth_clusters, check_with_user = True):
    dy_dx_avg = np.zeros((image.n_clusters, image.l-1))
    dE1_clusters = np.zeros(image.n_clusters)
    for i in range(image.n_clusters):
        dy_dx_avg[i,:] = np.average(dy_dx_clusters[i], axis=0)
        y_smooth_cluster_avg = np.average(y_smooth_clusters[i], axis=0)
        dE1_clusters[i] = find_dE1(image, dy_dx_avg[i,:], y_smooth_cluster_avg)
        
    if not check_with_user:
        return dE1_clusters
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if len(colors) < image.n_clusters:
        print("thats too many clusters to effectively plot, man")
        return dE1_clusters
        #TODO: be kinder
    der_deltaE = image.deltaE[:-1]
    plt.figure()
    for i in range(image.n_clusters):
        dx_dy_i_avg = dy_dx_avg[i,:]
        #dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        ci_low = np.nanpercentile(dy_dx_clusters[i],  16, axis=0)
        ci_high = np.nanpercentile(dy_dx_clusters[i],  84, axis=0)
        plt.fill_between(der_deltaE,ci_low, ci_high, color = colors[i], alpha = 0.2)
        plt.vlines(dE1_clusters[i], -3E3, 2E3, ls = 'dotted', color= colors[i])
        if i == 0:
            lab = "vacuum"
        else:
            lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg, color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[0,0], color = 'black')
    plt.title("derivatives of EELS per cluster, and range of first \npositive derivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("dy/dx")
    plt.legend()
    plt.xlim(np.min(dE1_clusters)/4, np.max(dE1_clusters)*2)
    plt.ylim(-3e3,2e3)
    
    
    plt.figure()
    for i in range(1,image.n_clusters):
        dx_dy_i_avg = dy_dx_avg[i,:]
        dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        #dx_dy_i_min = np.min(dy_dx_clusters[i], axis = 0)
        #plt.fill_between(image.deltaE, dx_dy_i_avg-dx_dy_i_std, dx_dy_i_avg+dx_dy_i_std, color = colors[i], alpha = 0.2)
        #plt.axvspand(dE1_i_avg-dE1_i_std, dE1_i_avg+dE1_i_std, color = colors[i], alpha=0.1)
        plt.vlines(dE1_clusters[i], -2, 1, ls = 'dotted', color= colors[i])
        lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg/dy_dx_avg[0,:], color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[1,1], color = 'black')
    plt.title("ratio between derivatives of EELS per cluster and the  \nderivative of vacuum cluster, and average of first positive \nderivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio dy/dx sample and dy/dx vacuum")
    plt.legend()
    plt.xlim(np.min(dE1_clusters)/4, np.max(dE1_clusters)*2)
    plt.ylim(-2,3)
    plt.show()
    print("please review the two auxillary plots on the derivatives of the EEL spectra. \n"+\
          "dE1 is the point before which the influence of the sample on the spectra is negligiable.") #TODO: check spelling
    
    for i in range(image.n_clusters):
        if i == 0: name = "vacuum cluster"
        else: name = "sample cluster " + str(i)
        dE1_clusters[i] = user_check("dE1 of " + name, dE1_clusters[i])
    return dE1_clusters


def create_data(image, spectra_clusters, dE1, dE2, units_per_bin):
    min_pseudo_bins = 20
    #TODO: do we want to do this?
    #n_pseudo_bins = math.floor(len(image.deltaE[image.deltaE>dE2])/units_per_bin)
    #n_pseudo_bins = max(min_pseudo_bins, n_pseudo_bins)
    n_pseudo_bins = min_pseudo_bins
    
    cluster_intensities = np.zeros(0)
    spectra_log_var = np.zeros(0)#image.n_clusters, dtype=object)
    spectra_log_mean = np.zeros(0)
    deltaE = np.zeros(0)#image.n_clusters, dtype=object)
    #pseudo_data = np.zeros(image.n_clusters, dtype=object)
    for i in range(image.n_clusters):
        n_bins = math.floor(len(image.deltaE[image.deltaE<dE1[i]])/units_per_bin)
        n = n_bins*units_per_bin
        #[spectra[i], spectra_var[i]], edges   = binned_statistics(image.deltaE[:n], spectra[i][:, :n], n_bins, stats=["mean", "var"])
        #deltaE[i] = (edges[1:]+edges[:-1])/2
        [i_log_means, i_log_vars], edges = binned_statistics(image.deltaE[:n], np.log(spectra_clusters[i][:, :n]), n_bins, stats=["mean", "var"])
        spectra_log_mean = np.append(spectra_log_mean, i_log_means)
        spectra_log_var = np.append(spectra_log_var, i_log_vars)
        deltaE = np.append(deltaE, np.linspace((image.deltaE[0]+image.deltaE[units_per_bin])/2, (image.deltaE[n-1]+image.deltaE[n-units_per_bin-1])/2, n_bins))
        ddeltaE = image.ddeltaE*units_per_bin
        
        #pseudodata
        #print(n_bins, n, '\n', edges, '\n', (edges[1:]-edges[:-1])/2)
        spectra_log_mean = np.append(spectra_log_mean, 0.5 * np.ones(n_pseudo_bins))
        spectra_log_var = np.append(spectra_log_var, 0.8 * np.ones(n_pseudo_bins))
        deltaE = np.append(deltaE, dE2 + np.linspace(0,n_pseudo_bins-1, n_pseudo_bins)*ddeltaE)
        
        cluster_intensities = np.append(cluster_intensities, np.ones(n_bins+n_pseudo_bins) * image.clusters[i])
    print(spectra_log_mean.shape, spectra_log_var.shape, deltaE.shape)
    return spectra_log_mean, spectra_log_var, cluster_intensities, deltaE

def find_clusters_dE1(image, dy_dx_clusters, y_smooth_clusters):
    dE1_clusters = np.zeros(image.n_clusters, dtype=object)
    for i in range(image.n_clusters):
        dy_dx_cluster = dy_dx_clusters[i]
        y_smooth_cluster = y_smooth_clusters[i]
        dE1_cluster = np.zeros(len(y_smooth_cluster))
        for j in range(len(dy_dx_cluster)):
            dy_dx = dy_dx_cluster[j]
            y_smooth = y_smooth_cluster[j]
            dE1_cluster[j] = find_dE1(image, dy_dx, y_smooth)
        dE1_clusters[i] = dE1_cluster
        i_avg = round(np.average(dE1_clusters[i]),4)
        i_std = round(np.std(dE1_clusters[i]), 4)
        i_ci_low = np.nanpercentile(dE1_clusters[i],  16)
        i_ci_high = np.nanpercentile(dE1_clusters[i],  84)
        i_min = round(np.min(dE1_clusters[i]),4)
        print("dE1 cluster ", i, " avg: ", i_avg, ", std: ", i_ci_high-i_ci_low, ", min: ", i_min)
    return dE1_clusters
    pass

def determine_dE1(image, dE1_clusters, dy_dx_clusters = None, check_with_user =True):
    dE1_min_avg = np.average(dE1_clusters[0])
    for i in range(1,image.n_clusters):
        dE1_avg_cluster = np.average(dE1_clusters[i])
        if dE1_avg_cluster < dE1_min_avg:
            dE1_min_avg = dE1_avg_cluster
    
    if not check_with_user:
        return dE1_min_avg
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if len(colors) < image.n_clusters:
        print("thats too many clusters to effectively plot, man")
        return dE1_min_avg
        #TODO: be kinder
    der_deltaE = image.deltaE[:-1]
    plt.figure()
    for i in range(image.n_clusters):
        dE1_i_avg = np.average(dE1_clusters[i])
        dE1_i_std = np.std(dE1_clusters[i])
        #dE1_i_min = np.min(dE1_clusters[i])
        dx_dy_i_avg = np.average(dy_dx_clusters[i], axis = 0)
        dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        #dx_dy_i_min = np.min(dy_dx_clusters[i], axis = 0)
        plt.fill_between(der_deltaE, dx_dy_i_avg-dx_dy_i_std, dx_dy_i_avg+dx_dy_i_std, color = colors[i], alpha = 0.2)
        plt.axvspan(dE1_i_avg-dE1_i_std, dE1_i_avg+dE1_i_std, color = colors[i], alpha=0.1)
        plt.vlines(dE1_i_avg, -2, 1, ls = '--', color= colors[i])
        if i == 0:
            lab = "vacuum"
        else:
            lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg, color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[0,0], color = 'black')
    plt.title("derivatives of EELS per cluster, and range of first \npositive derivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("dy/dx")
    plt.legend()
    plt.xlim(dE1_min_avg/2, dE1_min_avg*3)
    plt.ylim(-3e3,2e3)
    
    
    plt.figure()
    dx_dy_0_avg = np.average(dy_dx_clusters[0], axis = 0)
    dx_dy_0_std = np.std(dy_dx_clusters[0], axis = 0)
    for i in range(1,image.n_clusters):
        dE1_i_avg = np.average(dE1_clusters[i])
        dE1_i_std = np.std(dE1_clusters[i])
        #dE1_i_min = np.min(dE1_clusters[i])
        dx_dy_i_avg = np.average(dy_dx_clusters[i], axis = 0)
        dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        #dx_dy_i_min = np.min(dy_dx_clusters[i], axis = 0)
        #plt.fill_between(image.deltaE, dx_dy_i_avg-dx_dy_i_std, dx_dy_i_avg+dx_dy_i_std, color = colors[i], alpha = 0.2)
        #plt.axvspand(dE1_i_avg-dE1_i_std, dE1_i_avg+dE1_i_std, color = colors[i], alpha=0.1)
        plt.vlines(dE1_i_avg, -2, 1, color= colors[i])
        lab = "sample cl." + str(i)
        plt.plot(der_deltaE, dx_dy_i_avg/dx_dy_0_avg, color = colors[i], label = lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]],[1,1], color = 'black')
    plt.title("ratio between derivatives of EELS per cluster and the  \nderivative of vacuum cluster, and average of first positive \nderivative of EELSs per cluster")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio dy/dx sample and dy/dx vacuum")
    plt.legend()
    plt.xlim(dE1_min_avg/2, dE1_min_avg*3)
    plt.ylim(-1,2)
    plt.show()
    print("please review the two auxillary plots on the derivatives of the EEL spectra. \n"+\
          "dE1 is the point before which the influence of the sample on the spectra is negligiable.") #TODO: check spelling
    return user_check("dE1", dE1_min_avg)


def determine_dE2_new(image, spectra_smooth_clusters, dy_dx_clusters):
    desired_ratio = 3 #TODO: think
    
    dE2_option1 = np.zeros(image.n_clusters-1)
    dE2_option2 = np.zeros(image.n_clusters-1)
    
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.figure()
    I_0_avg = np.average(spectra_smooth_clusters[0], axis = 0)
    I_0_std = np.std(spectra_smooth_clusters[0], axis = 0)
    for i in range(1,image.n_clusters):
        I_i_avg = np.average(spectra_smooth_clusters[i], axis = 0)
        I_i_std = np.std(spectra_smooth_clusters[i], axis = 0)
        
        dE2_option1[i-1] = image.deltaE[np.argmax(I_i_avg/I_0_avg)]
        plt.vlines(dE2_option1[i-1], np.min(I_i_avg/I_0_avg), np.max(I_i_avg/I_0_avg), color= colors[i])
        lab = "sample cl." + str(i)
        plt.plot(image.deltaE, I_i_avg/I_0_avg, color = colors[i], label = lab)
    plt.plot([image.deltaE[0], image.deltaE[-1]],[1,1], color = 'black')
    plt.title("ratio between average intensities of EELS per cluster and the  \naverage intensity of vacuum cluster, and max value of ratios")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio intensity sample and intensity vacuum")
    plt.legend()
    #plt.xlim(np.min(dE2_option1)/2, np*3)
    #plt.ylim(-1,2)
    plt.show()
    
    for i in range(1,image.n_clusters):
        I_i_avg = np.average(spectra_smooth_clusters[i], axis = 0)
        I_i_std = np.std(spectra_smooth_clusters[i], axis = 0)
        
        dE2_option2[i-1] = image.deltaE[np.argmax((I_i_avg/I_0_avg)>desired_ratio)]
        plt.vlines(dE2_option2[i-1], np.min(I_i_avg/I_0_avg), np.max(I_i_avg/I_0_avg), color= colors[i])
        lab = "sample cl." + str(i)
        plt.plot(image.deltaE, I_i_avg/I_0_avg, color = colors[i], label = lab)
    plt.plot([image.deltaE[0], image.deltaE[-1]],[1,1], color = 'black')
    plt.title("ratio between average intensities of EELS per cluster and the  \naverage intensity of vacuum cluster, and first crossing of " + str(desired_ratio))
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio intensity sample and intensity vacuum")
    plt.legend()
    #plt.xlim(np.min(dE2_option1)/2, np*3)
    #plt.ylim(-1,2)
    plt.show()
    
    dE2 = np.average(dE2_option1)
    
    dE2 = user_check("dE2", dE2)
    return dE2
     
def determine_dE2(image, vacuum_cluster, nbins, dE1, check_with_user=True):
    x_bins = np.linspace(image.deltaE.min(),image.deltaE.max(), nbins)
    [y_0_bins, sigma_0_bins], edges = binned_statistics(image.deltaE, vacuum_cluster, nbins, ["mean", "var"])
    ratio_0_std = np.divide(y_0_bins,sigma_0_bins)
    #ratio_0_var = np.divide(y_0_bins,np.power(sigma_0_bins,2))
    
    dE2 = np.min(x_bins[(x_bins>dE1) * (ratio_0_std <1)])
    
    if not check_with_user:
        return dE2
    
    plt.plot(x_bins, ratio_0_std)
    plt.title("I_vacuum_bins/std_vacuum_bins")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio")
    plt.plot([x_bins[0], x_bins[-1]],[1,1])
    plt.show()
    """
    plt.plot(x_bins, ratio_0_var)
    plt.title("I_vacuum_bins/var_vacuum_bins")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("ratio")
    plt.plot([x_bins[0], x_bins[-1]],[1,1])
    plt.show()
    """
    print("please review the auxillary plot on the ratio between the variance and the amplitude of "\
          +"the intensity of the vacuum EEL spectra. \n"+\
          "dE2 is the point after which the influence of the ZLP on the spectra is negligiable.") #TODO: check spelling
    return user_check("dE2", dE2)


def user_check(dE12, value):
    #TODO: opschonen?
    ans = input("Are you happy with a " + dE12 + " of " + str(round(value, 4)) + "? [y/n/wanted "+dE12+"] \n")
    if ans[0] not in["y", "n","0","1","2","3","4","5","6","7","8","9"]:
        ans = input("Please respond with either 'yes', 'no', or your wanted " + dE12 + ", otherwise assumed yes: \n")
    if ans[0] not in["y", "n","0","1","2","3","4","5","6","7","8","9"]:
        print("Stupid, assumed yes, using " + dE12 + " of " + str(round(value, 4)))
        return value
    elif ans[0] == 'y':
        print("Perfect, using " + dE12 + " of " + str(round(value, 4)))
        return value
    elif ans[0] == 'n':
        ans = input("Please input your desired " + dE12 + ": \n")
    if ans[0] not in["0","1","2","3","4","5","6","7","8","9"]:
        ans = input("Last chance, input your desired " + dE12 + ": \n")
    if ans[0] not in["0","1","2","3","4","5","6","7","8","9"]:
        print("Stupid, using old " + dE12 + " of " + str(round(value, 4)))
        return value
    else: 
        try:
            return (float(ans))
        except:
            print("input was invalid number, using original " + dE12)
            return value

#ZLP FITTING





