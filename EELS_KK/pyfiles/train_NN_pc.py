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
import os
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
import time
from datetime import datetime
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


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
    smoothed_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters

def derivative_clusters(image, clusters):
    dx = image.ddeltaE
    der_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
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

def train_NN_pc(image, spectra, intensities):#, vacuum_in):
    
    #reset tensorflow
    tf.get_default_graph()
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
    for i  in range(len(spectra)):
        spectra[i][spectra[i]<1] = 1
    
    
    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)
    #dE1s = find_clusters_dE1(image, smooth_dy_dx, spectra_smooth)
    
    added_dE1 = 0.3
    dE1 = determine_dE1_new(image, smooth_dy_dx, spectra_smooth) - added_dE1 #dE1s, dy_dx)
    
    
    #TODO: instead of the binned statistics, just use xth value to dischart -> neh says Juan    
    times_dE1 = 8
    dE2 = times_dE1 *dE1 #determine_dE2_new(image, spectra_smooth, smooth_dy_dx)#[0], nbins, dE1)
    
    print("dE1 & dE2:", np.round(dE1,3), dE2)
    
    spectra_mean, spectra_var, cluster_intensities, deltaE = create_data(image, spectra, intensities, dE1, dE2, units_per_bin)
    
    
    
    full_x = np.vstack((deltaE,cluster_intensities)).T
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
    
    #print("Dataset is split into train subset (80%) and validation subset (20%)")
    
    
    
    
    function_train(image, full_x, full_y, full_sigma, intensities)
    
    
    
    
    
    
    
    pass

def function_train(image, full_x, full_y, full_sigma, intensities):
    
    """
    Callbacks:
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    
    """
    
    n_input = 1
    
    
    
    tf.reset_default_graph()
    tf.disable_eager_execution()
    now = datetime.now()
    x = tf.placeholder("float", [None, 2], name="x")#["float", "float"], [None, 1], name="x")
    y = tf.placeholder("float", [None, 1], name="y")
    sigma = tf.placeholder("float", [None, 1], name="sigma")
    
    if n_input == 1:
        x = tf.placeholder("float", [None, 1], name="x")
    
    predictions = make_model(x,1)
    
    #MONTE CARLO
    N_rep = 50
    N_full = len(full_y)

    full_y_reps = np.zeros(shape=(N_full, N_rep))
    for i in range(N_rep):
        full_rep = np.random.normal(0, full_sigma)
        full_y_reps[:,i] = (full_y + full_rep).reshape(N_full)
        
            
    std_reps = np.std(full_y_reps, axis=1)
    mean_reps = np.mean(full_y_reps, axis=1)
    
    print('MC pseudo data has been created for ', N_rep, ' replicas')
    
    ratio_test = 0.8
    
    predict_x = np.empty((0,2))
    for i in range(len(intensities)):
        predict_x = np.concatenate((predict_x, np.vstack((image.deltaE,np.ones(image.l)*intensities[i])).T))
    #image.deltaE #np.linspace(pred_min,pred_max,N_pred).reshape(N_pred,1)
    N_pred = image.l * len(intensities)
    predict_x = predict_x.reshape(N_pred, 2)
    if n_input ==1:
        N_pred = image.l
        predict_x = image.deltaE.reshape(N_pred,1)
        full_x = full_x[:, 0]
    
    chi_array = []
    
    cost = tf.reduce_mean(tf.square((y-predictions)/sigma), name="cost_function")
    eta = 5.5e-3
    optimizer = tf.train.RMSPropOptimizer(learning_rate=eta, decay=0.9, momentum=0.0, epsilon=1e-10).minimize(cost)
    saver = tf.train.Saver(max_to_keep=1000)
    
    #print("Start training on", '%04d'%(N_train), "and validating on",'%0.4d'%(N_test), "samples")
    
    #Nrep = 100
    map_name = 'Models'
    i=0
    while os.path.exists(map_name):
        map_name = 'Models' + str(i)
        i += 1
            
            
            
    for i in range(0,N_rep):
        
        
        
        full_y = full_y_reps[:, i].reshape(N_full,1)
        
        train_x, test_x, train_y, test_y, train_sigma, test_sigma = \
            train_test_split(full_x, full_y, full_sigma, test_size=ratio_test)
    
        #print(len(train_x))
        N_train = len(train_y)
        N_test = len(test_y)
        if n_input == 2:
            train_x, test_x = train_x.reshape(N_train,2), test_x.reshape(N_test,2)
        else:
            train_x, test_x = train_x.reshape(N_train,1), test_x.reshape(N_test,1)            
        train_y, test_y = train_y.reshape(N_train,1), test_y.reshape(N_test,1)
        train_sigma, test_sigma = train_sigma.reshape(N_train,1), test_sigma.reshape(N_test,1)
        
        
        ### Train and validate
        prev_test_cost = 0
        prev_epoch = 0
        avg_cost = 0

        array_train = []
        array_test = []

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            
            training_epochs = 35000
            display_step  = 1000

            for epoch in range(training_epochs):

                _, c = sess.run([optimizer, cost], 
                                feed_dict={
                                    x: train_x,
                                    y: train_y,
                                    sigma: train_sigma
                                })

                avg_cost = c
                
                test_cost = cost.eval({x: test_x, y: test_y, sigma: test_sigma})


                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "| Training cost=", "{:.9f}".format(avg_cost), "| Validation cost=", "{:.9f}".format(test_cost))
                    array_train.append(avg_cost)
                    array_test.append(test_cost)
                    path_to_data = map_name + '/All_models/'
                    Path(path_to_data).mkdir(parents=True, exist_ok=True)
                    saver.save(sess, path_to_data + 'my-model.ckpt', global_step=epoch , write_meta_graph=False) 

                    
                elif test_cost < prev_test_cost:
                    prev_test_cost = test_cost
                    prev_epoch = epoch

            best_iteration = np.argmin(array_test) 
            best_epoch = best_iteration * display_step
            best_model = map_name + '/All_models/my-model.ckpt-%(s)s' % {'s': best_epoch}

            print("Optimization %(i)s Finished! Best model after epoch %(s)s" % {'i': i, 's': best_epoch})
            


            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            d_string = now.strftime("%d.%m.%Y")
            t_string = now.strftime("%H:%M:%S")
            
            saver.restore(sess, best_model)
            path_to_data = map_name + '/Best_models/%(s)s/'  % {'s': d_string}
            Path(path_to_data).mkdir(parents=True, exist_ok=True)
            saver.save(sess, path_to_data + 'best_model_%(i)s' %{'i': i})


            predictions_values = sess.run(predictions, 
                                feed_dict={
                                    x: train_x,
                                    y: train_y 
                                }) 


            extrapolation = sess.run(predictions,
                                feed_dict={
                                    x: predict_x
                                })
            

        sess.close()
        

        nownow = datetime.now()
        print("time elapsed", nownow-now)

        d = array_train
        e = array_test
        
    
    
        path_to_data = map_name + '/Results/%(date)s/'% {"date": d_string} 
        Path(path_to_data).mkdir(parents=True, exist_ok=True)
        
        #np.savetxt(path_to_data + 'Predictions_%(k)s.csv' % {"k": i}, list(zip(a,b,c)),  delimiter=',', fmt='%f')
        np.savetxt(path_to_data + 'Cost_%(k)s.csv' % {"k": i}, list(zip(d,e)),  delimiter=',',fmt='%f')
        #np.savetxt(path_to_data + 'Extrapolation_%(k)s.csv' % {"k":i}, list(zip(k, l)),  delimiter=',', fmt='%f')




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
    

def determine_dE1_new(image, dy_dx_clusters, y_smooth_clusters, check_with_user = False):
    dy_dx_avg = np.zeros((len(y_smooth_clusters), image.l-1))
    dE1_clusters = np.zeros(len(y_smooth_clusters))
    for i in range(len(y_smooth_clusters)):
        dy_dx_avg[i,:] = np.average(dy_dx_clusters[i], axis=0)
        y_smooth_cluster_avg = np.average(y_smooth_clusters[i], axis=0)
        dE1_clusters[i] = find_dE1(image, dy_dx_avg[i,:], y_smooth_cluster_avg)
        
    if not check_with_user:
        return dE1_clusters
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if len(colors) < len(y_smooth_clusters):
        print("thats too many clusters to effectively plot, man")
        return dE1_clusters
        #TODO: be kinder
    der_deltaE = image.deltaE[:-1]
    plt.figure()
    for i in range(len(y_smooth_clusters)):
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
    for i in range(1,len(y_smooth_clusters)):
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
    
    for i in range(len(y_smooth_clusters)):
        name = "sample cluster " + str(i)
        dE1_clusters[i] = user_check("dE1 of " + name, dE1_clusters[i])
    return dE1_clusters


def create_data(image, spectra_clusters, intensities, dE1, dE2, units_per_bin):
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
    for i in range(len(spectra_clusters)):
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
        
        cluster_intensities = np.append(cluster_intensities, np.ones(n_bins+n_pseudo_bins) * intensities[i])
    print(spectra_log_mean.shape, spectra_log_var.shape, deltaE.shape)
    return spectra_log_mean, spectra_log_var, cluster_intensities, deltaE






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




#BOOTSTRAPPING
#overbodig
def rand_ints(shape, n_max = None):
    if n_max is None:
        n_max = np.product(shape)
    
    rand = np.random.rand(shape)
    randint = np.floor(rand * n_max).astype(int)
    return randint


def bootstrap(values, n_b = 500):
    rand = np.random.rand(n_b)
    randint = np.floor(rand * len(values)).astype(int)
    return values[randint]
    





