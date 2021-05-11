#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:32:33 2021

@author: isabel
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from image_class_bs import Spectral_image

def median(data):
    return np.nanpercentile(data, 50, axis = 0)

def low(data):
    return np.nanpercentile(data, 16, axis = 0)

def high(data):
    return np.nanpercentile(data, 84, axis = 0)

def summary_distribution(data):
    return[median(data), low(data), high(data)]

def bandgap(x, amp, BG,b):
    result = np.zeros(x.shape)
    result[x<BG] = 1
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result

def bandgap_b(x, amp, BG):
    b = 1.5 #standard value according to ..... #TODO
    result = np.zeros(x.shape)
    result[x<BG] = 1
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result

def find_pixel_cordinates(idx, n_y):
    y = int(idx%n_y)
    x = int((idx-y)/n_y)
    return [x,y]


path_to_models = PATH + TO + 'train_004_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
path_to_dm_image = PATH + TO + 'h-ws2_eels-SI_004.dm4'

im = Spectral_image.load_data(path_to_dm_image)
name = " 004"
im.load_ZLP_models_smefit(path_to_models, name_in_path = False)





#TWO OPTIONS:
#%% FIRST OPTION
[n_x, n_y] = im.image_shape
n_models = len(im.ZLP_models)
ieels = np.zeros(np.append(im.shape, n_models))

im.pool(5)
signal = "pooled"
select_ZLPs = False

for i in range(n_x):
    for j in range(n_y):
        zlp_pix = im.calc_ZLPs(i,j, signal = signal, select_ZLPs = select_ZLPs)
        signal = im.get_signal(i,j, signal = signal)
        if select_ZLPs: n_models = len(zlp_pix)
        for k in range(n_models):
            ieels[i,j,:,k] = im.deconvolute(i,j,zlp_pix[k], signal = signal)


#%% SECCOND OPTION, you get a lot more info, if you want

im.set_n(4.1462, n_vac = 2.1759)
im.e0 = 200 #keV
im.beta = 67.2 #mrad
[n_x, n_y] = im.image_shape
n_models = len(im.ZLP_models)

im.pool(5)
signal = "pooled"
select_ZLPs = False

ieels = np.zeros(np.append(im.shape, n_models))

for i in range(n_x):
    for j in range(n_y):
        [ts, IEELSs, max_ieelss], [epss, ts_p, S_ss_p, IEELSs_p, max_ieels_p] = im.KK_pixel(i,j,  signal = "pooled", select_ZLPs=select_ZLPs)
        if select_ZLPs: n_models = len(zlp_pix)
        for k in range(n_models):
            ieels[i,j,:,k] = IEELSs_p[k,:]


