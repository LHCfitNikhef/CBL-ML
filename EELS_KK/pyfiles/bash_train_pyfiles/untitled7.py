#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:59:40 2021

@author: isabel
"""
import numpy as np
import sys
import os
from mpi4py import MPI
from scipy.optimize import curve_fit
from k_means_clustering import k_means


from image_class_bs import Spectral_image, smooth_1D

def median(data):
    return np.nanpercentile(data, 50)

def low(data):
    return np.nanpercentile(data, 16)

def high(data):
    return np.nanpercentile(data, 84)

def summary_distribution(data):
    return[median(data), low(data), high(data)]

def bandgap(x, amp, BG,b):
    result = np.zeros(x.shape)
    result[x<BG] = 50
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result

def find_pixel_cordinates(idx, n_y):
    y = int(idx%n_y)
    x = int((idx-y)/n_y)
    return [x,y]


path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/dE1/E1_05/"
# im = im
im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
im.cluster(5)

rank = 84
j = 80
im.load_zlp_models(path_to_models=path_to_models, n_rep = 500)
ZLPs = im.calc_zlps(rank, j, path_to_models=path_to_models, n_rep = 500)
n_model = len(im.ZLP_models)
epss, ts, S_Es, IEELSs = im.KK_pixel(rank, j)

E_bands = np.zeros(n_model)

E_cross_pix = np.empty(0)
n_cross_pix = np.zeros(n_model)
for i in range(n_model):
    IEELS = IEELSs[i]
    popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>0.5) & (im.deltaE<3.7)], IEELS[(im.deltaE>0.5) & (im.deltaE<3.7)], p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
    E_bands[i] = popt[1]
    
    crossing = np.concatenate((np.array([0]),(smooth_1D(np.real(epss[i]),50)[:-1]<0) * (smooth_1D(np.real(epss[i]),50)[1:] >=0)))
    deltaE_n = im.deltaE[im.deltaE>0]
    #deltaE_n = deltaE_n[50:-50]
    crossing_E = deltaE_n[crossing.astype('bool')]
    
    E_cross_pix = np.append(E_cross_pix, crossing_E)
    n_cross_pix[i] = len(crossing_E)

n_cross_lim = round(np.nanpercentile(n_cross_pix, 90))
# if n_cross_lim > E_cross.shape[1]:
#     E_cross_new = np.zeros((im.image_shape[1],n_cross_lim,3))
#     E_cross_new[:,E_cross.shape[1],:] = E_cross
#     E_cross = E_cross_new
#     del E_cross_new
E_cross_pix_n, r = k_means(E_cross_pix, n_cross_lim)
E_cross_pix_n = np.zeros((n_cross_lim, 3))
for i in range(n_cross_lim):
    E_cross_pix_n[i, :] = summary_distribution(E_cross_pix[r[i].astype(bool)])
    # E_cross[j,i,:] = summary_distribution(E_cross_pix[r[i].astype(bool)])


# eps[j,:,:] = summary_distribution(epss)
# t[j,:] = summary_distribution(ts)
# # E_cross[j,:] = E_cross_pix_n
# n_cross[j,:] = summary_distribution(n_cross_pix)
# E_band[j] = summary_distribution(E_bands)











