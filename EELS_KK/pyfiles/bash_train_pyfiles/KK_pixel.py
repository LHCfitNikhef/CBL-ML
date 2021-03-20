#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:01:54 2021

@author: isabel
"""
import numpy as np
import sys
from scipy.optimize import curve_fit


from image_class_bs import Spectral_image

def median(data):
    return np.nanpercentile(data, 50)

def low(data):
    return np.nanpercentile(data, 50)

def high(data):
    return np.nanpercentile(data, 50)

def summary_distribution(data):
    return[median(data), low(data), high(data)]

def bandgap(x, amp, BG,b):
    return amp * (x - BG)**(b)


im = sys.argv[1]
x = sys.argv[2]
y = sys.argv[3]

n_model = len(im.ZLP_models)

# epss = np.zeros(np.append(n_model,im.deltaE>0))
# ts = np.zeros(n_model)
# E_cross = np.zeros(n_model)
# E_cross = np.zeros(n_model)
# E_band = np.zeros(n_model)


epss, ts, S_Ss, IEELSs = im.KK_pixel(x,y)


E_bands = np.zeros(n_model)

energies = np.empty(0)
for i in range(len(n_model)):
    IEELS = IEELSs[i]
    popt, pcov = curve_fit(bandgap, im.deltaE, IEELS)
    E_bands[i] = popt[1]
    
    
    crossing = np.concatenate((np.array([0]),(epss[i,:-1]<0) * (epss[i,1:] >=0)))
    deltaE_n = im.deltaE[im.deltaE>0]
    #deltaE_n = deltaE_n[50:-50]
    crossing_E = deltaE_n[crossing.astype('bool')]
    
    energies = np.append(energies, crossing_E)

eps = summary_distribution(epss)
t = summary_distribution(ts)
E_cross = summary_distribution(E_crosss)
n_cross = summary_distribution(n_crosss)
E_band = summary_distribution(E_bands)

exit(eps, t, E_cross, n_cross, E_band)
sys.out(eps, t, E_cross, n_cross, E_band)

