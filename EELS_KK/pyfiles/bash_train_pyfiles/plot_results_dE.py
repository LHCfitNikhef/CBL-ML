#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:19:40 2021

@author: isabel
"""
#EVALUTING dE1

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from image_class_bs import Spectral_image
from train_nn_torch_bs import train_nn_scaled, MC_reps, binned_statistics
import torch

def gen_ZLP_I(image, I):
    deltaE = np.linspace(0.1,0.9, image.l)
    predict_x_np = np.zeros((image.l,2))
    predict_x_np[:,0] = deltaE
    predict_x_np[:,1] = I

    predict_x = torch.from_numpy(predict_x_np)
    count = len(image.ZLP_models)
    ZLPs = np.zeros((count, image.l)) #np.zeros((count, len_data))
        
    for k in range(count): 
        model = image.ZLP_models[k]
        with torch.no_grad():
            predictions = np.exp(model(predict_x.float()).flatten())
        ZLPs[k,:] = predictions#matching(energies, np.exp(mean_k), data)
        
    return ZLPs

#im = Spectral_image.load_data('../../data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4')

im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_003.dm4')
# im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
# im.cluster(5)

#im=im

# path_to_models = 'dE1/E1_05'
# path_to_models = 'models/train_lau_log'
path_to_models = 'models/train_003_pooled_5_3'
# path_to_models = 'models/train_004'
 
im.load_ZLP_models_smefit(path_to_models=path_to_models)
xlim = [np.min(im.dE1[1,:])/4, np.max(im.dE1[1,:])*1.5]

name = " 004" # "Lau's sample, clustered on log" 

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_title("predictions for scaled intensities 0.1-0.9 of " + name)
ax1.set_xlabel("energy loss [eV]")
ax1.set_ylabel("intensity")
ax2.set_title("predictions for scaled intensities 0.1-0.9 of " + name)
ax2.set_xlabel("energy loss [eV]")
ax2.set_ylabel("intensity")




for I in [0.1,0.3,0.5,0.7,0.9]:
    ZLPs = gen_ZLP_I(im, I)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.average(ZLPs, axis = 0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    #[mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I_scales = " + str(I))


    
ax2.set_ylim(-100,1e3)
ax2.set_xlim(xlim)# 0,2)
ax1.legend()
ax2.legend()

#%%

fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
ax5.set_title("predictions for spectrum pixxel[50,60] of " + name)
ax5.set_xlabel("energy loss [eV]")
ax5.set_ylabel("intensity")
ax6.set_title("predictions for spectrum pixxel[50,60] of " + name)
ax6.set_xlabel("energy loss [eV]")
ax6.set_ylabel("intensity")
ax6.set_ylim(-200,1.5e3)
ax6.set_xlim(xlim)

ZLPs = im.calc_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
ax6.fill_between(im.deltaE, low, high, alpha = 0.2)
ax5.plot(im.deltaE, mean, label = "ddE = 0")
ax6.plot(im.deltaE, mean, label = "ddE = 0")




"""
#
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = 0")
ax1.set_xlabel("energy loss [eV]")
ax1.set_ylabel("intensity")
ax2.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = 0")
ax2.set_xlabel("energy loss [eV]")
ax2.set_ylabel("intensity")

fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
ax3.set_title("predictions for scaled intensity 0.5 at ddE1 = 0, -0.3 and -0.5")
ax3.set_xlabel("energy loss [eV]")
ax3.set_ylabel("intensity")
ax4.set_title("predictions for scaled intensity 0.5 at ddE1 = 0, -0.3 and -0.5")
ax4.set_xlabel("energy loss [eV]")
ax4.set_ylabel("intensity")


fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
ax5.set_title("predictions for spectrum pixxel[50,60] at ddE1 = 0, -0.3 and -0.5")
ax5.set_xlabel("energy loss [eV]")
ax5.set_ylabel("intensity")
ax6.set_title("predictions for spectrum pixxel[50,60] at ddE1 = 0, -0.3 and -0.5")
ax6.set_xlabel("energy loss [eV]")
ax6.set_ylabel("intensity")
ax6.set_ylim(-500,2e3)
ax6.set_xlim(0,2)


for I in [0.1,0.3,0.5,0.7,0.9]:
    ZLPs = gen_ZLP_I(im, I)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.average(ZLPs, axis = 0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    #[mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    
    if I == 0.5:
        ax3.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax4.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax3.plot(im.deltaE, mean, label = "ddE = 0")
        ax4.plot(im.deltaE, mean, label = "ddE = 0")
ax2.set_ylim(-500,6e3)
ax2.set_xlim(0,2)
ax1.legend()
ax2.legend()
ax4.set_ylim(-500,6e3)
ax4.set_xlim(0,2)
ax3.legend()
ax4.legend()

ZLPs = im.calc_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
ax6.fill_between(im.deltaE, low, high, alpha = 0.2)
ax5.plot(im.deltaE, mean, label = "ddE = 0")
ax6.plot(im.deltaE, mean, label = "ddE = 0")





path_to_models = 'dE1/E1_03'
im.load_ZLP_models_smefit(n_rep = 500, path_to_models=path_to_models)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = -0.3")
ax1.set_xlabel("energy loss [eV]")
ax1.set_ylabel("intensity")
ax2.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = -0.3")
ax2.set_xlabel("energy loss [eV]")
ax2.set_ylabel("intensity")




for I in [0.1,0.3,0.5,0.7,0.9]:
    ZLPs = gen_ZLP_I(im, I)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.average(ZLPs, axis = 0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    #[mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    if I == 0.5:
        ax3.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax4.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax3.plot(im.deltaE, mean, label = "ddE = -0.3")
        ax4.plot(im.deltaE, mean, label = "ddE = -0.3")
ax2.set_ylim(-500,6e3)
ax2.set_xlim(0,2)
ax1.legend()
ax2.legend()
    

ZLPs = im.calc_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
ax6.fill_between(im.deltaE, low, high, alpha = 0.2)
ax5.plot(im.deltaE, mean, label = "ddE = -0.3")
ax6.plot(im.deltaE, mean, label = "ddE = -0.3")


path_to_models = 'dE1/E1_05'
im.load_ZLP_models_smefit(n_rep = 500, path_to_models=path_to_models)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = -0.5")
ax1.set_xlabel("energy loss [eV]")
ax1.set_ylabel("intensity")
ax2.set_title("predictions for scaled intensities 0.1-0.9 at ddE1 = -0.5")
ax2.set_xlabel("energy loss [eV]")
ax2.set_ylabel("intensity")




for I in [0.1,0.3,0.5,0.7,0.9]:
    ZLPs = gen_ZLP_I(im, I)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.average(ZLPs, axis = 0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    #[mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    if I == 0.5:
        ax3.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax4.fill_between(im.deltaE, low, high, alpha = 0.2)
        ax3.plot(im.deltaE, mean, label = "ddE = -0.5")
        ax4.plot(im.deltaE, mean, label = "ddE = -0.5")
ax2.set_ylim(-500,6e3)
ax2.set_xlim(0,2)
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
"""
# path_to_models = 'dE1/E1_05'
# im.load_ZLP_models_smefit(path_to_models=path_to_models)


fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
ax5.set_title("ZLP matching results at pixel[50,60]" + name)
ax5.set_xlabel("energy loss [eV]")
ax5.set_ylabel("intensity")
ax6.set_title("ZLP matching results at pixel[50,60]" + name)
ax6.set_xlabel("energy loss [eV]")
ax6.set_ylabel("intensity")
ax6.set_ylim(-500,3e3)
ax6.set_ylim(-200,1.5e3)
ax6.set_xlim(xlim)


ZLPs = im.calc_gen_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
ax6.fill_between(im.deltaE, low, high, alpha = 0.2)
ax5.plot(im.deltaE, mean, label = "gen")
ax6.plot(im.deltaE, mean, label = "gen")

signal = im.get_pixel_signal(50,60)
#ax5.plot(im.deltaE, signal, label = "signal")
#ax6.plot(im.deltaE, signal, label = "signal")
#"""
ZLPs = im.calc_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
ax6.fill_between(im.deltaE, low, high, alpha = 0.2)
ax5.plot(im.deltaE, mean, label = "matched")
ax6.plot(im.deltaE, mean, label = "matched")

ZLPs = im.calc_ZLPs(50,60)
low = np.nanpercentile(ZLPs, 16, axis=0)
high = np.nanpercentile(ZLPs, 84, axis=0)
mean = np.average(ZLPs, axis = 0)
mean = np.nanpercentile(ZLPs, 50, axis=0)
ax5.fill_between(im.deltaE, signal-low, signal-high, alpha = 0.2)
ax6.fill_between(im.deltaE, signal-low, signal-high, alpha = 0.2)
ax5.plot(im.deltaE, signal-mean, label = "result")
ax6.plot(im.deltaE, signal-mean, label = "result")


#"""
ax5.legend()
ax6.legend()




