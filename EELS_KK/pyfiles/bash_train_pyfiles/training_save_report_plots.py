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
import os

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


def gen_gen_ZLP_I(image, I):
    deltaE = np.linspace(0.1,0.9, 200)
    predict_x_np = np.zeros((200,2))
    predict_x_np[:,0] = deltaE
    predict_x_np[:,1] = I

    predict_x = torch.from_numpy(predict_x_np)
    count = len(image.ZLP_models)
    ZLPs = np.zeros((count, 200)) #np.zeros((count, len_data))
        
    for k in range(count): 
        model = image.ZLP_models[k]
        with torch.no_grad():
            predictions = np.exp(model(predict_x.float()).flatten())
        ZLPs[k,:] = predictions#matching(energies, np.exp(mean_k), data)
        
    return ZLPs


def select_ZLPs(image, ZLPs):
    dE1_min = min(image.dE1[1,:])
    dE2_max = 3*max(image.dE1[1,:])
    
    
    ZLPs_c = ZLPs[:,(image.deltaE>dE1_min) & (image.deltaE<dE2_max)]
    low = np.nanpercentile(ZLPs_c, 2, axis=0)
    high = np.nanpercentile(ZLPs_c, 95, axis=0)
    
    threshold = (low[0]+high[0])/100
    
    low[low<threshold] = 0
    high[high<threshold] = threshold
    
    check = (ZLPs_c<low)|(ZLPs_c>=high)
    check = np.sum(check, axis=1)/check.shape[1]
    
    threshold = 0.01
    
    return [check<threshold]
    
    

  
    
    

#im = Spectral_image.load_data('../../data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4')

im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
# im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
# im.cluster(5)

#im=im

# path_to_models = 'dE1/E1_05'
# path_to_models = 'models/train_lau_log'
path_to_models = 'models/train_004_pooled_5_3'
path_to_models = 'models/train_004'
path_to_models = 'models/train_004_not_pooled_2'
# path_to_models = 'models/train_004_not_pooled_CI_68'
# path_to_models = 'models/train_004_not_pooled_CI_68_dE1_03'

# path_to_models = 'models/train_004__pooled_5_CI_68_dE1_04_epochs_1e6'
# path_to_models = 'models/train_004_not_pooled_CI_68_dE1_03_epochs_1e6'

# path_to_models = 'models/train_004_pooled_5_CI_68_dE1_min_04_epochs_1e6'
# path_to_models = 'models/train_004__pooled_5_CI_68_dE1_03_epochs_1e6'
# # path_to_models = 'models/train_004_pooled_5_CI_68_dE1_0_epochs_1e6'
# path_to_models = 'models/train_004_pooled_5_CI_68_dE1_03_cl0_not_epochs_1e6'
# path_to_models = 'models/train_004_pooled_5_CI_68_dE1_03_cl0_not_epochs_1e6_scale_on_pooled'
# path_to_models = 'models/train_004_pooled_5_CI_68_dE1_03_cl0_not_epochs_1e6_scale_on_pooled_clu_10'


path_to_models = 'models/dE2_3_times_dE1/train_004_pooled_5_CI_1_dE1_06_epochs_1e6_scale_on_pooled_clu_5'
#path_to_models = 'models/dE2_3_times_dE1/train_004_not_pooled_CI_1_dE1_06_epochs_1e6_scale_on_pooled_clu_5'

path_to_models = 'models/dE2_4_times_dE1/train_004_pooled_5_CI_1_dE1_06_epochs_1e6_scale_on_pooled_clu_5_2'
# path_to_models = 'models/dE2_4_times_dE1/train_004_not_pooled_CI_1_dE1_06_epochs_1e6_scale_on_pooled_clu_5_2'

path_to_models = 'models/dE2_3_times_dE1/train_004_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_5/'
path_to_models = 'models/dE2_3_times_dE1/train_004_not_pooled_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'



path_to_models = 'models/dE2_4_times_dE1/004_clu10_p5_final'

plotim = '004'
path_to_models += (path_to_models[-1]!='/')*'/'

file_name = path_to_models.split('/')[-2]
save_loc = "../../plots/Latex_overviews/" + file_name 

if plotim == 'lau':
    im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
    path_to_models = 'models/dE2_3_times_dE1/train_lau_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10'
    path_to_models = 'models/dE2_4_times_dE1/lau_clu10_p5_final'
    path_to_models = 'models/report/lau_clu10_p5_final_35dE1_06dE1/'
    path_to_models = 'models/report/lau_clu10_p5_final_4dE1_06dE1/'
    #path_to_models = 'models/report/lau_clu10_p5_final_5dE1_06dE1/'
    save_loc = "../../plots/final_report/lau/dE1_0_set25"
    try_pixels = [[17,64]]#[[7,64],[47,64],[85,64]]
    name = "sample from study Roest et al."
    xlim_times_dE1 = 6


elif plotim == '004':
    im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
    path_to_models = 'models/dE2_3_times_dE1/train_004_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
    path_to_models = '/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/models/dE2_8_times_dE1/train_004_pooled_5_3/'
    # path_to_models = 'models/dE2_4_times_dE1/004_clu10_p5_final'
    # path_to_models = 'models/report/004_clu10_p5_final_35dE1_06dE1/'
    # path_to_models = 'models/report/004_clu10_p5_final_3dE1_06dE1'

    save_loc = "../../plots/final_report/004_2"
    try_pixels = [[60,7],[47,60],[85,60]]#[[65,7],[65,45],[65,83]]
    name = "sample with WS$_2$ nanostructures"
    xlim_times_dE1 = 4


elif plotim == '003':
    im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_003.dm4')
    path_to_models = 'models/dE2_3_times_dE1/train_004_not_pooled_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
    path_to_models = 'models/dE2_4_times_dE1/004_clu10_p5_final'
    path_to_models = 'models/report/004_clu10_p5_final_35dE1_06dE1/'
    save_loc = "../../plots/final_report/003"



if not os.path.exists(save_loc):
    os.mkdir(save_loc)
    
save_loc += "/plots/"

if not os.path.exists(save_loc):
    os.mkdir(save_loc)


im.pool(5)
sig = "pooled"

im.load_ZLP_models_smefit(path_to_models=path_to_models)


#%%
if plotim == 'lau':
    im.dE1[1,0] = 2.5


"""
xlim = [np.min(im.dE1[1,:])/4, np.max(im.dE1[1,:])*xlim_times_dE1/2]
#xlim = [0.25,2.6]

# name = " 004 4x" #+ path_to_models # "Lau's sample, clustered on log" 

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
    # ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    # ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I_scales = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I_scales = " + str(I))


    
ax2.set_ylim(-100,1e3)
ax2.set_xlim(xlim)# 0,2)
ax1.legend()
ax2.legend()
plt.savefig(save_loc + 'general_predictions.pdf')
"""
#%%


#%%

#try_pixels = [[50,60], [30,70], [60,90]]

im.plot_heatmap(np.sum(im.data, axis = 2), title = r"$N_{\rm tot}$ of " + name,cmap='coolwarm_r', cbar_kws={'label': '[arb. units]'}, save_as="pixel_location")

#%%
if plotim == 'lau':
    im.dE1[1,0] = 2.5 

pixx = 30
pixy = 70
# try_pixels = [[0,65]]
# try_pixels = [[60,7]]
# name = '0.5, 10, 0.3'
for i in range(len(try_pixels)):
    [pixx, pixy] = try_pixels[i]
    dE1 = im.dE1[1,im.clustered[pixx,pixy]]
    
    r = ((im.deltaE> 0.3*dE1) & (im.deltaE < xlim_times_dE1*dE1))
    
    
    signal = im.get_pixel_signal(pixx,pixy, signal = sig)

    fig5, ax5 = plt.subplots()
    ax5.set_title("ZLP matching results at pixel " + str(i + 1) + " in " +name)#[" + str(pixx) + ","+ str(pixy) + "]" + name)
    ax5.set_xlabel("energy loss [eV]")
    ax5.plot(im.deltaE, signal, label = "signal", color = 'black')
    
    ZLPs = im.calc_gen_ZLPs(pixx,pixy, signal = sig, select_ZLPs=True)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
    ax5.plot(im.deltaE, mean, label = r"model prediction $I_{\rm ZLP}$")
    
    ZLPs = im.calc_ZLPs(pixx,pixy, signal=sig, select_ZLPs=True)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
    ax5.plot(im.deltaE, mean, label = r"matched $I_{\rm ZLP}$")
    

    ax5.fill_between(im.deltaE, signal-low, signal-high, alpha = 0.2)
    ax5.plot(im.deltaE, signal-mean, label = r"$I_{inel}$")
    
    ax5.set_ylabel("intensity [arb. units]")
    ymax = max(np.max(signal[r]-high[r]),65)
    ax5.set_ylim((-0.1*ymax, 1.9*ymax))
    ax5.set_xlim((0.3*dE1, xlim_times_dE1*dE1))
    

    ax5.legend()
    #plt.savefig(save_loc + 'predictions_summary_pix_' + str(pixx)+ '-'+ str(pixy) + '.pdf')
    plt.savefig(save_loc + 'predictions_summary_pix_' + str(i+ 1) + '.pdf')

    fig5, ax5 = plt.subplots()
    ax5.set_title("ZLP matching results at pixel " + str(i + 1) + " in " +name)#[" + str(pixx) + ","+ str(pixy) + "]" + name)
    ax5.set_xlabel("energy loss [eV]")
    ax5.plot(im.deltaE, signal, label = "signal", color = 'black')
    
    ZLPs = im.calc_gen_ZLPs(pixx,pixy, signal = sig, select_ZLPs=True)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
    ax5.plot(im.deltaE, mean, label = r"model prediction $I_{\rm ZLP}$")
    
    ZLPs = im.calc_ZLPs(pixx,pixy, signal=sig, select_ZLPs=True)
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    ax5.fill_between(im.deltaE, low, high, alpha = 0.2)
    ax5.plot(im.deltaE, mean, label = r"matched $I_{\rm ZLP}$")
    

    ax5.fill_between(im.deltaE, signal-low, signal-high, alpha = 0.2)
    ax5.plot(im.deltaE, signal-mean, label = r"$I_{inel}$")
    
    ax5.set_ylabel("intensity [arb. units]")
    
    ax5.plot([0.3*dE1,0.3*dE1], [-0.1*ymax, 1.9*ymax], color = 'black', lw = 0.75)
    ax5.plot([0.3*dE1,xlim_times_dE1*dE1], [1.9*ymax,1.9*ymax], color = 'black', lw = 0.75)
    ax5.plot([xlim_times_dE1*dE1,xlim_times_dE1*dE1], [-0.1*ymax,1.9*ymax], color = 'black', lw = 0.75)
    ax5.plot([0.3*dE1,xlim_times_dE1*dE1], [-0.1*ymax,-0.1*ymax], color = 'black', lw = 0.75)

    
    ymax = max(np.max(signal-high),150)
    ax5.set_ylim(-500,3e3)
    ax5.set_ylim((-0.1*ymax, 1.5*ymax))
    ax5.set_xlim(0)
    

    ax5.legend()
    #plt.savefig(save_loc + 'predictions_summary_pix_' + str(pixx)+ '-'+ str(pixy) + '.pdf')
    plt.savefig(save_loc + 'predictions_summary_pix_' + str(i+ 1) + '_full.pdf')
    
    # fig5, ax5 = plt.subplots()
    # ax5.set_title("random ZLP matching results at pixel[" + str(pixx) + ","+ str(pixy) + "]" + name)
    # ax5.set_xlabel("energy loss [eV]")
    # ax5.set_ylabel("intensity")
    # ax5.set_ylim(-500,3e3)
    # ax5.set_ylim(-100,0.8e3)
    # ax5.set_ylim(-200,1.5e3)
    # ax5.set_xlim(xlim)
    
    # # n_plot = 15
    # # for j in range(n_plot):#,  71, 100, 134, 169, 166,  84,  40]:#[27,  66, 155,  81, 148, 165,  64,   1]:
    # #     plt.plot(im.deltaE, ZLPs[j], color = 'black', alpha = 0.8)
    
    # ax5.plot(im.deltaE, signal, label = "signal")
    # ax5.plot(im.deltaE, mean, label = "mean")
    # ax5.fill_between(im.deltaE, low, high, color = 'orange', alpha = 0.2)
    # ax5.legend()
    
    
    fig5, ax5 = plt.subplots()
    ax5.set_title("random ZLP matching results at pixel " + str(i) + " in " +name)#[" + str(pixx) + ","+ str(pixy) + "]" + name)
    ax5.set_xlabel("energy loss [eV]")
    ax5.set_ylabel("intensity")
    ax5.set_ylim(-500,3e3)
    ax5.set_ylim(-100,0.8e3)
    ax5.set_ylim(-200,1.5e3)
    ax5.set_xlim(xlim)
    
    n_plot = len(ZLPs)
    for j in range(n_plot):
        plt.plot(im.deltaE, ZLPs[j], color = 'black', alpha = 0.1)
    
    ax5.plot(im.deltaE, signal, label = "signal")
    ax5.plot(im.deltaE, mean, label = "mean")
    ax5.fill_between(im.deltaE, low, high, color = 'orange', alpha = 0.5)
    ax5.legend()
    # plt.savefig(save_loc + 'predictions_all_pix_' + str(pixx)+ '-'+ str(pixy) + '.pdf')
    plt.savefig(save_loc + 'predictions_all_pix_' + str(i)+ '.pdf')


