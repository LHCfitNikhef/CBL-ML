#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:19:40 2021

@author: isabel
"""
#EVALUTING dE1

import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from spectral_image import SpectralImage
#from train_nn_torch_bs import train_nn_scaled, MC_reps, binned_statistics
import torch

plt.rcParams.update({'font.size': 12})

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
    
    return [check < threshold]
    
#im = Spectral_image.load_data('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n-dop-inse-B1_stem-eels-SI-processed_003.dm4')
#im = Spectral_image.load_data('C:/Users/abelbrokkelkam/PhD/data/dmfiles/h-ws2_eels-SI_004.dm4')
im = SpectralImage.load_data('C:/Users/abelbrokkelkam/PhD/data/dmfiles/area03-eels-SI-aligned.dm4')
# im.cluster(5)
#im=im
#path_to_models = 'C:/Users/abelbrokkelkam/PhD/data/MLdata/models/dE_n10-inse_SI-003/E1_09/'
#path_to_models = 'C:/Users/abelbrokkelkam/PhD/data/MLdata/models/dE_h-ws2_SI-004/E1_05/'
path_to_models = 'C:/Users/abelbrokkelkam/PhD/data/MLdata/models/dE_nf-ws2_SI-001/E1_new/'



im.pool(5)
im.cluster(5)
sig = "pooled"
title_specimen = 'WS2 nanoflower'#'InSe'
save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_nf-ws2_SI-001/pdfplots/E1_new"
#save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_n10-inse_SI-003/pdfplots/"
im.load_ZLP_models_smefit(path_to_models=path_to_models)

#%%
"""
fig1, ax1 = plt.subplots(dpi=200)
fig2, ax2 = plt.subplots(dpi=200)
ax1.set_title("Predictions for scaled intensities 0.1-0.9 of ")
ax1.set_xlabel("Energy loss [eV]")
ax1.set_ylabel("Intensity [a.u]")
ax2.set_title("predictions for scaled intensities 0.1-0.9 of ")
ax2.set_xlabel("Energy loss [eV]")
ax2.set_ylabel("Intensity [a.u]")

check = True

for I in [0.1,0.3,0.5,0.7,0.9]:
    ZLPs = gen_ZLP_I(im, I)
    if check: ZLPs = ZLPs[tuple(select_ZLPs(im, ZLPs))]
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    #[mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax2.fill_between(im.deltaE, low, high, alpha = 0.3)
    ax1.plot(im.deltaE, mean, label = "I$_{scales}$ = " + str(I))
    ax2.plot(im.deltaE, mean, label = "I$_{scales}$ = " + str(I))
 
ax2.set_ylim(0,1e3)
ax2.set_xlim(0.4,6)
ax1.legend()
ax2.legend()
print("predictions done")
"""
#%%

for i in np.arange(0, 31,30):
    for j in np.arange(0, 31,30):
        if i != 0 and j != 0:
            pixx = i
            pixy = j
            dE1 = im.dE1[1, int(im.clustered[pixy,pixx])]
            
            signal = im.get_pixel_signal(pixy, pixx, signal = sig)
            
            ZLPs_gen = im.calc_gen_ZLPs(pixy, pixx, signal = sig, select_ZLPs=False)
            #if check:
                #select = select_ZLPs(im, ZLPs_gen)
                #ZLPs_gen = ZLPs_gen[tuple(select)]
                
            low_gen = np.nanpercentile(ZLPs_gen, 16, axis=0)
            high_gen = np.nanpercentile(ZLPs_gen, 84, axis=0)
            mean_gen = np.nanpercentile(ZLPs_gen, 50, axis=0)
                
            ZLPs_match = im.calc_ZLPs(pixy, pixx, signal = sig, select_ZLPs=False)
            #if check: 
                #select = select_ZLPs(im, ZLPs_match)
                #ZLPs_match = ZLPs_match[tuple(select)]
            
            low_match = np.nanpercentile(ZLPs_match, 16, axis=0)
            high_match = np.nanpercentile(ZLPs_match, 84, axis=0)
            mean_match = np.nanpercentile(ZLPs_match, 50, axis=0)

            fig3, ax3 = plt.subplots(dpi=200)
            ax3.set_title(title_specimen + " specimen \nZLP matching result at pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax3.set_xlabel("Energy loss [eV]")
            ax3.set_ylabel("Intensity [a.u.]")
            ax3.set_ylim(0, 50000)
            ax3.set_xlim(0, 40)
            
            ax3.plot(im.deltaE, signal, label = "Signal", color='black')
            for k in range(100):
                zlp_idx = np.random.randint(0, len(ZLPs_gen))
                ax3.plot(im.deltaE, ZLPs_gen[zlp_idx], color= 'C0') 
            #for k in range(500):
            #    zlp_idx = np.random.randint(0, len(ZLPs_match))
            #    ax3.plot(im.deltaE, ZLPs_match[zlp_idx], color= 'C1') 
            
            #ax3.fill_between(im.deltaE, low_gen, high_gen, alpha = 0.2)
            #ax3.plot(im.deltaE, mean_gen, label = "Model prediction $I_{ZLP}$")
            #ax3.fill_between(im.deltaE, low_match, high_match, alpha = 0.2)
            #ax3.plot(im.deltaE, mean_match, label = "Matched $I_{ZLP}$")
            #ax3.fill_between(im.deltaE, signal - low_match, signal - high_match, alpha = 0.2)
            #ax3.plot(im.deltaE, signal - mean_match, label = "$I_{inel}$")
            
            ax3.legend(loc=1)
            
            plt.savefig(save_loc + title_specimen + '_ZLP_matching_pixel[' + str(pixx) + ','+ str(pixy) + '].pdf')
            
            fig4, ax4 = plt.subplots(dpi=200)
            ax4.set_title(title_specimen + " specimen \nZLP matching result at pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax4.set_xlabel("Energy loss [eV]")
            ax4.set_ylabel("Intensity [a.u.]")
            ax4.set_ylim(0,600)
            ax4.set_xlim(0.4,7)
            
            ax4.plot(im.deltaE, signal, label = "Signal", color='black')
            for k in range(100):
                zlp_idx = np.random.randint(0, len(ZLPs_gen))
                ax4.plot(im.deltaE, ZLPs_gen[zlp_idx], color= 'C0') 
            #for k in range(500):
            #    zlp_idx = np.random.randint(0, len(ZLPs_match))
            #    ax4.plot(im.deltaE, ZLPs_match[zlp_idx], color= 'C1') 
                
            #ax4.fill_between(im.deltaE, low_gen, high_gen, alpha = 0.2)
            #ax4.plot(im.deltaE, mean_gen, label = "Model prediction $I_{ZLP}$")
            #ax4.fill_between(im.deltaE, low_match, high_match, alpha = 0.2)
            #ax4.plot(im.deltaE, mean_match, label = "Matched $I_{ZLP}$")
            #ax4.fill_between(im.deltaE, signal - low_match, signal - high_match, alpha = 0.2)
            #ax4.plot(im.deltaE, signal - mean_match, label = "$I_{inel}$")
            ax4.legend(loc=1)
            
            plt.savefig(save_loc + title_specimen + '_ZLP_matching_pixel[' + str(pixx) + ','+ str(pixy) + ']_zoomed.pdf')

            """
            #Plotting random ZLPs
            fig5, ax5 = plt.subplots()
            ax5.set_title("random ZLP matching results at pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax5.set_xlabel("energy loss [eV]")
            ax5.set_ylabel("intensity")
            ax5.set_ylim(0,600)
            ax5.set_xlim(xlim)
        
            n_plot = 15
            for j in range(n_plot):
                plt.plot(im.deltaE, ZLPs[j], color = 'black', alpha = 0.8)
        
            ax5.plot(im.deltaE, signal, label = "signal")
            ax5.plot(im.deltaE, mean, label = "mean")
            ax5.fill_between(im.deltaE, low, high, color = 'orange', alpha = 0.2)
            ax5.legend()
        
            fig6, ax6 = plt.subplots()
            ax6.set_title("random ZLP matching results at pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax6.set_xlabel("energy loss [eV]")
            ax6.set_ylabel("intensity")
            ax6.set_ylim(0,600)
            ax6.set_xlim(xlim)
        
            n_plot = len(ZLPs)
            for k in range(n_plot):
                plt.plot(im.deltaE, ZLPs[k], color = 'black', alpha = 0.2)
        
            ax6.plot(im.deltaE, signal, label = "signal")
            ax6.plot(im.deltaE, mean, label = "mean")
            ax6.fill_between(im.deltaE, low, high, color = 'orange', alpha = 0.2)
            ax6.legend()
            """
            print("pixel[" + str(pixx) + ","+ str(pixy) + "] done, dE1 = " + str(round(dE1,4)))

#%%


#%%

"""    
path_to_models = 'dE1/train_004_ddE1_0_3'
im.train_ZLPs(n_clusters = 5, n_rep = 500, n_epochs = 100000, path_to_models = path_to_models, \
              added_dE1= 0.3, display_step = None)


path_to_models = 'dE1/train_004_ddE1_0_5'
im.train_ZLPs(n_clusters = 5, n_rep = 500, n_epochs = 100000, path_to_models = path_to_models, \
              added_dE1= 0.5, display_step = None)    
    
    
    

ZLPs = im.calc_ZLPs(30,60,path_to_models = path_to_models)

np.savetxt("004_zlps_I_scaled_5_pix_30_60.txt", ZLPs)

plt.figure()
plt.ylim(0,2e3)
for i in range(len(ZLPs)):
    plt.plot(ZLPs[i])
plt.savefig("004_zlps_I_scaled_5_pix_30_60.pdf")
"""



"""
#train_nn_scaled(im, path_to_model = "train_004", lr = 1e-3, n_epochs=30000)
n_bins = int(im.l/4)
spectra = im.get_cluster_spectra()
for cluster in range(im.n_clusters):
    [mean, var, low, high], edges = binned_statistics(im.deltaE, spectra[cluster], n_bins, ["mean", "var", "low", "high"])
    plt.figure()
    plt.title("distribution of cluster " + str(cluster) + " and three random spectra")
    plt.fill_between((edges[:-1]+edges[1:])/2, low, high, alpha = 0.5)
    plt.plot((edges[:-1]+edges[1:])/2, mean, label = "mean")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    plt.xlim(-0.2, 0.5)
    for i in range(3):
        idx = int(len(spectra[cluster])*(0.3*(i+1)))
        plt.plot(im.deltaE, spectra[cluster][idx]) 
    
    MC_rep = MC_reps(mean, var, 3)
    plt.figure()
    plt.title("distribution of cluster " + str(cluster) + " and three MC replicas")
    plt.fill_between((edges[:-1]+edges[1:])/2, low, high, alpha = 0.5)
    plt.plot((edges[:-1]+edges[1:])/2, mean, label = "mean")
    plt.xlabel("energy loss [eV]")
    plt.ylabel("intensity")
    plt.xlim(-0.2, 0.5)
    for i in range(3):
        plt.plot((edges[:-1]+edges[1:])/2, MC_rep[:,i]) 
"""    