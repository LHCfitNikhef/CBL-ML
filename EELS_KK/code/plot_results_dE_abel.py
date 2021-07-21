#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:19:40 2021

@author: isabel
"""
# EVALUTING dE1
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from spectral_image import SpectralImage
# from train_nn_torch_bs import train_nn_scaled, MC_reps, binned_statistics
import torch
from matplotlib import rc
import sys
import os

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
rc('text', usetex=True)


# plt.rcParams.update({'font.size': 10})

def gen_ZLP_I(image, I):
    deltaE = np.linspace(0.1, 0.9, image.l)
    predict_x_np = np.zeros((image.l, 2))
    predict_x_np[:, 0] = deltaE
    predict_x_np[:, 1] = I

    predict_x = torch.from_numpy(predict_x_np)
    count = len(image.ZLP_models)
    ZLPs = np.zeros((count, image.l))

    for k in range(count):
        model = image.ZLP_models[k]
        with torch.no_grad():
            predictions = np.exp(model(predict_x.float()).flatten())
        ZLPs[k, :] = predictions

    return ZLPs


def select_ZLPs(image, ZLPs):
    dE1_min = min(image.dE1[1, :])
    dE2_max = 3 * max(image.dE1[1, :])

    ZLPs_c = ZLPs[:, (image.deltaE > dE1_min) & (image.deltaE < dE2_max)]
    low = np.nanpercentile(ZLPs_c, 2, axis=0)
    high = np.nanpercentile(ZLPs_c, 95, axis=0)

    threshold = (low[0] + high[0]) / 100

    low[low < threshold] = 0
    high[high < threshold] = threshold

    check = (ZLPs_c < low) | (ZLPs_c >= high)
    check = np.sum(check, axis=1) / check.shape[1]

    threshold = 0.01

    return [check < threshold]


path_to_image = 'C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n-dop-inse-B1_stem-eels-SI-processed_003.dm4'
#im = Spectral_image.load_data('C:/Users/abelbrokkelkam/PhD/data/dmfiles/area03-eels-SI-aligned.dm4')
im = SpectralImage.load_data(path_to_image)

path_to_models = 'C:/Users/abelbrokkelkam/PhD/data/MLdata/models/dE_n10-inse_SI-003/E1_shift09_k5/'
#path_to_models = '/data/theorie/abelbk/bash_train_pyfiles/models/dE_nf-ws2_SI-001/E1_new/'
im.load_ZLP_models_smefit(path_to_models=path_to_models, plotting=True)

im.pool(5)
im.cluster(5)
sig = "pooled"

#%% General settings

# InSe
title_specimen = r'$\rm{InSe\;}$'
save_title_specimen = "InSe"
save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_n10-inse_SI-003/E1_shift09_k5/pdfplots/"

# WS2
"""
title_specimen = r'$\rm{WS_2\;nanoflower\;}$'
save_title_specimen = 'WS2_nanoflower_flake'
save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_nf-ws2_SI-001/pdfplots/new/"
"""

#%%




#%%

fig1, ax1 = plt.subplots(dpi=200)
fig2, ax2 = plt.subplots(dpi=200)
ax1.set_title(r"$\rm{Predicted\;ZLPs\;for\;cluster\;means}$")
ax1.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
ax1.set_ylabel(r"$\rm{Intensity\;[a.u]}$")
ax2.set_title(r"$\rm{Predicted\;ZLPs\;for\;cluster\;means}$")
ax2.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
ax2.set_ylabel(r"$\rm{Intensity\;[a.u]}$")

check = True

cluster_means = im.clusters
scaled_int = [im.scale_var_log_sum_I[0] * i + im.scale_var_log_sum_I[1] for i in cluster_means]

for i, cluster_mean_scaled in enumerate(scaled_int):
    ZLPs = gen_ZLP_I(im, cluster_mean_scaled)
    if check: ZLPs = ZLPs[tuple(select_ZLPs(im, ZLPs))]
    low = np.nanpercentile(ZLPs, 16, axis=0)
    high = np.nanpercentile(ZLPs, 84, axis=0)
    mean = np.nanpercentile(ZLPs, 50, axis=0)
    # [mean, var, low, high], edges = binned_statistics(im.deltaE, ZLPs, n_bins, ["mean", "var", "low", "high"])
    ax1.fill_between(im.deltaE, low, high, alpha=0.3)
    ax2.fill_between(im.deltaE, low, high, alpha=0.3)
    label = r"$\rm{Cluster\;%d}$"%i #r"$\rm{Vacuum}$" if i == 0 else r"$\rm{Cluster\;%d}$"%i
    ax1.plot(im.deltaE, mean, label=label)
    ax2.plot(im.deltaE, mean, label=label)

ax2.set_ylim(0, 1e3)
ax2.set_xlim(0, 2)
ax1.legend()
ax2.legend()
fig1.savefig(os.path.join(save_loc, 'scaled_int.pdf'))
fig2.savefig(os.path.join(save_loc, 'scaled_int.pdf'))
#im.plot_heatmap(im.clustered, title = title_specimen + r'$\rm{-\;K=5\;cluster\;}$', 
#                cbar_kws={'label': r'$\rm{[-]\;}$'}, discrete_colormap = True,
#                xlab = r'$\rm{[nm]\;}$', ylab = r'$\rm{[nm]\;}$')


print("predictions done")

# %%

for i in np.arange(0, im.shape[1], 30):
    for j in np.arange(0, im.shape[0], 30):
        if i != 0 and j != 0:
            pixx = i
            pixy = j
            dE1 = im.dE1[1, int(im.clustered[pixy, pixx])]
            dE2 = 3 * dE1

            signal = im.get_pixel_signal(pixy, pixx, signal=sig)

            ZLPs_gen = im.calc_gen_ZLPs(pixy, pixx, signal=sig, select_ZLPs=True)
            low_gen = np.nanpercentile(ZLPs_gen, 16, axis=0)
            high_gen = np.nanpercentile(ZLPs_gen, 84, axis=0)
            mean_gen = np.nanpercentile(ZLPs_gen, 50, axis=0)

            ZLPs_match = im.calc_ZLPs(pixy, pixx, signal=sig, select_ZLPs=True)
            low_match = np.nanpercentile(ZLPs_match, 16, axis=0)
            high_match = np.nanpercentile(ZLPs_match, 84, axis=0)
            mean_match = np.nanpercentile(ZLPs_match, 50, axis=0)

            fig3, ax3 = plt.subplots(dpi=200)
            ax3.set_title(title_specimen + r"$\rm{specimen}$" + "\n" + r"$\rm{ZLP\;matching\;result\;at\;pixel[%d,%d]}$" % (pixx, pixy))
            ax3.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
            ax3.set_ylabel(r"$\rm{Intensity\;[a.u.]}$")
            ax3.set_ylim(0, 5000)
            ax3.set_xlim(0, im.deltaE[-1])

            ax3.plot(im.deltaE, signal, label=r"$\rm{Signal}$", color='black')
            
            # Plot random ZLPs
            # for k in range(300):
            #    zlp_idx = np.random.randint(0, len(ZLPs_gen))
            #    ax3.plot(im.deltaE, ZLPs_gen[zlp_idx], color= 'C0')
            # for k in range(500):
            #    zlp_idx = np.random.randint(0, len(ZLPs_match))
            #    ax3.plot(im.deltaE, ZLPs_match[zlp_idx], color= 'C1') 
            
            
            ax3.axvline(dE1, 0, 1, color='C3', linestyle='--')
            ax3.axvline(dE2, 0, 1, color='C3', linestyle='--')
            ax3.axvspan(dE1, dE2, alpha=0.1, color='C3')
            ax3.fill_between(im.deltaE, low_gen, high_gen, alpha=0.2)
            ax3.plot(im.deltaE, mean_gen, label=r"$\rm{Model\;prediction\;}$" + "$I_{ZLP}$")
            ax3.fill_between(im.deltaE, low_match, high_match, alpha=0.2)
            ax3.plot(im.deltaE, mean_match, label=r"$\rm{Matched\;}$" + "$I_{ZLP}$")
            ax3.fill_between(im.deltaE, signal - low_match, signal - high_match, alpha=0.2)
            ax3.plot(im.deltaE, signal - mean_match, label="$I_{\mathrm{inel}}$")

            ax3.legend(loc=1, frameon=False)

            plt.savefig(save_loc + save_title_specimen + '_ZLP_matching_pixel[' + str(pixx) + ',' + str(pixy) + '].pdf')

            fig4, ax4 = plt.subplots(dpi=200)
            ax4.set_title(title_specimen + r"$\rm{specimen}$" + "\n" + r"$\rm{ZLP\;matching\;result\;at\;pixel[%d,%d]}$" % (pixx, pixy))
            ax4.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
            ax4.set_ylabel(r"$\rm{Intensity\;[a.u.]}$")
            ax4.set_ylim(0, 600)
            ax4.set_xlim(0,4)

            ax4.plot(im.deltaE, signal, label=r"$\rm{Signal}$", color='black')
            
            # Plot random ZLPs
            # for k in range(300):
            #    zlp_idx = np.random.randint(0, len(ZLPs_gen))
            #    ax4.plot(im.deltaE, ZLPs_gen[zlp_idx], color= 'C0')
            # for k in range(500):
            #    zlp_idx = np.random.randint(0, len(ZLPs_match))
            #    ax4.plot(im.deltaE, ZLPs_match[zlp_idx], color= 'C1')
            
            ax4.vlines(dE1, 0, 100000, color='C3', linestyle='--')
            ax4.vlines(dE2, 0, 100000, color='C3', linestyle='--')
            ax4.axvspan(dE1, dE2, alpha=0.1, color='C3')
            ax4.fill_between(im.deltaE, low_gen, high_gen, alpha=0.2)
            ax4.plot(im.deltaE, mean_gen, label=r"$\rm{Model\;prediction\;}$" + "$I_{ZLP}$")
            ax4.fill_between(im.deltaE, low_match, high_match, alpha=0.2)
            ax4.plot(im.deltaE, mean_match, label=r"$\rm{Matched\;}$" + "$I_{ZLP}$")
            ax4.fill_between(im.deltaE, signal - low_match, signal - high_match, alpha=0.2)
            ax4.plot(im.deltaE, signal - mean_match, label="$I_{\mathrm{inel}}$")

            ax4.legend(loc=1, frameon=False)

            plt.savefig(save_loc + save_title_specimen + '_ZLP_matching_pixel[' + str(pixx) + ',' + str(pixy) + ']_zoomed.pdf')
            print("pixel[" + str(pixx) + "," + str(pixy) + "] done, dE1 = " + str(round(dE1, 4)))

# %%


# %%

"""    
path_to_models = 'dE1/train_004_ddE1_0_3'
im.train_ZLPs(n_clusters = 5, n_rep = 500, n_epochs = 100000, path_to_models = path_to_models, \
              added_dE1= 0.3, display_step = None)


path_to_models = 'dE1/train_004_ddE1_0_5'
im.train_ZLPs(n_clusters = 5, n_rep = 500, n_epochs = 100000, path_to_models = path_to_models, \
              added_dE1= 0.5, display_step = None)    
    
    
    

ZLPs = im.calc_zlps_matched(30,60,path_to_models = path_to_models)

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
