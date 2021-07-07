#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:13:47 2021

@author: isabel
"""
import numpy as np
#import sys
#import os
#import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
#import seaborn as sns
from scipy.optimize import curve_fit
from image_class_bs import Spectral_image
import torch


#path_to_results = "C:/Users/abelbrokkelkam/PhD/data/MLdata/results/dE_n10-inse_SI-003/image_KK.pkl"
path_to_results = "C:/Users/abelbrokkelkam/PhD/data/MLdata/results/dE_nf-ws2_SI-001/image_KK_4.pkl"
im = Spectral_image.load_Spectral_image(path_to_results)

path_to_models = 'C:/Users/abelbrokkelkam/PhD/data/MLdata/models/dE_nf-ws2_SI-001/E1_07/'
im.load_ZLP_models_smefit(path_to_models=path_to_models)
im.pool(5)
#%% Settings for all heatmaps


# InSe general settings
"""
cmap="coolwarm" 
npix_xtick=24.5
npix_ytick=12.25
sig_ticks = 3
scale_ticks = 1000
tick_int = True
thicknesslimit = np.nanpercentile(im.t[im.clustered == 0],0)
mask = im.t[:,:,0] < thicknesslimit
cb_scale=0.4
title_specimen = 'InSe'
save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_n10-inse_SI-003/pdfplots/new/"

im.e0 = 200									# keV
im.beta = 21.3								# mrad
im.set_n(3.0)								# refractive index, InSe no background
"""
# WS2 SI general settings
cmap="coolwarm" 
npix_xtick=26.25
npix_ytick=26.25
sig_ticks = 3
scale_ticks = 1000
tick_int = True
thicknesslimit = np.nanpercentile(im.t[im.clustered == 2],99)
mask = ((np.isnan([im.t[:,:,0]])[0]) | (im.t[:,:,0] > thicknesslimit))
cb_scale=0.85
title_specimen = 'WS$_2$ nanoflower flake'
save_title_specimen = 'WS2_nanoflower_flake'
save_loc = "C:/Users/abelbrokkelkam/PhD/data/MLdata/plots/dE_nf-ws2_SI-001/pdfplots/new/"

im.e0       = 200                           # keV
im.dE1[1,0] = 2.5                           # Why do we fix the first dE1?
im.beta     = 67.2                          # mrad
#im.set_n(4.1462, n_background = 2.1759)    # refractive index, WS2 triangles with SiN substrate as background
im.set_n(4.1462, n_background = 1)          # refractive index, WS2 nanoflower with vacuum as background, note that calculations on SiN may get weird

def round_to_nearest(value, base=5):
    return base * round(float(value) / base)

#%% CLUSTER
im.cluster(5)
im.plot_heatmap(im.clustered, title = title_specimen + " - $K=5$ clusters", 
                cbar_kws={'label': '[-]','shrink':cb_scale}, discrete_colormap = True,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Clustered')

#%% THICKNESS
im.plot_heatmap(im.t[:,:,0], title = title_specimen + " - Thickness", 
                cbar_kws={'label': '[nm]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin=0,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Thickness')

im.plot_heatmap(im.t[:,:,0], title = title_specimen + " - Thickness", 
                cbar_kws={'label': '[nm]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap, 
                mask = mask, vmin = 0, vmax = 60, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Thickness_capped')

im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/(2*im.t[:,:,0]), title = title_specimen + " - Relative error thickness", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin = 0,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Thickness_Error')

im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/(2*im.t[:,:,0]), title = title_specimen + " - Relative error thickness", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin = 0, vmax = 0.03,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Thickness_Error_capped')
"""
im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/(im.t[:,:,0]), title = title_specimen + " - Relative broadness CI Thickness", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin = 0, vmax = 0.02,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Thickness_CI')
"""
#%% THICKNESS DISCRETIZED
"""
mask_t = (mask | ((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0] >= 1))
size_t_bins = np.nanpercentile((im.t[:,:,2]-im.t[:,:,1])[~mask_t],100)/0.3
t_round  = np.round(im.t[:,:,0]/size_t_bins) * size_t_bins
im.plot_heatmap(t_round, title = "Indium Selenide Sample \nThickness discretized", cbar_kws={'label':'Thickness [nm]','shrink':0.4}, xlab = "[nm]", ylab = "[nm]", vmax = 300, vmin = 50, cmap = cmap, mask = mask_t, color_bin_size = size_t_bins, discrete_colormap = True, sig=3, n_xticks=8, n_yticks=6)
"""

#%% MAX IEELS

im.plot_heatmap(im.max_ieels[:,:,0], title = title_specimen + " - Maximum IEELS", 
                cbar_kws={'label': 'Energy loss [eV]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Max_IEELS')

im.plot_heatmap((im.max_ieels[:,:,2]-im.max_ieels[:,:,1])/(2*im.max_ieels[:,:,0]), title = title_specimen + " - Relative error Maximum IEELS", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 0.001, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Max_IEELS_Error')
"""
im.plot_heatmap((im.max_ieels[:,:,2]-im.max_ieels[:,:,1])/(im.max_ieels[:,:,0]), title = title_specimen + " - Relative broadness CI Maximum IEELS", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 0.001, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Max_IEELS_CI')
"""
#%% MAX IEELS DISCRETIZED

mask_max_ieels = (mask | ((im.max_ieels[:,:,2]-im.max_ieels[:,:,1])/im.max_ieels[:,:,0] >= 1))
size_ieels_bins = round_to_nearest(np.nanpercentile((im.max_ieels[:,:,0])[~mask_max_ieels],50)/2,0.5)
ieels_round  = np.round(im.max_ieels[:,:,0]/size_ieels_bins) * size_ieels_bins
im.plot_heatmap(ieels_round, title = title_specimen + " - Maximum IEELS", 
                cbar_kws={'label':'Energy loss [eV]', 'shrink':cb_scale}, color_bin_size = size_ieels_bins, discrete_colormap = True,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin = 21, vmax = 26,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Max_IEELS_Discretized')


#%% THICKNESS CROSSSECTION

fig1, ax1 = plt.subplots(dpi=200)
ax1.set_title(title_specimen + " - Thickness cross section y-axis")
ax1.set_xlabel("x-axis [nm]")
ax1.set_ylabel("Thickness [nm]")
for i in np.arange(5,len(im.y_axis),5):
    row = i
    colors = cm.coolwarm(np.linspace(0,1,len(im.t[0,:,0])))
    ax1.set_prop_cycle(color=colors)
    for j in range(len(im.t[row,:,0]) - 1):
        ax1.plot(im.x_axis[j:j + 2], im.t[row,:,0][j:j + 2])
        ax1.fill_between(im.x_axis[j:j + 2], im.t[row,:,2][j:j + 2], im.t[row,:,1][j:j + 2], alpha = 0.3)
    
    
    #ax1.plot(im.x_axis, im.t[row,:,0], label = "Row " + str(row))
    #ax1.fill_between(im.x_axis, im.t[row,:,2], im.t[row,:,1], alpha = 0.3)
ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x*1000))))
#ax1.legend(loc=2)


fig2, ax2 = plt.subplots(dpi=200)
ax2.set_title(title_specimen + " - Thickness cross section x-axis")
ax2.set_xlabel("y-axis [nm]")
ax2.set_ylabel("Thickness[nm]")
for i in np.arange(5,len(im.x_axis),5):
    column = i
    colors = cm.coolwarm(np.linspace(0,1,len(im.t[0,:,0])))
    ax2.plot(im.y_axis, im.t[:,column,0], color = colors[i])
    #ax2.plot(im.y_axis, im.t[:,column,0], label = "Column " + str(column), color = colors[i])
    ax2.fill_between(im.y_axis, im.t[:,column,2], im.t[:,column,1], alpha = 0.3, color = colors[i])
ax2.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x*1000))))
#ax2.legend(loc=2)


#%% NUM CROSSINGS

im.n_cross = np.round(im.n_cross)

mask_cross = (mask | (im.n_cross[:,:,0] == 0))

im.plot_heatmap(im.n_cross[:,:,0], title = title_specimen + " - Crossings $\u03B5_{1}$", 
                cbar_kws={'label': 'nr. crossings','shrink':cb_scale}, discrete_colormap = True,
                xlab = "[nm]",ylab = "[nm]", cmap = cmap,
                mask = mask_cross,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Crossings')

im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1]), title = title_specimen + " - Relative broadness CI Crossings $\u03B5_{1}$", 
                cbar_kws={'label': 'Nr. crossings','shrink':cb_scale}, discrete_colormap = True, 
                xlab = "[nm]",ylab = "[nm]", cmap = cmap,
                mask = mask_cross,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Crossings_CI')

#mask_cross = (mask | (im.n_cross[:,:,2] == 0))
#im.plot_heatmap(im.n_cross[:,:,0], title = "Indium Selenide Sample \nCrossings $\u03B5_{1}$ \nAlternative mask", cbar_kws={'label': 'nr. crossings','shrink':0.4}, cmap = cmap, mask = mask_cross, discrete_colormap = True, n_xticks=8, n_yticks=6)
#im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1]), title = "Indium Selenide Sample \nRelative broadness CI Crossings $\u03B5_{1}$ \nAlternative mask", cbar_kws={'label': 'nr. crossings','shrink':0.4}, cmap = cmap, mask = mask_cross, discrete_colormap = True, n_xticks=8, n_yticks=6)


#%% ENERGY AT FIRST CROSSINGS

first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) > 0:
                first_crossings[i,j,:] = im.E_cross[i,j][0,:]
                first_crossings_CI[i,j] = (im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1])/(2*im.E_cross[i,j][0,0])
        
mask_cross = (mask | (im.n_cross[:,:,0] == 0))        
im.plot_heatmap(first_crossings[:,:,0], title = title_specimen + " - Energy first crossings $\u03B5_{1}$", 
                cbar_kws={'label': 'Energy [eV]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_cross, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Crossings')

im.plot_heatmap(first_crossings_CI, title = title_specimen + " - Relative error energy first crossings $\u03B5_{1}$", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_cross, vmax = 0.2,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_First_Crossings_CI')
"""
im.plot_heatmap(first_crossings_CI, title = title_specimen + " - Relative broadness CI energy first crossings $\u03B5_{1}$", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_cross, vmax = 0.01,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_First_Crossings_CI')
"""
#mask_cross = (mask | (im.n_cross[:,:,2] == 0))        
#im.plot_heatmap(first_crossings[:,:,0], title = "Indium Selenide Sample \nenergy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1) \nAlternative mask", cbar_kws={'label': 'energy [eV]','shrink':0.4}, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)
#im.plot_heatmap(first_crossings_CI, title = "Indium Selenide Sample \nrelative broadness CI energy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1) \nAlternative mask", cbar_kws={'label': 'ratio [-]','shrink':0.4}, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)
#im.plot_heatmap(first_crossings_CI, title = "Indium Selenide Sample \nrelative broadness CI energy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1), capped at 0.005 \nAlternative mask", cbar_kws={'label': 'ratio [-]','shrink':0.4}, vmax = 0.005, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)

#%% CROSSINGS AT MAX IEELS

mask_max_cross = (mask | (im.n_cross[:,:,0] == 0) | (first_crossings[:,:,0] < 20) | (first_crossings[:,:,0] > 25) )        
im.plot_heatmap(first_crossings[:,:,0], title = title_specimen + " - Energy crossings $\u03B5_{1}$ IEELS max", 
                cbar_kws={'label': 'Energy [eV]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_max_cross, vmin = 21, vmax = 25, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Max_Crossings')

im.plot_heatmap(first_crossings_CI, title = title_specimen + " - Relative error Energy crossings $\u03B5_{1}$ IEELS max", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_max_cross, vmax = 0.2,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Max_Crossings_CI')
"""
im.plot_heatmap(first_crossings_CI, title = title_specimen + " - Relative broadness CI Energy crossings $\u03B5_{1}$ IEELS max", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_cross, vmax = 0.01,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Max_Crossings_CI')
"""
#mask_cross = (mask | (im.n_cross[:,:,2] == 0))        
#im.plot_heatmap(first_crossings[:,:,0], title = "Indium Selenide Sample \nenergy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1) \nAlternative mask", cbar_kws={'label': 'energy [eV]','shrink':0.4}, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)
#im.plot_heatmap(first_crossings_CI, title = "Indium Selenide Sample \nrelative broadness CI energy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1) \nAlternative mask", cbar_kws={'label': 'ratio [-]','shrink':0.4}, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)
#im.plot_heatmap(first_crossings_CI, title = "Indium Selenide Sample \nrelative broadness CI energy first crossings $\u03B5_{1}$ \n(for chance at least 1 crossing > 0.1), capped at 0.005 \nAlternative mask", cbar_kws={'label': 'ratio [-]','shrink':0.4}, vmax = 0.005, cmap = cmap, mask = mask_cross, n_xticks=8, n_yticks=6)


#%% ENERGY CROSSINGS ??
"""
first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if im.n_cross[i,j,0] > 1:
            if len(im.n_cross[i,j]) > 0:
                first_crossings[i,j,:] = im.n_cross[i,j][0,:]
                first_crossings_CI[i,j] = im.n_cross[i,j][0,2]-im.n_cross[i,j][0,1]
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function\n(for chance at least 1 crossing > 0.5)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.5)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap)
"""

#%% ENERGY CROSSING DISCRETIZED

mask_E_cross = (mask | (im.n_cross[:,:,0] == 0))
size_E_cross_bins = round_to_nearest(np.nanpercentile((first_crossings[:,:,2]-first_crossings[:,:,1])[~mask_E_cross],50)/0.1,1.0)
E_cross_round  = np.round(first_crossings[:,:,0]/size_E_cross_bins) * size_E_cross_bins
im.plot_heatmap(E_cross_round, title = title_specimen + " - Energy first crossing $\u03B5_{1}$", 
                cbar_kws={'label': 'Energy [eV]','shrink':cb_scale}, color_bin_size = size_E_cross_bins, discrete_colormap = True, sig_cbar = 2,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_cross,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Crossings_Discretized')

mask_E_cross = (mask | (im.n_cross[:,:,0] == 0))
size_E_cross_bins = round_to_nearest(np.nanpercentile((first_crossings[:,:,2]-first_crossings[:,:,1])[~mask_E_cross],50)/0.5,0.5)
E_cross_round  = np.round(first_crossings[:,:,0]/size_E_cross_bins) * size_E_cross_bins
im.plot_heatmap(E_cross_round, title = title_specimen + " - Energy Max crossing $\u03B5_{1}$", 
                cbar_kws={'label': 'Energy [eV]','shrink':cb_scale}, color_bin_size = size_E_cross_bins, discrete_colormap = True, sig_cbar = 2,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask_max_cross,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Energy_Max_Crossings_Discretized')


#mask_E_cross = (mask | (im.n_cross[:,:,2] == 0))
#size_E_cross_bins = np.nanpercentile((first_crossings[:,:,2]-first_crossings[:,:,1])[~mask_cross],50)/0.05
#E_cross_round  = np.round(first_crossings[:,:,0]/size_E_bins) * size_E_bins
#im.plot_heatmap(E_cross_round, title = "Indium Selenide Sample \nenergy first crossing $\u03B5_{1}$ discretized \n(for chance at least 1 crossing > 0.1) \nAlternative mask", cbar_kws={'label': 'energy [eV]','shrink':0.4}, cmap = cmap, mask = mask_E_cross, color_bin_size = size_E_cross_bins, discrete_colormap = True, sig=3, n_xticks=8, n_yticks=6)

#%% DIELECTRIC FUNCTION INDIVIDUAL PIXELS

for i in np.arange(0, len(im.x_axis), 30):
    for j in np.arange(0, len(im.y_axis), 30):
        if i != 0 and j != 0:
            pixx=i
            pixy=j
            
            epsilon1 = im.eps[pixy,pixx,0].real
            epsilon2 = im.eps[pixy,pixx,0].imag
            
            fig1, ax1 = plt.subplots(dpi=200)
            ax1.plot(im.deltaE[(len(im.deltaE)-len(epsilon1)):], epsilon1, label = "$\u03B5_{1}$")
            ax1.plot(im.deltaE[(len(im.deltaE)-len(epsilon2)):], epsilon2, label = "$\u03B5_{2}$")
            ax1.axhline(0, color='black')
            ax1.set_title(title_specimen + " - Dielectric function pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax1.set_xlabel("Energy loss [eV]")
            ax1.set_ylabel("Dielectric function [F/m]")
            ax1.set_ylim(-0.2,5)
            ax1.legend()
            """
            fig2, ax2 = plt.subplots()
            ax2.set_title("InSe specimen \nBandgap fit pixel[" + str(pixx) + ","+ str(pixy) + "]")
            ax2.set_xlabel("energy loss [eV]")
            ax2.set_ylabel("intensity")
            ax2.set_ylim(0,500)
            ax2.set_xlim(-0,3)
            ax2.fill_betweenx([-100,1000], x1 = range1, x2 = range2, color='r', alpha = 0.2)
            ax2.fill_between(im.deltaE, p_low, p_high, alpha = 0.2)
            ax2.plot(im.deltaE, p_ieels, label = "Spectrum")
            ax2.plot(im.deltaE, bandgap_test(im.deltaE,popt[0],popt[1],popt[2]), label = "Fit")
            ax2.legend(loc=2)
            """
            plt.savefig(save_loc + save_title_specimen + '_Dielectric_function_pixel[' + str(pixx) + ','+ str(pixy) + '].pdf')
        
#%% BANDGAP
im.plot_heatmap(im.E_band[:,:,0], title = title_specimen + " - Bandgap energy", 
                cbar_kws={'label':'Energy [eV]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap')

im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/(2*im.E_band[:,:,0]), title = title_specimen + " - Relative error Bandgap energy", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap, 
                mask = mask,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_Error')

im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/(2*im.E_band[:,:,0]), title = title_specimen + " - Relative error Bandgap energy", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap, 
                mask = mask, vmax=0.2,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_Error_capped')
"""
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/(im.E_band[:,:,0]), title = title_specimen + " - Relative broadness CI Bandgap energy", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap, 
                mask = mask,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_CI')

im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/(im.E_band[:,:,0]), title = title_specimen + " - Relative broadness CI Bandgap energy, \ncapped at 0.1", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 0.1, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_CI_capped')
"""
#%% BANDGAP EXPONENT
"""
im.plot_heatmap(im.b[:,:,0], title = title_specimen + " - Bandgap exponent (b)", 
                cbar_kws={'label':'[-]', 'shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]",  cmap = cmap,
                mask = mask, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponent')

im.plot_heatmap(im.b[:,:,0], title = title_specimen + " - Bandgap exponent (b), \n b = 1", 
                cbar_kws={'label':'[-]', 'shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 1.0,  
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponent_capped')

im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/(2*im.b[:,:,0]), title = title_specimen + " - Relative error Bandgap exponent", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]" ,ylab = "[nm]", cmap = cmap,
                mask = mask, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponenent_Error')

im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/(2*im.b[:,:,0]), title = title_specimen + " - Relative error Bandgap exponent", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]" ,ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 1.0,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponenent_Error_capped')
"""
"""
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/(im.b[:,:,0]), title = title_specimen + " - Relative broadness CI Bandgap exponent", 
                cbar_kws={'label': 'Ratio [-]','shrink':cb_scale}, 
                xlab = "[nm]" ,ylab = "[nm]", cmap = cmap,
                mask = mask, 
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponenent_CI')

im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/(im.b[:,:,0]), title = title_specimen + " - Relative broadness CI Bandgap exponent, \ncapped at max 0.2", 
                cbar_kws={'label': 'Ratio [-] ','shrink':cb_scale}, 
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax = 0.2,  
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponent_CI_capped')
"""
#%% BANDGAP DISCRETIZED


mask_E_band = (mask | ((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0] >= 1))
size_E_band_bins = round_to_nearest(np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[~mask_E_band],50)/2,0.2)
E_band_round  = np.round(im.E_band[:,:,0]/size_E_band_bins) * size_E_band_bins
im.plot_heatmap(E_band_round, title = title_specimen + " - Bandgap energy", 
                cbar_kws={'label':'Energy [eV]','shrink':cb_scale}, color_bin_size = size_E_band_bins, discrete_colormap = True, sig_cbar = 2,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmin = 0.6, vmax = 2.6,  
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_Discretized')

#%% BANDGAP EXPONENT
"""
mask_b = (mask | (im.b[:,:,0] == 0))
size_b_bins = round_to_nearest(np.nanpercentile((im.b[:,:,2]-im.b[:,:,1])[~mask_b],50)/2,0.2)
b_round  = np.round(im.b[:,:,0]/size_b_bins) * size_b_bins
im.plot_heatmap(b_round, title = title_specimen + " - Bandgap exponent", 
                cbar_kws={'label':'[-]','shrink':cb_scale}, color_bin_size = size_b_bins, discrete_colormap = True, sig_cbar = 2,
                xlab = "[nm]", ylab = "[nm]", cmap = cmap,
                mask = mask, vmax=1,
                sig_ticks = sig_ticks, scale_ticks = scale_ticks, npix_xtick = npix_xtick, npix_ytick = npix_ytick, tick_int = tick_int, 
                save_as = save_loc + save_title_specimen + '_Bandgap_exponent_Discretized')
"""

#%% BANDGAP FIT INDIVIDUAL PIXELS

def bandgap_test(x, amp, BG, b=1.5):
    result = np.zeros(x.shape)
    result[x<BG] = 1
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result

for i in np.arange(0, len(im.x_axis), 30):
    for j in np.arange(0, len(im.y_axis), 30):
        try:
            if i != 0 and j != 0:
                pixx=i
                pixy=j
                [ts, IEELSs, max_IEELSs], [epss, ts_p, S_ss_p, IEELSs_p, max_IEELSs_p] = im.KK_pixel(pixy, pixx, signal = "pooled", iterations=5)
                data = IEELSs_p
                p_ieels_median = np.nanpercentile(data, 50, axis = 0)
                p_ieels_low = np.nanpercentile(data, 16, axis = 0)
                p_ieels_high = np.nanpercentile(data, 84, axis = 0)
                dE1 = im.dE1[1, int(im.clustered[pixy,pixx])]
                
                #p_ieels = im.ieels_p[pixy,pixx,0,:]
                #p_low = im.ieels_p[pixy,pixx,1,:]
                #p_high = im.ieels_p[pixy,pixx,2,:]
                range1 = dE1 - 0.4 # Check dE1 and adjust range
                range2 = dE1 + 0.8
                baseline = np.average(p_ieels_median[(im.deltaE > range1 - 0.1) & (im.deltaE < range1)])
                popt, pcov = curve_fit(bandgap_test, im.deltaE[(im.deltaE > range1) & (im.deltaE < range2)], 
                                       p_ieels_median[(im.deltaE > range1) & (im.deltaE < range2)] - baseline, 
                                       p0 = [400,1.3], bounds=([0, 0.5], np.inf))
                #fig1, ax1 = plt.subplots()
                #ax1.fill_between(im.deltaE, p_low, p_high, alpha = 0.2)
                #ax1.plot(im.deltaE, p_ieels, label = "Spectrum")
                #ax1.plot(im.deltaE, ZLPs, label = "ZLP")
                #ax1.set_title("Energy loss function pixel[" + str(pixx) + ","+ str(pixy) + "]")
                #ax1.set_xlabel("energy loss [eV]")
                #ax1.set_ylabel("intensity")
                #ax1.plot(im.deltaE, bandgap_test(im.deltaE,popt[0],popt[1],popt[2]), label = "Spectrum")
                #ax1.legend()
                
                fig2, ax2 = plt.subplots(dpi=200)
                ax2.set_title(title_specimen + " - Bandgap fit pixel[" + str(pixx) + ","+ str(pixy) + "]")
                ax2.set_xlabel("Energy loss [eV]")
                ax2.set_ylabel("Intensity [a.u.]")
                ax2.set_ylim(0,500)
                ax2.set_xlim(0,5)
                ax2.fill_betweenx([-100,1000], x1 = range1, x2 = range2, color='r', alpha = 0.2)
                ax2.fill_between(im.deltaE, p_ieels_low, p_ieels_high, alpha = 0.2)
                ax2.plot(im.deltaE, p_ieels_median, label = "$I_{inel}$")
                ax2.plot(im.deltaE, bandgap_test(im.deltaE,popt[0],popt[1]), label = "Fit")
                ax2.legend(loc=2)
                
                plt.savefig(save_loc + save_title_specimen + '_Bandgap_fit_pixel[' + str(pixx) + ','+ str(pixy) + '].pdf')
                
                print("pixel[" + str(pixx) + ","+ str(pixy) + "] done, dE1 = " + str(round(dE1,4)) + ", BG = " + str(round(popt[1],4)))
                #print("pixel[" + str(pixx) + ","+ str(pixy) + "] done, dE1 = " + str(round(dE1,4)) + ", BG = " + str(round(popt[1],4)) + ", b = " + str(popt[2]))
        except:
            print("Whatever you wanted, it failed")
#%% EPSILON

"""
eps = im.eps[30,30,0,:]
eps_1 = eps.real
eps_2 = eps.imag
deltaE_eps = np.arange(0,len(eps)*15, 15)/1000

#eps_1_DM = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_1_pixel_30_30_m3d038_to_26d422.txt')
#eps_2_DM = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_2_pixel_30_30_m3d038_to_26d422.txt')
eps_1_DM = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_1_pixel_30_30_m3d038_to_26d422_single_iter_reftail.txt')
eps_2_DM = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_2_pixel_30_30_m3d038_to_26d422_single_iter_reftail.txt')

eps_1_DM_n20 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_1_pixel_30_30_m3d038_to_26d422_20_iter_reftail.txt')
eps_2_DM_n20 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_2_pixel_30_30_m3d038_to_26d422_20_iter_reftail.txt')


#eps_1_DM_n1 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_1_pixel_30_30_m3d038_to_26d422_single_iter.txt')
#eps_2_DM_n1 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Epsilon_1_pixel_30_30_m3d038_to_26d422_single_iter.txt')

fig3, ax3 = plt.subplots(dpi=200)
fig4, ax4 = plt.subplots(dpi=200)
ax3.set_title("epsilon 1 pixel[30,30]")
ax3.set_xlabel("energy loss [eV]")
#ax3.set_ylabel("Crossings")
ax4.set_title("epsilon 2 pixel[30,30]")
ax4.set_xlabel("energy loss [eV]")
#ax4.set_ylabel("Crossings")
ax3.set_ylim(-1,7)
ax4.set_ylim(0,6)
#ax3.set_xlim(-3,5)
#ax3.fill_between(im.deltaE, p_low, p_high, alpha = 0.2)
ax3.plot(deltaE_eps, eps_1, label = "EELSfitter")
ax3.plot(im.deltaE, eps_1_DM, label = "DM, n=1")
ax3.plot(im.deltaE, eps_1_DM_n20, label = "DM, n=20")
ax3.axhline(y=0, color='black', linestyle='-')
ax4.plot(deltaE_eps, eps_2, label = "EELSfitter")
ax4.plot(im.deltaE, eps_2_DM, label = "DM, n=1")
ax4.plot(im.deltaE, eps_2_DM_n20, label = "DM, n=20")
ax4.axhline(y=0, color='black', linestyle='-')

ax3.legend()
ax4.legend()
"""
#%% Surface loss function and intensity
"""
Surloss_func_n1 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Surface-loss-function_pixel_30_30_m3d038_to_26d422_single_iter_reftail.txt')
Surloss_func_n20 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Surface-loss-function_pixel_30_30_m3d038_to_26d422_20_iter_reftail.txt')

Surloss_int_n1 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Surface-loss-intensity_pixel_30_30_m3d038_to_26d422_single_iter_reftail.txt')
Surloss_int_n20 = np.loadtxt('C:/Users/abelbrokkelkam/PhD/data/m20210331/eels/eels-SI/10n_dop_inse-B1_stem-eels-SI-processed_003_Surface-loss-intensity_pixel_30_30_m3d038_to_26d422_20_iter_reftail.txt')


fig5, ax5 = plt.subplots(dpi=200)
fig6, ax6 = plt.subplots(dpi=200)
ax5.set_title("Surface-Loss function pixel[30,30]")
ax5.set_xlabel("energy loss [eV]")
#ax5.set_ylabel("Crossings")
ax6.set_title("Surface-Loss intensity pixel[30,30]")
ax6.set_xlabel("energy loss [eV]")
#ax6.set_ylabel("Crossings")
#ax5.set_ylim(-1,7)
#ax6.set_ylim(0,6)
#ax5.set_xlim(-3,5)
#ax5.fill_between(im.deltaE, p_low, p_high, alpha = 0.2)
#ax5.plot(deltaE_eps, eps_1, label = "EELSfitter")
ax5.plot(im.deltaE, Surloss_func_n1, label = "n=1")
ax5.plot(im.deltaE, Surloss_func_n20, label = "n=20")
ax5.axhline(y=0, color='black', linestyle='-')
#ax6.plot(deltaE_eps, eps_2, label = "EELSfitter")
ax6.plot(im.deltaE, Surloss_int_n1, label = "n=1")
ax6.plot(im.deltaE, Surloss_int_n20, label = "n=20")
ax6.axhline(y=0, color='black', linestyle='-')

ax5.legend()
ax6.legend()
"""
#%%

