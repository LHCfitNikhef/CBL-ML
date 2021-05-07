#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:13:47 2021

@author: isabel
"""
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from image_class_bs import Spectral_image


# path_to_results = "../../KK_results/image_KK_003_p_5_again.pkl"
# im = Spectral_image.load_Spectral_image(path_to_results)

path_to_results = "../../KK_results/image_KK_003_p_5_own_zlps.pkl"
im = Spectral_image.load_Spectral_image(path_to_results)
# # im.pixelsize *=1E6
im.calc_axes()
im.cluster(5, based_upon = 'log')


im = im
cmap="YlGnBu"
cmap="coolwarm"


#%%
thicknesslimit = np.nanpercentile(im.t[im.clustered == 0],97)
mask = (im.t[:,:,0]<thicknesslimit)

save_loc = "../../plots/plots_symposium/003/"

im.plot_heatmap(im.t[:,:,0], title = "thickness sample", cbar_kws={'label': '[nm]'}, cmap = cmap, mask = mask, save_as = save_loc + "003_t")
im.plot_heatmap(im.t[:,:,0], title = "thickness sample, capped at max 40", cbar_kws={'label': '[nm]'}, vmax = 40, vmin =0, cmap = cmap, mask = mask, save_as = save_loc + "003_t_capped_40")
im.plot_heatmap(im.t[:,:,0], title = "thickness sample, capped at max 40", cbar_kws={'label': '[nm]'}, vmax = 40, vmin =0, cmap = cmap, save_as = save_loc + "003_t_capped_40")

im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0], title = "relative broadness CI thickness sample", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0], title = "relative broadness CI thickness sample, capped at 0, 0.10", cbar_kws={'label': '[-]'}, cmap = cmap, vmax=0.1, vmin=0, mask = mask, save_as = save_loc + "003_t_CI")
im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0], title = "relative broadness CI thickness sample, capped at 0, 0.20", cbar_kws={'label': '[-]'}, cmap = cmap, vmax=0.2, vmin=0, save_as = save_loc + "003_t_CI")


#%% PLOT MAX

max_ieels = im.max_ieels # im.deltaE[np.argmax(im.ieels, axis = 3)]

im.plot_heatmap(max_ieels[:,:,0], title = "max IEELS spectrum", cbar_kws={'label': '[eV]'}, cmap = cmap, mask = mask)
im.plot_heatmap(max_ieels[:,:,0], title = "max IEELS spectrum, capped at 22,26", cbar_kws={'label': '[eV]'}, cmap = cmap, mask = mask, vmin = 22, vmax=26, save_as = save_loc + "003_max")
im.plot_heatmap(max_ieels[:,:,0], title = "max IEELS spectrum", cbar_kws={'label': '[eV]'}, cmap = cmap)


im.plot_heatmap((max_ieels[:,:,2]-max_ieels[:,:,1])/max_ieels[:,:,0], title = "relative broadness CI maximum IEELS, capped at max 0.5", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask, vmax = 0.5, save_as = save_loc + "003_max_CI")

#%% NUM CROSSINGS
#TODO

im.n_cross = np.round(im.n_cross)

mask_cross = (mask | (im.n_cross[:,:,0] == 0))
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1]), title = "broadness CI numbers crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
#im.plot_heatmap(im.n_cross[:,:,2]-im.n_cross[:,:,1], title = "broadness CI numbers crossings real part dielectric function, \ncapped at max 3", cbar_kws={'label': 'nr. crossings'}, vmax=3, cmap = cmap)

mask_cross = (mask | (im.n_cross[:,:,2] == 0))
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1]), title = "broadness CI numbers crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)


#%%

first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) >0:
                first_crossings[i,j,:] = im.E_cross[i,j][0,:]
                first_crossings_CI[i,j] = (im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1])/im.E_cross[i,j][0,0]
        
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap, mask = mask_cross, save_as = save_loc + "E_cross")
#im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at min 21, max 23.5", cbar_kws={'label':  'energy [eV]'}, vmin=21, vmax=23.5, cmap = cmap)

im.plot_heatmap(first_crossings_CI, title = "relative broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'ratio [-]'}, cmap = cmap, mask = mask_cross, save_as = save_loc + "E_cross_CI")
# im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 3", cbar_kws={'label':  'energy [eV]'}, vmax=3, cmap = cmap)
# im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 0.5", cbar_kws={'label':  'energy [eV]'}, vmax=0.5, cmap = cmap)


size_E_bins = np.nanpercentile(first_crossings_CI[~mask_cross],50)/2
size_E_bins = np.nanpercentile((first_crossings[:,:,2]-first_crossings[:,:,1])[~mask_cross],50)/2
E_round  = np.round(first_crossings[:,:,0]/size_E_bins) * size_E_bins

im.plot_heatmap(E_round, title = "discretized energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap, mask = mask_cross, discrete_colormap = True, save_as = save_loc + "E_cross_discr")



"""
first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings = np.zeros(np.append(im.image_shape, 3))
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if im.n_cross[i,j,0] > 1:
            if len(im.E_cross[i,j]) >0:
                first_crossings[i,j,:] = im.E_cross[i,j][0,:]
                first_crossings_CI[i,j] = im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1]
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function\n(for chance at least 1 crossing > 0.5)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.5)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap)
"""
#%%

im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band")
im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at max 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_max", vmax= 2.5)
im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_min", vmin= 2.5)


# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2, mask = mask)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "relative broadness CI bandgap energies sample", cbar_kws={'label': ' [-]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_CI")
im.plot_heatmap(im.E_band[:,:,2]-im.E_band[:,:,1], title = "relative broadness CI bandgap energies sample, \ncapped at max 2", cbar_kws={'label': 'energy [eV]'}, vmax=2, cmap = cmap, save_as = save_loc + "E_band_CI_capped", mask = mask)

#%%
mask_E_band = (mask | ((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0] >= 1))

size_E_bins = np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[~mask_E_band],50)/2
E_round  = np.round(im.E_band[:,:,0]/size_E_bins) * size_E_bins

im.plot_heatmap(E_round, title = "discretized bandgap energies sample", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask_E_band, save_as = save_loc + "E_band_discr", color_bin_size = size_E_bins, discrete_colormap = True, sig=3)
im.plot_heatmap(E_round, title = "discretized bandgap energies sample, capped at 3", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmax = 3.06, mask = mask_E_band, save_as = save_loc + "E_band_discr_capped", color_bin_size = size_E_bins, discrete_colormap = True, sig=3)





# im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-] (??)'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) sample, \ncapped at max 1", cbar_kws={'label': '[-] '}, vmax=1, cmap = cmap, mask = mask)

#%%
losse_figure = False
column = 70
plt.figure()
plt.plot(im.y_axis, im.t[:,column,0], label = str(column))
plt.fill_between(im.y_axis, im.t[:,column,2], im.t[:,column,1], alpha = 0.3)
plt.title("thickness over column " + str(column))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")

column = 20
if losse_figure: plt.figure()
plt.plot(im.y_axis, im.t[:,column,0], label = str(column))
plt.fill_between(im.y_axis, im.t[:,column,2], im.t[:,column,1], alpha = 0.3)
plt.title("thickness over column " + str(column))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")

column = 120
if losse_figure: plt.figure()
plt.plot(im.y_axis, im.t[:,column,0], label = str(column))
plt.fill_between(im.y_axis, im.t[:,column,2], im.t[:,column,1], alpha = 0.3)
plt.title("thickness over column " + str(column))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")


if not losse_figure: 
    plt.legend()
    plt.title("thickness over different columns")

plt.savefig(save_loc + "thickness_over three_columns.pdf")

"""
#%% Adjust mask to pooling

mask = mask[1:-2,1:-2]


#%% BANDGAP RESULTS 0.8
# size_b_bins = np.nanpercentile((im.b[:,:,2]-im.b[:,:,1])[im.clustered != 0],50)/2
# b_round  = np.round(im.b[:,:,0]/size_b_bins) * size_b_bins

# im.plot_heatmap(b_round, title = "b-value (exponent in bandgap equation) sample, \ndiscretized but the colorbar not yet...", cbar_kws={'label': '[-] (??)'}, cmap = cmap)


BG08 = np.load("../../KK_results/E_band_04_08.npy")[:-3,:-3,:]
b08 = np.load("../../KK_results/b_band_04_08.npy")[:-3,:-3,:]
BG12 = np.load("../../KK_results/E_band_04_12.npy")[:-3,:-3,:]
b12 = np.load("../../KK_results/b_band_04_12.npy")[:-3,:-3,:]

b08[b08==0] = 1e-14 
b12[b12==0] = 1e-14 
BG08[BG08==0] = 1e-14 
BG12[BG12==0] = 1e-14 

im.b = b08
im.E_band = BG08

size_b_bins = np.nanpercentile((im.b[:,:,2]-im.b[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
b_round  = np.round(im.b[:,:,0]/size_b_bins) * size_b_bins
size_E_bins = np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
E_round  = np.round(im.E_band[:,:,0]/size_b_bins) * size_b_bins



im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample (0.8)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask)
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample(0.8), capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "relative broadness CI bandgap energies sample (0.8)", cbar_kws={'label': 'CI/median'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "broadness CI bandgap energies sample, \ncapped at max 4", cbar_kws={'label': 'energy [eV]'}, vmax=4, cmap = cmap)

im.plot_heatmap(b_round, title = "bandgap exponent sample (0.8), discretized", cbar_kws={'label':  '[-] ??'}, cmap = cmap, discrete_colormap = True, mask = mask)
im.plot_heatmap(E_round, title = "bandgap energy sample (0.8), discretized", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, discrete_colormap = True, mask = mask)


im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample (0.8)", cbar_kws={'label': '[-] (??)'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (0.8)", cbar_kws={'label': 'CI/median'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (0.8), capped at max 5", cbar_kws={'label': 'CI/median'}, vmax=5, cmap = cmap, mask = mask)

#%% BANDGAP RESULTS 1.2
im.b = b12
im.E_band = BG12

im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample (1.2)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask)
#im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample (1.2), capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "relative broadness CI bandgap energies sample (1.2)", cbar_kws={'label': 'CI/median'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "broadness CI bandgap energies sample, \ncapped at max 4", cbar_kws={'label': 'energy [eV]'}, vmax=4, cmap = cmap)


im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample (1.2)", cbar_kws={'label': '[-] (??)'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (1.2)", cbar_kws={'label':'CI/median'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (1.2), capped at max 5", cbar_kws={'label': 'CI/median'}, vmax=5, cmap = cmap, mask = mask)

size_b_bins = np.nanpercentile((im.b[:,:,2]-im.b[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
b_round  = np.round(im.b[:,:,0]/size_b_bins) * size_b_bins
size_E_bins = np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
E_round  = np.round(im.E_band[:,:,0]/size_b_bins) * size_b_bins

im.plot_heatmap(b_round, title = "bandgap exponent sample (1.2), discretized", cbar_kws={'label':  '[-] ??'}, cmap = cmap, discrete_colormap = True, mask = mask)
im.plot_heatmap(E_round, title = "bandgap energy sample (1.2), discretized", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, discrete_colormap = True, mask = mask)

#%% BANDGAP RESULTS 1.6



BG16 = np.load("../../KK_results/E_band_04_16.npy")[:-3,:-3,:]
b16 = np.load("../../KK_results/b_band_04_16.npy")[:-3,:-3,:]

b16[b16==0] = 1e-14 
BG16[BG16==0] = 1e-14 

im.b = b16
im.E_band = BG16

size_b_bins = np.nanpercentile((im.b[:,:,2]-im.b[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
b_round  = np.round(im.b[:,:,0]/size_b_bins) * size_b_bins
size_E_bins = np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[im.clustered[:-3,:-3] != 0],50)/2
E_round  = np.round(im.E_band[:,:,0]/size_b_bins) * size_b_bins


im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample (1.6)", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask)
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample(0.8), capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "relative broadness CI bandgap energies sample (1.6)", cbar_kws={'label': 'CI/median'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0], title = "broadness CI bandgap energies sample, \ncapped at max 4", cbar_kws={'label': 'energy [eV]'}, vmax=4, cmap = cmap)

im.plot_heatmap(b_round, title = "bandgap exponent sample (1.6), discretized", cbar_kws={'label':  '[-] ??'}, cmap = cmap, discrete_colormap = True, mask = mask)
im.plot_heatmap(E_round, title = "bandgap energy sample (1.6), discretized", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, discrete_colormap = True, mask = mask)


im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample (1.6)", cbar_kws={'label': '[-] (??)'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (1.6)", cbar_kws={'label': 'CI/median'}, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (1.6), capped at max 5", cbar_kws={'label': 'CI/median'}, vmax=5, cmap = cmap, mask = mask)
im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) \nsample (1.6), capped at max 2", cbar_kws={'label': 'CI/median'}, vmax=2, cmap = cmap, mask = mask)

"""


#%%
"""
first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) >1:
                first_crossings[i,j,:] = im.E_cross[i,j][1,:]
                first_crossings_CI[i,j] = im.E_cross[i,j][1,2]-im.E_cross[i,j][0,1]
        
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy second crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap)
im.plot_heatmap(first_crossings[:,:,0], title = "energy second crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at min 20, max 24", cbar_kws={'label':  'energy [eV]'}, vmin=20, vmax=24, cmap = cmap)

im.plot_heatmap(first_crossings_CI, title = "broadness CI energy second crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'nr. crossings'}, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy second crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 3", cbar_kws={'label':  'energy [eV]'}, vmax=3, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy second crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 0.5", cbar_kws={'label':  'energy [eV]'}, vmax=0.5, cmap = cmap)
"""
#%%
"""
first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) >0:
                first_crossings[i,j,:] = im.E_cross[i,j][0,:]
                first_crossings_CI[i,j] = im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1]
        
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap)
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at min 20, max 24", cbar_kws={'label':  'energy [eV]'}, vmin=1, vmax=5, cmap = cmap)

im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'nr. crossings'}, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 3", cbar_kws={'label':  'energy [eV]'}, vmax=3, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 0.5", cbar_kws={'label':  'energy [eV]'}, vmax=0.5, cmap = cmap)
"""