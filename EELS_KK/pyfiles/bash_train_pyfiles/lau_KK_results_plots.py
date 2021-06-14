#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:40:39 2021

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

path_to_results = "../../KK_results/image_KK_004_clu_10_p_5.pkl"
path_to_results = "../../KK_results/image_KK_lau_clu5_pooled_5.pkl"
path_to_results = "../../KK_results/report/image_KK_004_clu10_p5_35dE1_06dE1.pkl"
path_to_results = "../../KK_results/report/image_KK_ 004_clu10_pooled_5_35dE1_06dE1_5iter.pkl"


plotim = 'lau'





if plotim == 'lau':
    im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
    path_to_models = 'models/report/lau_clu10_p5_final_5dE1_06dE1/'
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_1iter.pkl"
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"
    # path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_1iter_dE1_0_set16.pkl"
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_1iter_dE1_0_set25_nbg11.pkl"


    save_loc = "../../plots/final_report/lau/KK/png/"
    try_pixels = [[7,65],[45,65],[83,65]]
    name = "sample from study Roest et al."
    xlim_times_dE1 = 6





if plotim == '004':
    im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
    path_to_models = 'models/dE2_3_times_dE1/train_004_not_pooled_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'

    save_loc = "../../plots/final_report/004/KK/"
    try_pixels = [[7,65],[45,65],[83,65]]
    name = "sample with WS$_2$ nanostructures"
    xlim_times_dE1 = 4
    
    path_to_results = "../../KK_results/report/image_KK_004_clu10_pooled_5_3dE1_07dE1_1iter.pkl"




path_to_models += (path_to_models[-1]!='/')*'/'

im = Spectral_image.load_Spectral_image(path_to_results)

im.ieels = None
im.ieels_p = None
# im.data = None


# # im.pixelsize *=1E6
im.calc_axes()
# im.cluster(5, based_upon = 'log')




#path_to_models = 'models/dE2_3_times_dE1/train_lau_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_5/'
#path_to_models = 'models/report/004_clu10_p5_final_35dE1_06dE1/'

im.load_ZLP_models_smefit(path_to_models, name_in_path = False)
#im.dE1[0,1] = 1.6 

im = im
cmap="YlGnBu"
cmap="coolwarm"
t_OG = np.copy(im.t)
#%%

n_vac = 1.1
n_ws2 = 4.1462

im.t = np.copy(t_OG)

n_1 = n_vac/2 + n_ws2/2
n_2 = 9*n_vac/10 + n_ws2/10
n_3 = 19*n_vac/20 + n_ws2/20

#n_2 = n_1
n_1 = n_ws2
adap_K_1 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_1 ** 2)
adap_K_2 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_2 ** 2) 
adap_K_3 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_3 ** 2) 

# im.plot_heatmap(np.maximum(im.t[:,:,0],0), mask = (im.clustered!=0), title = "thickness " + name, cbar_kws={'label': '[nm]'}, cmap = cmap)

for j in range(im.shape[1]):
    i= np.max(np.argwhere(im.clustered[:,j] ==0))
    im.t[i,j,:] *= adap_K_1
    im.t[i-1,j,:] *= adap_K_2
    im.t[i-2,j,:] *= adap_K_3
    
# im.plot_heatmap(np.maximum(im.t[:,:,0],0), mask = (im.clustered!=0), title = "thickness " + name, cbar_kws={'label': '[nm]'}, cmap = cmap)

for i in range(im.n_clusters):
    print("thickness cluster", i, ":", np.round(np.nanpercentile(im.t[:,:,0][im.clustered == i],50),3))
#%%




#im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name + ",\n surface corrected, capped at 1.8", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmax = 2, mask = (mask | mask_E_band), save_as = save_loc + "E_band_discr_capped", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)
#print("stop maar")
#%%
thicknesslimit = np.nanpercentile(im.t[im.clustered == 0],99.5)
mask = (im.t[:,:,0]<thicknesslimit)
#%%
#save_loc = "../../plots/final_report/004/plots/KK/"
im.plot_heatmap(im.clustered, title = "clustered image of " + name, cbar_kws={'label': '[-]'}, cmap = cmap, discrete_colormap=True, save_as = save_loc + "clustered")
im.plot_heatmap(np.maximum(im.t[:,:,0],0), title = "thickness " + name, cbar_kws={'label': '[nm]'}, cmap = cmap,  save_as = save_loc + "t")
im.plot_heatmap(np.maximum(im.t[:,:,0],0), title = "thickness " + name+", capped at max 50", cbar_kws={'label': '[nm]'}, cmap = cmap, vmax=50, save_as = save_loc + "t")
im.plot_heatmap(np.maximum(im.t[:,:,0],0), title = "thickness "+ name +", capped at max 50", cbar_kws={'label': '[nm]'}, vmax = 50, cmap = cmap,  mask=mask, save_as = save_loc + "t_capped_40")
# im.plot_heatmap(np.maximum(im.t[:,:,0],0), title = "thickness "+ name +", capped at max 200", cbar_kws={'label': '[nm]'}, vmax = 200, cmap = cmap, save_as = save_loc + "t_capped_40")

im.plot_heatmap(np.absolute((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0]/2), title = "relative broadness CI thickness in "+ name, cbar_kws={'label': '[-]'}, vmax = 0.05, cmap = cmap)#, mask = mask)
#im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0]/2, title = "relative broadness CI thickness "+ name +", capped at 0, 0.10", cbar_kws={'label': '[-]'}, cmap = cmap, vmax=0.1, vmin=0, mask = mask, save_as = save_loc + "t_CI")
#im.plot_heatmap((im.t[:,:,2]-im.t[:,:,1])/im.t[:,:,0]/2, title = "relative broadness CI thickness "+ name +", capped at 0, 0.20", cbar_kws={'label': '[-]'}, cmap = cmap, vmax=0.2, vmin=0, save_as = save_loc + "t_CI")


#%% PLOT MAX

#max_ieels = im.max_ieels # im.deltaE[np.argmax(im.ieels, axis = 3)]


"""
im.plot_heatmap(im.max_ieels[:,:,0], title = "max IEELS spectrum", cbar_kws={'label': '[eV]'}, cmap = cmap)#, mask = mask)
im.plot_heatmap((im.max_ieels[:,:,2]-im.max_ieels[:,:,1])/im.max_ieels[:,:,0]/2, title = "relative broadness CI maximum IEELS, capped at max 0.5", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask, vmax = 0.5, save_as = save_loc + "max_CI")

im.plot_heatmap(im.max_ieels[:,:,0], title = "max IEELS spectrum, capped at 22,26", cbar_kws={'label': '[eV]'}, cmap = cmap, mask = mask, vmin = 22, vmax=26, save_as = save_loc + "max")
im.plot_heatmap(im.max_ieels[:,:,0], title = "max IEELS spectrum", cbar_kws={'label': '[eV]'}, cmap = cmap)


im.plot_heatmap((max_ieels[:,:,2]-max_ieels[:,:,1])/max_ieels[:,:,0]/2, title = "relative broadness CI maximum IEELS, capped at max 0.5", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask, vmax = 0.5, save_as = save_loc + "max_CI")
"""
#%%
# im.plot_heatmap(im.ssratio[:,:,0], title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask, save_as = save_loc + "ssratio")
# im.plot_heatmap(im.ssratio[:,:,0], title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'},  vmax = 1, cmap = cmap, mask = mask, save_as = save_loc + "ssratio_capped")

# im.plot_heatmap(im.ssratio[:,:,0], title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, vmax = 1, cmap = cmap, save_as = save_loc + "ssratio")
# im.plot_heatmap((im.ssratio[:,:,2]-im.ssratio[:,:,1])/im.ssratio[:,:,0]/2, title = "relative broadness CI ssratio "+ name ,vmin= 0, vmax = 0.5, cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask, save_as = save_loc + "ssratio_CI")
# im.plot_heatmap((im.ssratio[:,:,2]-im.ssratio[:,:,1])/im.ssratio[:,:,0]/2, title = "relative broadness CI ssratio "+ name, vmin= 0, vmax = 0.5,cbar_kws={'label': '[-]'}, cmap = cmap, save_as = save_loc + "ssratio_CI")



#%% NUM CROSSINGS
#TODO
"""
im.n_cross = np.round(im.n_cross)

mask_cross = (mask | (im.n_cross[:,:,0] == 0)) 
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1])/2, title = "broadness CI numbers crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
#im.plot_heatmap(im.n_cross[:,:,2]-im.n_cross[:,:,1], title = "broadness CI numbers crossings real part dielectric function, \n capped at max 3", cbar_kws={'label': 'nr. crossings'}, vmax=3, cmap = cmap)

mask_cross = (mask | (im.n_cross[:,:,2] == 0))
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1])/2, title = "broadness CI numbers crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)


#%%

first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) >0:
                if (im.E_cross[i,j][:,0] > 7).any():
                    first_crossings[i,j,:] = im.E_cross[i,j][im.E_cross[i,j][:,0] > 7,:][0,:]
                    first_crossings_CI[i,j] = (first_crossings[i,j,2]-first_crossings[i,j,1])/first_crossings[i,j,0]/2# (im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1])/im.E_cross[i,j][0,0]/2
                if (im.E_cross[i,j] > 7).any():
                    im.n_cross[i,j] -= 1

mask_cross = (mask | (im.n_cross[:,:,0] == 0)) 
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1])/2, title = "broadness CI numbers crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
#im.plot_heatmap(im.n_cross[:,:,2]-im.n_cross[:,:,1], title = "broadness CI numbers crossings real part dielectric function, \n capped at max 3", cbar_kws={'label': 'nr. crossings'}, vmax=3, cmap = cmap)

mask_cross = (mask | (im.n_cross[:,:,2] == 0))
im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)
im.plot_heatmap((im.n_cross[:,:,2]-im.n_cross[:,:,1])/2, title = "broadness CI numbers crossings real part dielectric function in "+ name, cbar_kws={'label': 'nr. crossings'}, cmap = cmap, mask = mask_cross, discrete_colormap = True)




mask_cross = (mask | ~(first_crossings[:,:,2] > 0))
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function in "+ name +"  \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap, mask = mask_cross, save_as = save_loc + "E_cross")
#im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at min 21, max 23.5", cbar_kws={'label':  'energy [eV]'}, vmin=21, vmax=23.5, cmap = cmap)

im.plot_heatmap(first_crossings_CI, title = "relative broadness CI energy first crossing real part dielectric function in "+ name +"\n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'ratio [-]'}, cmap = cmap, mask = mask_cross, save_as = save_loc + "E_cross_CI")
# im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 3", cbar_kws={'label':  'energy [eV]'}, vmax=3, cmap = cmap)
# im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 0.5", cbar_kws={'label':  'energy [eV]'}, vmax=0.5, cmap = cmap)


size_E_bins = np.nanpercentile(first_crossings_CI[~mask_cross],50)/2
size_E_bins = np.nanpercentile((first_crossings[:,:,2]-first_crossings[:,:,1])[~mask_cross],50)/2
E_round  = np.round(first_crossings[:,:,0]/size_E_bins) * size_E_bins

# im.plot_heatmap(E_round, title = "discretized energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap, mask = mask_cross, discrete_colormap = True, save_as = save_loc + "E_cross_discr")

"""

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
im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies "+ name , cbar_kws={'label':  'energy [eV]'}, cmap = cmap,  save_as = save_loc + "E_band")
im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies "+ name , cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band")
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies "+ name + ", capped at max 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_max", vmax= 2.5)
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_min", vmin= 2.5)


# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2, mask = mask)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0]/2, title = "relative broadness CI bandgap energies "+ name, cbar_kws={'label': ' [-]'}, cmap = cmap,  vmax=0.5, save_as = save_loc + "E_band_CI")
# im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0]/2, title = "relative broadness CI bandgap energies "+ name, cbar_kws={'label': ' [-]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_CI", vmin = 0, vmax=0.5)
im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0]/2, title = "relative broadness CI bandgap energies "+ name +", \ncapped at max 0.5", cbar_kws={'label': '[-]'}, vmax=0.5, cmap = cmap, save_as = save_loc + "E_band_CI_capped", mask = mask)

#%%
mask_E_band = ( ~((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0] < 0.5))
mask = (im.clustered == 0)


size_E_bins = np.nanpercentile((im.E_band[:,:,2]-im.E_band[:,:,1])[~mask_E_band],50)/2
E_round  = np.round(im.E_band[:,:,0]/size_E_bins) * size_E_bins

im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name, cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask_E_band, save_as = save_loc + "E_band_discr", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)
# im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name +", capped at 1.7", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmax = 1.7, mask = (mask | mask_E_band), save_as = save_loc + "E_band_discr_capped", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)
im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name, cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = (mask | mask_E_band), save_as = save_loc + "E_band_discr_extra", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)





# im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-] (??)'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask)
# im.plot_heatmap((im.b[:,:,2]-im.b[:,:,1])/im.b[:,:,0], title = "relative broadness CI b-value (exponent in bandgap equation) sample, \ncapped at max 1", cbar_kws={'label': '[-] '}, vmax=1, cmap = cmap, mask = mask)

#%%
"""
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
#%%

ssratio_1 = im.ssratio[:,:,0]
ssratio1 = (im.ssratio[:,:,2]-im.ssratio[:,:,1])/2/im.ssratio[:,:,0]
ss_1 = im.ss[:,:,0,:]
#%%
if plotim == 'lau':
    path_to_models = 'models/report/lau_clu10_p5_final_5dE1_06dE1/'
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter_dE1_0_set25_nbg11.pkl"
    # path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"

if plotim == '004':
    # im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
    path_to_models = 'models/dE2_3_times_dE1/train_004_not_pooled_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
    path_to_results = "../../KK_results/report/image_KK_004_clu10_pooled_5_3dE1_07dE1_5iter.pkl"




path_to_models += (path_to_models[-1]!='/')*'/'

im = Spectral_image.load_Spectral_image(path_to_results)

im.ieels = None
im.ieels_p = None
# im.data = None

"""
ss_sum_med = np.sum(np.absolute(im.ss[:,:,0,:]), axis = 2)
ss_sum_low = np.sum(np.minimum(np.absolute(im.ss[:,:,1,:]),np.absolute(im.ss[:,:,2,:])), axis = 2)
ss_sum_high = np.sum(np.maximum(np.absolute(im.ss[:,:,1,:]),np.absolute(im.ss[:,:,2,:])), axis = 2)

ieels_sum = np.sum(np.absolute(im.ieels[:,:,0,:]), axis = 2)

ssratio5 = ss_sum_med/ieels_sum
ssratio5_CI = (ss_sum_high-ss_sum_low)/ieels_sum/2

ssratio5 = im.ssratio[:,:,0]
ssratio5 = (im.ssratio[:,:,2]-im.ssratio[:,:,1])/2/im.ssratio[:,:,0]
"""

# # im.pixelsize *=1E6
im.calc_axes()
# im.cluster(5, based_upon = 'log')

#path_to_models = 'models/dE2_3_times_dE1/train_lau_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_5/'
#path_to_models = 'models/report/004_clu10_p5_final_35dE1_06dE1/'

im.load_ZLP_models_smefit(path_to_models, name_in_path = False)


#%%

n_vac = 1.1
n_ws2 = 4.1462

im.t = np.copy(t_OG)

n_1 = n_vac/2 + n_ws2/2
n_2 = 9*n_vac/10 + n_ws2/10
n_3 = 19*n_vac/20 + n_ws2/20

#n_2 = n_1
n_1 = n_ws2
adap_K_1 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_1 ** 2)
adap_K_2 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_2 ** 2) 
adap_K_3 = (1 - 1. / n_vac ** 2) / (1 - 1. / n_3 ** 2) 

# im.plot_heatmap(np.maximum(im.t[:,:,0],0), mask = (im.clustered!=0), title = "thickness " + name, cbar_kws={'label': '[nm]'}, cmap = cmap)

for j in range(im.shape[1]):
    i= np.max(np.argwhere(im.clustered[:,j] ==0))
    im.t[i,j,:] *= adap_K_1
    im.t[i-1,j,:] *= adap_K_2
    im.t[i-2,j,:] *= adap_K_3
    
#%%
#ssratio_ratio = np.absolute(im.ssratio[:,:,0]/ssratio_1)
mask_ss_conv = np.sum((np.absolute(im.ss[:,:,0,:]/ss_1)>3) * (np.absolute(im.ss[:,:,0,:]) > 100), axis = 2) >0
im.plot_heatmap(~(mask_ss_conv)*1, title = "pixels with converging surface scattering caulatioins \n in " + name, discrete_colormap = True, color_bin_size =1, save_as = save_loc + "non_conv_ss")
im.plot_heatmap(((~mask)*mask_ss_conv)*1, title = "overlap between non-converging pixels, and pixels with \n" + r" WS$_2$ present in " + name, discrete_colormap = True, color_bin_size =1, save_as = save_loc + "overlap_non_conv_ss")
#%%
"""
im.plot_heatmap(ssratio1, title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, cmap = cmap, save_as = save_loc + "ssratio")
im.plot_heatmap(ssratio1, title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, vmax = 0.01, cmap = cmap, save_as = save_loc + "ssratio")

im.plot_heatmap(ssratio1, title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, vmax = 0.01, cmap = cmap, mask = mask_ss_conv, save_as = save_loc + "ssratio")
im.plot_heatmap(ssratio1_CI, title = "relative broadness CI ssratio "+ name, cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask_ss_conv, save_as = save_loc + "ssratio_CI")
im.plot_heatmap(ssratio1_CI, title = "relative broadness CI ssratio "+ name, cbar_kws={'label': '[-]'}, cmap = cmap, save_as = save_loc + "ssratio_CI")

"""

#%%
mask_ss = mask_ss_conv | mask

print("first percentile thickness does converge:", np.nanpercentile(im.t[:,:,0][~mask_ss],1))
print("first promille thickness does converge:", np.nanpercentile(im.t[:,:,0][~mask_ss],0.1))
print("last percentile thickness does not converge:", np.nanpercentile(im.t[:,:,0][mask_ss],99))
print("last promille thickness does not converge:", np.nanpercentile(im.t[:,:,0][mask_ss],99.9))



im.plot_heatmap(im.ssratio[:,:,0], title = "surface scattering influence ratio "+ name , cbar_kws={'label': '[-]'}, cmap = cmap, mask = mask_ss, save_as = save_loc + "ssratio_5iter")
im.plot_heatmap(im.ssratio[:,:,0], title = "surface scattering influence ratio "+ name + ", capped at 0.05", cbar_kws={'label': '[-]'}, vmin= 0, vmax = 0.07, cmap = cmap, mask = mask_ss, save_as = save_loc + "ssratio_capped_5iter")
im.plot_heatmap(np.maximum(im.t[:,:,0],19.7), title = "thickness where surface scattering converges in\n"+ name , cbar_kws={'label': '[nm]'}, cmap = cmap, mask = mask_ss, vmax = 200, save_as = save_loc + "t_ssconv_5iter")
im.plot_heatmap(np.absolute(im.t[:,:,0]), title = "thickness where surface scattering does not \nconverge in "+ name , cbar_kws={'label': '[nm]'}, cmap = cmap, mask = ~mask_ss, save_as = save_loc + "t_not_ssconv_5iter")
im.plot_heatmap((im.ssratio[:,:,2]-im.ssratio[:,:,1])/im.ssratio[:,:,0]/2, title = "relative broadness CI ssratio "+ name , cbar_kws={'label': '[-]'}, vmax = 0.02, cmap = cmap, mask = mask_ss, save_as = save_loc + "_ssratio_CI_5iter")

#%%

im.plot_heatmap(im.E_band_sscor[:,:,0], title = "bandgap energies "+ name, cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_sscor")
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies "+ name + ", capped at max 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_max", vmax= 2.5)
# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 2.5", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_capped_min", vmin= 2.5)


# im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at min 0.7 eV, max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmin = 0.7, vmax = 2, mask = mask)
# im.plot_heatmap((im.E_band[:,:,2]-im.E_band[:,:,1])/im.E_band[:,:,0]/2, title = "relative broadness CI bandgap energies "+ name , cbar_kws={'label': ' [-]'}, cmap = cmap, mask = mask, save_as = save_loc + "E_band_CI")
im.plot_heatmap(np.absolute((im.E_band_sscor[:,:,2]-im.E_band_sscor[:,:,1]))/im.E_band_sscor[:,:,0]/2, title = "relative broadness CI bandgap energies "+ name + ", \ncapped at max 1", cbar_kws={'label': ' [-]'}, vmax=0.5, cmap = cmap, save_as = save_loc + "E_band_sscor_CI_capped", mask = mask)

#%%
mask_E_band = (( ~((im.E_band_sscor[:,:,2]-im.E_band_sscor[:,:,1])/im.E_band_sscor[:,:,0] < 0.5)) | ((im.E_band_sscor[:,:,0]<0.1)) | mask_ss_conv)

size_E_bins = np.nanpercentile((im.E_band_sscor[:,:,2]-im.E_band_sscor[:,:,1])[~mask_E_band],50)/2
E_round  = np.round(im.E_band_sscor[:,:,0]/size_E_bins) * size_E_bins

im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name + ",\n sample surface corrected", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = mask_E_band, save_as = save_loc + "E_band_sscor_discr", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)
im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name + ",\n surface corrected, capped at 1.8", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmax = 1.2, mask = (mask | mask_E_band), save_as = save_loc + "E_band_sscor_discr_capped", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)
im.plot_heatmap(E_round, title = "discretized bandgap energies "+ name, cbar_kws={'label':  'energy [eV]'}, cmap = cmap, mask = (mask | mask_E_band), vmax = 1.2, save_as = save_loc + "E_band_sscor_discr_extra", color_bin_size = size_E_bins, discrete_colormap = True, sig=2)



