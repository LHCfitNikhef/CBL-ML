#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:15:21 2021

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
    #im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
    path_to_models = 'models/report/lau_clu10_p5_final_5dE1_06dE1/'
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_1iter.pkl"
    # path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"


    save_loc = "../../plots/final_report/lau/hs"
    try_pixels = [[65,7],[65,45],[65,83]]
    name = "sample from study Roest et al."
    xlim_times_dE1 = 6




im = Spectral_image.load_spectral_image(path_to_results)

im.load_zlp_models(path_to_models=path_to_models)
im.pool(5)
im.set_n(4.1462, n_background = 2.1759)

im = im
im.set_n(4.1462)
im.e0 = 200 #keV
im.beta = 30#67.2 #mrad

pixx = try_pixels[1][0]
pixy = try_pixels[1][1]


ieels = im.ieels[pixx,pixy,0,:]
zlp = im.pooled[pixx,pixy,:] - ieels

np.save("ieels_" + str(pixx) + '_' + str(pixy) + '.npy', ieels)
np.save("zlp_" + str(pixx) + '_' + str(pixy) + '.npy', zlp)

#%%
ieelss = im.ieels[:,:,0,:]
zlps = im.pooled[:,:,:] - ieelss

np.save("ieels_lau.npy", ieelss)
np.save("zlps_lau.npy", zlps)

del ieelss, zlps

#%%

eps_on_avg, t, ss = im.kramers_kronig_hs( ieels,
                          N_ZLP=np.sum(zlp),
                          iterations=1,
                          n=im.n[im.clustered[pixx,pixy]])



de = im.deltaE[im.deltaE>0]


#%%

eps_hs = np.load("/Users/isabel/Documents/Studie/MEP/CBL-ML/hyperspy/eps_hs_lau.npy")
path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"
im = Spectral_image.load_spectral_image(path_to_results)
#%%



plt.figure()
plt.title("Comparison results real part of the dieelectric funtion with \n results HyperSpy on " +  name + ", pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, eps_on_avg.real, label =r"our results on median $I_{\rm inel}$")
plt.plot(de[1:], eps_hs[pixx,pixy,:].real, linestyle = (0, (3,8)), label =r"HS results on median $I_{\rm inel}$")
plt.plot(de,im.eps[pixx,pixy,0,:].real, color= 'black', lw= 0.75, label = r"median $\epsilon_1$ from ZLP distribution")
plt.legend()
plt.xlabel("energy loss [eV]")
plt.ylabel("real part dielectric function [F/m]")
plt.savefig("HS_compare_eps1.pdf")

plt.figure()
plt.title("Comparison results imaginairy part of the dieelectric funtion with \n results HyperSpy on " +  name + ", pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, eps_on_avg.imag, label =r"our results on median $I_{\rm inel}$")
plt.plot(de[1:], eps_hs[pixx,pixy,:].imag, linestyle = (0, (3,8)), label =r"HS results on median $I_{\rm inel}$")
plt.plot(de,im.eps[pixx,pixy,0,:].imag,  color= 'black',lw= 0.75, label = r"median $\epsilon_2$ from ZLP distribution")
plt.legend()
plt.xlabel("energy loss [eV]")
plt.ylabel("imaginairy part dielectric function [F/m]")
plt.savefig("HS_compare_eps2.pdf")


ss_hs = np.load("/Users/isabel/Documents/Studie/MEP/CBL-ML/hyperspy/S_s_hs_lau_i2.npy")/0.014999999664723873

plt.figure()
plt.title("Comparison results surface scattering after one iteration \n with results HyperSpy \n on " +  name + "pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, ss)
plt.plot(de[1:], ss_hs[pixx, pixy,:])
plt.plot(de[:], im.ss[pixx, pixy,0,:],  color= 'black',lw= 0.75, linestyle = (0, (5,13)), label = r"median $I_{s}$ from ZLP distribution")

plt.xlabel("energy loss [eV]")
plt.ylabel("intensity [arb. units]")
plt.legend()




#%%

if plotim == 'lau':
    #im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
    path_to_models = 'models/report/lau_clu10_p5_final_5dE1_06dE1/'
    path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"
    # path_to_results = "../../KK_results/report/image_KK_lau_clu10_pooled_5_5dE1_06dE1_5iter.pkl"


    save_loc = "../../plots/final_report/lau/hs"
    try_pixels = [[65,7],[65,45],[65,83]]
    name = "sample from study Roest et al."
    xlim_times_dE1 = 6



im = Spectral_image.load_spectral_image(path_to_results)

im.load_zlp_models(path_to_models=path_to_models)
im.pool(5)
im.set_n(4.1462, n_background = 2.1759)
im.e0 = 200 #keV
im.beta = 30#67.2 #mrad


eps_hs = np.load("/Users/isabel/Documents/Studie/MEP/CBL-ML/hyperspy/eps_hs_lau_i5.npy")
ss_hs = np.load("/Users/isabel/Documents/Studie/MEP/CBL-ML/hyperspy/S_s_hs_lau_i6.npy")#/0.014999999664723873/0.014999999664723873

eps_on_avg_5, t, ss_5 = im.kramers_kronig_hs( ieels,
                          N_ZLP=np.sum(zlp),
                          iterations=5,
                          n=im.n[im.clustered[pixx,pixy]])
#%%

plt.figure()
plt.title("Comparison results real part of the dieelectric funtion with \n results HyperSpy on " +  name + ", pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, eps_on_avg_5.real, label =r"our results on median $I_{\rm inel}$")
plt.plot(de[1:], eps_hs[pixx,pixy,:].real, '--', label =r"HS results on median $I_{\rm inel}$")
plt.plot(de,im.eps[pixx,pixy,0,:].real, color= 'black', lw= 0.75, linestyle = (0, (5,13)), label = r"median $\epsilon_1$ from ZLP distribution")
plt.legend()
plt.xlabel("energy loss [eV]")
plt.ylabel("real part dielectric function [F/m]")

plt.figure()
plt.title("Comparison results imaginairy part of the dieelectric funtion with \n results HyperSpy on " +  name + ", pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, eps_on_avg_5.imag, label =r"our results on median $I_{\rm inel}$")
plt.plot(de[1:], eps_hs[pixx,pixy,:].imag, '--', label =r"HS results on median $I_{\rm inel}$")
plt.plot(de,im.eps[pixx,pixy,0,:].imag,  color= 'black',lw= 0.75, linestyle = (0, (5,13)), label = r"median $\epsilon_2$ from ZLP distribution")
plt.legend()
plt.xlabel("energy loss [eV]")
plt.ylabel("real part dielectric function [F/m]")

plt.figure()
plt.title("Comparison results surface scattering after five iteration with results \nHyperSpy on " +  name + "pixel (" +str(pixx) + "," + str(pixy) + ")")
plt.plot(de, ss, label =r"our results on median $I_{\rm inel}$")
#plt.plot(de,im.ss[pixx,pixy,0,:])
plt.plot(de[1:], ss_hs[pixx, pixy,:], linestyle = (0, (3,8)), label =r"HS results on median $I_{\rm inel}$")
plt.plot(de[:], im.ss[pixx, pixy,0,:],  color= 'black',lw= 0.75, label = r"median $I_{s}$ from ZLP distribution")
plt.xlabel("energy loss [eV]")
plt.ylabel("intensity [arb. units]")
plt.legend()
plt.savefig("HS_compare_Ss.pdf")
