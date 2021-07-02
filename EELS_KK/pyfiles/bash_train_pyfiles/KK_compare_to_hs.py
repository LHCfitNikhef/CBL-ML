#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:29:41 2021

@author: isabel
"""
import numpy as np

from image_class_bs import Spectral_image, smooth_1D
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def median(data):
    return np.nanpercentile(data, 50, axis = 0)

def low(data):
    return np.nanpercentile(data, 16, axis = 0)

def high(data):
    return np.nanpercentile(data, 84, axis = 0)

def summary_distribution(data):
    return[median(data), low(data), high(data)]

def bandgap(x, amp, BG,b):
    result = np.zeros(x.shape)
    result[x<BG] = 50
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result


path_to_results = "../../KK_results/image_KK.pkl"
im = Spectral_image.load_spectral_image(path_to_results)

eps_hs = np.load('eps_hs_004.npy')
t_hs = np.load('t_hs_004.npy')

cmap="coolwarm"


ratio_t = t_hs/im.t[:,:,0]

im.plot_heatmap(t_hs, title = "hyperspy thickness from median IEELS", cmap=cmap)
im.plot_heatmap(t_hs, title = "hyperspy thickness from median IEELS, capped at 150", cmap=cmap, vmax = 150)
im.plot_heatmap(im.t[:,:,0], title = "median thickness from our code, capped at 150", cmap=cmap, vmax = 150, vmin=0)


im.plot_heatmap(ratio_t, title = "ratio hyperspy thickness from median IEELS and median \nthickness", cmap=cmap)
im.plot_heatmap(ratio_t, title = "ratio hyperspy thickness from median IEELS and median \nthickness, capped at max 2", cmap=cmap, vmax = 2)

ratio_eps = eps_hs[:,:,1:]/im.eps[:,:,0,:]
med_rat_eps = np.nanpercentile(ratio_eps, 50, axis=2)
low_rat_eps = np.nanpercentile(ratio_eps, 16, axis=2)
high_rat_eps = np.nanpercentile(ratio_eps, 84, axis=2)

#%%

im.plot_heatmap(np.imag(med_rat_eps), title = "median over energy axis ratio hyperspy eps_1 from \nmedian IEELS and median eps_1", cmap=cmap)
im.plot_heatmap(np.imag(med_rat_eps), title = "median over energy axis ratio hyperspy eps_1 from \nmedian IEELS and median eps_1, capped at 0,2", cmap=cmap, vmin=0, vmax=2)
im.plot_heatmap(np.imag(high_rat_eps-low_rat_eps), title = "CI ratio eps_1 from median IEELS and median eps_1, capped at max 1", cmap=cmap, vmax=1)
#im.plot_heatmap(np.imag(high_rat_eps-low_rat_eps), title = "CI ratio eps_1 from median IEELS and median eps_1, capped at max 1", cmap=cmap, vmax=1)
num_neg_ratio = np.sum(ratio_eps<0)/ratio_eps.size

#%%







E_cross = np.zeros(im.image_shape, dtype = 'object')
#E_cross = np.zeros((im.image_shape[1],1,3))
n_cross = np.zeros(im.image_shape)
first_crossing = np.zeros(im.image_shape)

for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        
        
        crossing = ((smooth_1D(np.imag(eps_hs[i,j]),50)[:-1]<0) * (smooth_1D(np.imag(eps_hs[i,j]),50)[1:] >=0))
        deltaE_n = im.deltaE[im.deltaE>0]
        #deltaE_n = deltaE_n[50:-50]
        crossing_E = deltaE_n[crossing.astype('bool')]
        
        E_cross[i,j] = crossing_E
        n_cross[i,j] = len(crossing_E)
        
        if len(crossing_E) > 0:
            first_crossing[i,j] = crossing_E[0]

    # if n_cross_lim > E_cross.shape[1]:
    #     E_cross_new = np.zeros((im.image_shape[1],n_cross_lim,3))
    #     E_cross_new[:,:E_cross.shape[1],:] = E_cross
    #     E_cross = E_cross_new
    #     del E_cross_new


im.plot_heatmap(n_cross, title = "number of crossings according to hs calculations", cmap=cmap)
im.plot_heatmap(n_cross, title = "number of crossings according to hs calculations, capped at max 5", cmap=cmap, vmax = 5)

im.plot_heatmap(first_crossing, title = "energy of first crossing according to hs calculations", cmap=cmap)
im.plot_heatmap(first_crossing, title = "energy of first crossing according to hs calculations capped around 20", cmap=cmap, vmin = 19 )
im.plot_heatmap(first_crossing, title = "energy of first crossing according to hs calculations, capped at max 12", cmap=cmap, vmax = 12)

#%%

cross_arg = np.argwhere(first_crossing > 0)


plt.figure()
plt.title("dielectric functions of one of the pixels that do cross in hyperspy, but not with us")
plt.plot(im.deltaE[-1773:], np.imag(eps_hs[77,71]), label = "hs")
plt.plot(im.deltaE[-1772:], np.imag(im.eps[77,71,0]), label = "us")
plt.fill_between(im.deltaE[-1772:], np.imag(im.eps[77,71,1]), np.imag(im.eps[77,71,2]))
plt.xlabel("energy loss [eV]")
plt.ylabel("dielectric function")
plt.legend()



plt.figure()
plt.title("dielectric functions of one of the pixels that do cross in hyperspy, but not with us")
plt.plot(im.deltaE[-1773:], np.imag(eps_hs[50,114]), label = "hs")
plt.plot(im.deltaE[-1772:], np.imag(im.eps[50,114,0]), label = "us")
plt.xlabel("energy loss [eV]")
plt.ylabel("dielectric function")
plt.legend()
plt.figure()
plt.title("dielectric functions of one of the pixels that do cross in hyperspy, but not with us")
plt.plot(im.deltaE[-1773:], np.imag(eps_hs[50,114]), label = "hs")
plt.plot(im.deltaE[-1772:], np.imag(im.eps[50,114,0]), label = "us")
plt.xlabel("energy loss [eV]")
plt.ylabel("dielectric function")
plt.legend()
plt.ylim(-1,6.5)


plt.figure()
plt.title("dielectric functions of one of the pixels that do cross in hyperspy, but not with us")
plt.plot(im.deltaE[-1773:], np.imag(eps_hs[62,95]), label = "hs")
plt.plot(im.deltaE[-1772:], np.imag(im.eps[62,95,0]), label = "us")
plt.xlabel("energy loss [eV]")
plt.ylabel("dielectric function")
plt.legend()

plt.figure()
plt.title("dielectric functions of one of the pixels that do cross in hyperspy, but not with us")
plt.plot(im.deltaE[-1773:], np.imag(eps_hs[62,95]), label = "hs")
plt.plot(im.deltaE[-1772:], np.imag(im.eps[62,95,0]), label = "us")
plt.xlabel("energy loss [eV]")
plt.ylabel("dielectric function")
plt.ylim(-1,4)
plt.legend()



#%%











