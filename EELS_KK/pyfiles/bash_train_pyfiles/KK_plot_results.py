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


path_to_results = "KK_results/image_KK.pkl"
im = Spectral_image.load_Spectral_image(path_to_results)
im.pixelsize *=1E6
im.calc_axes()

cmap="YlGnBu"
cmap="coolwarm"

im.plot_heatmap(im.t[:,:,0], title = "thickness sample", cbar_kws={'label': '[nm]'}, cmap = cmap)
im.plot_heatmap(im.t[:,:,0], title = "thickness sample, capped at max 150", cbar_kws={'label': '[nm]'}, vmax = 150, cmap = cmap)

im.plot_heatmap(im.t[:,:,2]-im.t[:,:,1], title = "broadness CI thickness sample", cbar_kws={'label': '[nm]'}, cmap = cmap)
im.plot_heatmap(im.t[:,:,2]-im.t[:,:,1], title = "broadness CI thickness sample, capped at max 150", cbar_kws={'label': '[nm]'}, cmap = cmap, vmax=150)



im.plot_heatmap(im.n_cross[:,:,0], title = "number crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap)
im.plot_heatmap(im.n_cross[:,:,2]-im.n_cross[:,:,1], title = "broadness CI numbers crossings real part dielectric function", cbar_kws={'label': 'nr. crossings'}, cmap = cmap)
im.plot_heatmap(im.n_cross[:,:,2]-im.n_cross[:,:,1], title = "broadness CI numbers crossings real part dielectric function, \ncapped at max 3", cbar_kws={'label': 'nr. crossings'}, vmax=3, cmap = cmap)


first_crossings = np.zeros(np.append(im.image_shape, 3))
first_crossings_CI = np.zeros(im.image_shape)
for i in range(im.image_shape[0]):
    for j in range(im.image_shape[1]):
        if type(im.E_cross[i,j]) == np.ndarray:
            if len(im.E_cross[i,j]) >0:
                first_crossings[i,j,:] = im.E_cross[i,j][0,:]
                first_crossings_CI[i,j] = im.E_cross[i,j][0,2]-im.E_cross[i,j][0,1]
        
        
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'energy [eV]'}, cmap = cmap)
im.plot_heatmap(first_crossings[:,:,0], title = "energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at min 20, max 24", cbar_kws={'label':  'energy [eV]'}, vmin=20, vmax=24, cmap = cmap)

im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1)", cbar_kws={'label': 'nr. crossings'}, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 3", cbar_kws={'label':  'energy [eV]'}, vmax=3, cmap = cmap)
im.plot_heatmap(first_crossings_CI, title = "broadness CI energy first crossing real part dielectric function \n(for chance at least 1 crossing > 0.1), \ncapped at max 0.5", cbar_kws={'label':  'energy [eV]'}, vmax=0.5, cmap = cmap)


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

#%%

im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample", cbar_kws={'label':  'energy [eV]'}, cmap = cmap)
im.plot_heatmap(im.E_band[:,:,0], title = "bandgap energies sample, capped at max 2 eV", cbar_kws={'label':  'energy [eV]'}, cmap = cmap, vmax = 2)
im.plot_heatmap(im.E_band[:,:,2]-im.E_band[:,:,1], title = "broadness CI bandgap energies sample", cbar_kws={'label': 'energy [eV]'}, cmap = cmap)
im.plot_heatmap(im.E_band[:,:,2]-im.E_band[:,:,1], title = "broadness CI bandgap energies sample, \ncapped at max 4", cbar_kws={'label': 'energy [eV]'}, vmax=4, cmap = cmap)


im.plot_heatmap(im.b[:,:,0], title = "b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-] (??)'}, cmap = cmap)
im.plot_heatmap(im.b[:,:,2]-im.b[:,:,1], title = "broadness CI b-value (exponent in bandgap equation) sample", cbar_kws={'label': '[-] (??)'}, cmap = cmap)
im.plot_heatmap(im.b[:,:,2]-im.b[:,:,1], title = "broadness CI b-value (exponent in bandgap equation) sample, \ncapped at max 1.5", cbar_kws={'label': '[-] (??)'}, vmax=1.5, cmap = cmap)

#%%
losse_figure = False
row = 70
plt.figure()
plt.plot(im.x_axis, im.t[row,:,0], label = str(row))
plt.fill_between(im.x_axis, im.t[row,:,2], im.t[row,:,1], alpha = 0.3)
plt.title("thickness over row " + str(row))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")

row = 20
if losse_figure: plt.figure()
plt.plot(im.x_axis, im.t[row,:,0], label = str(row))
plt.fill_between(im.x_axis, im.t[row,:,2], im.t[row,:,1], alpha = 0.3)
plt.title("thickness over row " + str(row))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")

row = 120
if losse_figure: plt.figure()
plt.plot(im.x_axis, im.t[row,:,0], label = str(row))
plt.fill_between(im.x_axis, im.t[row,:,2], im.t[row,:,1], alpha = 0.3)
plt.title("thickness over row " + str(row))
plt.ylabel("thickness [nm]")
plt.xlabel("[nm]")


if not losse_figure: 
    plt.legend()
    plt.title("thickness over different rows")


#%%

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

#%%

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
