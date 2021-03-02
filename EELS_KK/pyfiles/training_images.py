#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:52:45 2021

@author: isabel
"""
import matplotlib.pyplot as plt
import seaborn as sns
from image_class import Spectral_image
from train_nn_torch import train_nn_scaled, MC_reps, binned_statistics




im = Spectral_image.load_data('../dmfiles/h-ws2_eels-SI_004.dm4')#('pyfiles/area03-eels-SI-aligned.dm4')
im.plot_sum()

for i in [5]:#[3,4,5,10]:
    im.cluster(n_clusters = i)
    plt.figure()
    plt.title("spectral image, clustered with " + str(i) + " clusters")
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    xticks, yticks = im.get_ticks()
    ax = sns.heatmap(im.clustered, xticklabels=xticks, yticklabels=yticks)
    plt.show()


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
    