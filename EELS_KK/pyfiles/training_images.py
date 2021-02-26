#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:52:45 2021

@author: isabel
"""
import matplotlib.pyplot as plt
import seaborn as sns
from image_class import Spectral_image
from train_nn_torch import train_nn_scaled


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


train_nn_scaled(im, path_to_model = "train_004", lr = 1e-3, n_epochs=30000)
