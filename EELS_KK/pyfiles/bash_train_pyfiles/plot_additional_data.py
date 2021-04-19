#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:39:37 2021

@author: isabel
"""
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from image_class_bs import Spectral_image


#PLOT ADDITIONAL DATA

im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_003.dm4', load_additional_data=True)
save_loc = "../../plots/plots_symposium/003/"
cmap="coolwarm"

for i in range(len(im.additional_data)):
    dataset = im.additional_data[i]
    if dataset['data'].ndim == 2:
        im.pixelsize = dataset['pixelSize']
        im.calc_axes(image_shape = dataset['data'].shape)
        im.plot_heatmap(dataset['data'], title='STEM intensity sample', cbar_kws={'label':  'arb. units'}, cmap = cmap, save_as = save_loc + "STEM_image_" + str(i))