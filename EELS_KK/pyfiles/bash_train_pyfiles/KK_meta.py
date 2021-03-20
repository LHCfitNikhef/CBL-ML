#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:49:58 2021

@author: isabel
"""
import numpy as np
import sys
import os

from image_class_bs import Spectral_image

path_to_models = sys.argv[1]
path_to_save = sys.argv[2]
smooth = sys.argv[3]


im = Spectral_image.load_data("path")
im.cluster(5)
im.load_ZLP_models_smefit(path_to_models)
if smooth:
    im.smooth(window_len=50)

[n_x, n_y] = im.image_shape


eps = np.zeros(np.append(3,np.append(im.image_shape, im.deltaE>0)))
t = np.zeros(np.append(3,im.image_shape))
E_cross = np.zeros(np.append(3,im.image_shape))
n_cross = np.zeros(np.append(3,im.image_shape))
E_band = np.zeros(np.append(3,im.image_shape))


bash_file = "submit_KK_pixel.sh" #OFZO WEET NOG NIET HOE DIT WERKT

for i in range(n_x):
    for j in range(n_y):
        results = bash_file(im, i, j)
        eps[:,i,j,:] = results[0]
        t[:,i,j] = results[1]
        E_cross[:,i,j] = results[2]
        n_cross[:,i,j] = results[3]
        E_band[:,i,j] = results[4]


path_to_save += (path_to_save[0] != '/')*'/'
if not os.path.exist(path_to_save):
    os.mkdir(path_to_save)

with open("summary/eps.npy", 'wb') as f:
    np.save(f, eps)
with open("summary/t.npy", 'wb') as f:
    np.save(f, t)
with open("summary/E_cross.npy", 'wb') as f:
    np.save(f, E_cross)
with open("summary/n_cross.npy", 'wb') as f:
    np.save(f, n_cross)
with open("summary/E_band.npy", 'wb') as f:
    np.save(f, E_band)




