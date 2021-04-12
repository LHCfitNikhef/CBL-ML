#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:22:35 2021

@author: isabel
"""
#PLOTTING RESULTS
import numpy as np
import sys
import os
import pickle
from image_class_bs import Spectral_image

path_to_models = sys.argv[1]
path_to_save = sys.argv[2]
path_to_save += (path_to_save[-1] != '/')*'/'


path = '/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4'
im = Spectral_image.load_data(path)
[n_x, n_y] = im.image_shape




ieels_im = (1+1j)*np.zeros(np.append(np.append(im.image_shape, 3), im.l))
eps_im = (1+1j)*np.zeros(np.append(np.append(im.image_shape, 3), np.sum(im.deltaE>0)))
t_im = np.zeros(np.append(im.image_shape,3))
E_cross_im = np.zeros(im.image_shape, dtype = 'object')
n_cross_im = np.zeros(np.append(im.image_shape,3))
E_band_im = np.zeros(np.append(im.image_shape,3))
b_im = np.zeros(np.append(im.image_shape,3))


for i in range(im.image_shape[0]):
    # recv_dict = comm.recv(source=i)
    path_dict = path_to_save + 'row_dicts/row_dict_' + str(i) + '.p'
    try:
        with open(path_dict, 'rb') as fp:
            row_dict = pickle.load(fp)
        ieels_im[i,:,:,:] = row_dict["ieels"]
        eps_im[i,:,:,:] = row_dict["eps"]
        t_im[i,:,:] = row_dict["t"]
        E_cross_im[i,:] = row_dict["E_cross"]
        n_cross_im[i,:,:] = row_dict["n_cross"]
        E_band_im[i,:,:] = row_dict["E_band"]
        b_im[i,:,:] = row_dict["b"]
        os.remove(path_dict)
    except:
        pass
im.ieels = ieels_im
im.eps = eps_im
im.t = t_im
im.E_cross = E_cross_im
im.n_cross = n_cross_im
im.E_band = E_band_im
im.b = b_im


path_to_save_sum = path_to_save + "summary/"
if not os.path.exists(path_to_save_sum):
    os.mkdir(path_to_save_sum)

with open(path_to_save_sum+"ieels.npy", 'wb') as f:
    np.save(f, ieels_im)
with open(path_to_save_sum+"eps.npy", 'wb') as f:
    np.save(f, eps_im)
with open(path_to_save_sum+"t.npy", 'wb') as f:
    np.save(f, t_im)
with open(path_to_save_sum+"E_cross.npy", 'wb') as f:
    np.save(f, E_cross_im)
with open(path_to_save_sum+"n_cross.npy", 'wb') as f:
    np.save(f, n_cross_im)
with open(path_to_save_sum+"E_band.npy", 'wb') as f:
    np.save(f, E_band_im)
with open(path_to_save_sum+"b_band.npy", 'wb') as f:
    np.save(f, b_im)

#TODO: check exsits --> other path
path_to_save_im = path_to_save + "image_KK.pkl"
i = 2
while os.path.exists(path_to_save_im):
    path_to_save_im = path_to_save + "image_KK_" + str(i) + ".pkl" 
    i += 1
im.save_image(path_to_save_im)