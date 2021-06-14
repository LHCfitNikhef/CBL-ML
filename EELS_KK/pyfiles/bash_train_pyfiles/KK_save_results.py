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
import warnings
import traceback

path_to_models = sys.argv[1]
path_to_save = sys.argv[2]
# path_to_models = 'models/dE2_3_times_dE1/train_004_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
# path_to_save = '../../KK_results/004_clu10_pooled_5/'
# path_to_save = '../../KK_results/004_clu10_pooled_5'
path_to_save += (path_to_save[-1] != '/')*'/'




path = '/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4'
path = '/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/area03-eels-SI-aligned.dm4'

# path = '../../dmfiles/h-ws2_eels-SI_004.dm4'
im = Spectral_image.load_data(path)
[n_x, n_y] = im.image_shape




ieels_im = np.zeros(np.append(np.append(im.image_shape, 3), im.l))
ieels_p_im = np.zeros(np.append(np.append(im.image_shape, 3), im.l))
ss_im = np.zeros(np.append(np.append(im.image_shape, 3), np.sum(im.deltaE>0)))
ssratio_im = np.zeros(np.append(im.image_shape,3))
eps_im = (1+1j)*np.zeros(np.append(np.append(im.image_shape, 3), np.sum(im.deltaE>0)))

t_im = np.zeros(np.append(im.image_shape,3))
E_cross_im = np.zeros(im.image_shape, dtype = 'object')
n_cross_im = np.zeros(np.append(im.image_shape,3))
E_band_im = np.zeros(np.append(im.image_shape,3))
E_band_sscor_im = np.zeros(np.append(im.image_shape,3))
# b_im = np.zeros(np.append(im.image_shape,3))
A_im = np.zeros(np.append(im.image_shape,3))
max_ieels_im = np.zeros(np.append(im.image_shape,3))

for i in range(im.image_shape[0]):
    # recv_dict = comm.recv(source=i)
    path_dict = path_to_save + 'row_dicts/row_dict_' + str(i) + '.p'
    try:
        with open(path_dict, 'rb') as fp:
            row_dict = pickle.load(fp)
        ieels_im[i,:,:,:] = row_dict["ieels"]
        ieels_p_im[i,:,:,:] = row_dict["ieels_p"]
        ss_im[i,:,:,:] = row_dict["ss"]
        ssratio_im[i,:,:] = row_dict["ssratio"]
        eps_im[i,:,:,:] = row_dict["eps"]
        t_im[i,:,:] = row_dict["t"]
        E_cross_im[i,:] = row_dict["E_cross"]
        n_cross_im[i,:,:] = row_dict["n_cross"]
        E_band_im[i,:,:] = row_dict["E_band"]
        E_band_sscor_im[i,:,:] = row_dict["E_band_sscor"]
        # b_im[i,:,:] = row_dict["b"]
        A_im[i,:,:] = row_dict["A"]
        max_ieels_im[i,:,:] = row_dict["max_ieels"]
        # os.remove(path_dict)
        print("succesfully read dict " + str(i))
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        warnings.warn("Something went wrong on dict " + str(i) + ": " + str(type(e).__name__) + ": " + str(e))
        pass
im.ieels = ieels_im
im.ieels_p = ieels_p_im
im.ss = ss_im
im.ssratio = ssratio_im
im.eps = eps_im
im.t = t_im
im.E_cross = E_cross_im
im.n_cross = n_cross_im
im.E_band = E_band_im
im.E_band_sscor = E_band_sscor_im
# im.b = b_im
im.A = A_im
im.max_ieels = max_ieels_im


path_to_save_sum = path_to_save + "summary/"
if not os.path.exists(path_to_save_sum):
    os.mkdir(path_to_save_sum)

with open(path_to_save_sum+"ieels.npy", 'wb') as f:
    np.save(f, ieels_im)
with open(path_to_save_sum+"ieels_p.npy", 'wb') as f:
    np.save(f, ieels_p_im)
with open(path_to_save_sum+"ss.npy", 'wb') as f:
    np.save(f, ss_im)
with open(path_to_save_sum+"ssratio.npy", 'wb') as f:
    np.save(f, ssratio_im)
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
with open(path_to_save_sum+"E_band_sscor.npy", 'wb') as f:
    np.save(f, E_band_sscor_im)
# with open(path_to_save_sum+"b_band.npy", 'wb') as f:
#     np.save(f, b_im)
with open(path_to_save_sum+"A_band.npy", 'wb') as f:
    np.save(f, A_im)
with open(path_to_save_sum+"max_ieels.npy", 'wb') as f:
    np.save(f, max_ieels_im)
    

#TODO: check exsits --> other path
path_to_save_im = path_to_save + "image_KK.pkl"
i = 2
while os.path.exists(path_to_save_im):
    path_to_save_im = path_to_save + "image_KK_" + str(i) + ".pkl" 
    i += 1
im.save_image(path_to_save_im)