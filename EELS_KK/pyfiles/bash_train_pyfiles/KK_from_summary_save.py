#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:34:37 2021

@author: isabel
"""
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

im.load_ZLP_models_smefit(path_to_models)



path_to_save_sum = path_to_save + "summary/"
# im.ieels = np.load(path_to_save_sum+"ieels.npy")
im.eps = np.load(path_to_save_sum+"eps.npy")
im.t = np.load(path_to_save_sum+"t.npy")
im.E_cross = np.load(path_to_save_sum+"E_cross.npy", allow_pickle=True)
im.n_cross = np.load(path_to_save_sum+"n_cross.npy")
im.E_band = np.load(path_to_save_sum+"E_band.npy")
# im.b = np.load(path_to_save_sum+"b_band.npy")
im.max_ieels = np.load(path_to_save_sum+"max_ieels.npy")
im.ieels_p = np.load(path_to_save_sum+"ieels_p.npy")


#TODO: check exsits --> other path
path_to_save_im = path_to_save + "image_KK" #".pkl"
i = 2
while os.path.exists(path_to_save_im):
    path_to_save_im = path_to_save + "image_KK_" + str(i) #+ ".pkl" 
    i += 1
#im.save_image(path_to_save_im)
im.save_compressed_image(path_to_save_im)