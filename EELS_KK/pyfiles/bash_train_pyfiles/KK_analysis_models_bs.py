#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:57:24 2021

@author: isabel
"""
import sys


from image_class_bs import Spectral_image
import seaborn as sns



bs_rep_num = int(sys.argv[1])
path_to_models = sys.argv[2]
path_to_save = sys.argv[3]

try:
    smooth = bool(int(sys.argv[4]))
except:
    smooth = False  


# bs_rep_num = 1
# path_to_models = "./dE1/E1_05"
# path_to_save = "./dE1"
# im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')

im = Spectral_image.load_data('/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4')

if not im.check_cost_smefit(path_to_models, bs_rep_num):
    sys.exit()

im.cluster(5)
im.load_zlp_models(path_to_models, idx = bs_rep_num)

im.im_dielectric_function_bs(save_index = bs_rep_num, save_path = path_to_save)




