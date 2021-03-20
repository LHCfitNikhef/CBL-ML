#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:35:13 2021

@author: isabel
"""

import numpy as np
import sys
import os
from scipy.optimize import curve_fit
from k_means_clustering import k_means
import pickle

from image_class_bs import Spectral_image, smooth_1D

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

def find_pixel_cordinates(idx, n_y):
    y = int(idx%n_y)
    x = int((idx-y)/n_y)
    return [x,y]

path_to_models = sys.argv[1]
path_to_save = sys.argv[2]
row = int(sys.argv[3])
path_to_save += (path_to_save[-1] != '/')*'/'


path = '/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4'
im = Spectral_image.load_data(path)
[n_x, n_y] = im.image_shape

if row != 47:#>= n_x:
    sys.exit()


im.cluster(5)
im.load_ZLP_models_smefit(path_to_models, n_rep=500, name_in_path = False)




eps = (1+1j)*np.zeros((im.image_shape[1],3, np.sum(im.deltaE>0)))
t = np.zeros((im.image_shape[1],3))
E_cross = np.zeros(im.image_shape[1], dtype = 'object')
#E_cross = np.zeros((im.image_shape[1],1,3))
n_cross = np.zeros((im.image_shape[1],3))
E_band = np.zeros((im.image_shape[1],3))
b = np.zeros((im.image_shape[1],3))

n_model = len(im.ZLP_models)
n_fails = 0

for j in range(n_y):
    epss, ts, S_Es, IEELSs = im.KK_pixel(row, j)
    E_bands = np.zeros(n_model)
    bs = np.zeros(n_model)

    E_cross_pix = np.empty(0)
    n_cross_pix = np.zeros(n_model)
    for i in range(n_model):
        IEELS = IEELSs[i]
        try:
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>0.5) & (im.deltaE<3.7)], IEELS[(im.deltaE>0.5) & (im.deltaE<3.7)], p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands[i] = popt[1]
            bs[i] = popt[2]
        except:
            n_fails += 1
            print("fail nr.: ", n_fails, "failed curve-fit, row: ", row, ", pixel: ", j, ", model: ", i)
            E_bands[i] = 0
            bs[i] = 0
        
        crossing = np.concatenate((np.array([0]),(smooth_1D(np.real(epss[i]),50)[:-1]<0) * (smooth_1D(np.real(epss[i]),50)[1:] >=0)))
        deltaE_n = im.deltaE[im.deltaE>0]
        #deltaE_n = deltaE_n[50:-50]
        crossing_E = deltaE_n[crossing.astype('bool')]
        
        E_cross_pix = np.append(E_cross_pix, crossing_E)
        n_cross_pix[i] = len(crossing_E)
    
    n_cross_lim = round(np.nanpercentile(n_cross_pix, 90))
    # if n_cross_lim > E_cross.shape[1]:
    #     E_cross_new = np.zeros((im.image_shape[1],n_cross_lim,3))
    #     E_cross_new[:,:E_cross.shape[1],:] = E_cross
    #     E_cross = E_cross_new
    #     del E_cross_new
    
    if n_cross_lim > 0:
        E_cross_pix_n, r = k_means(E_cross_pix, n_cross_lim)
        E_cross_pix_n = np.zeros((n_cross_lim, 3))
        for i in range(n_cross_lim):
            E_cross_pix_n[i, :] = summary_distribution(E_cross_pix[r[i].astype(bool)])
            #E_cross[j,i,:] = summary_distribution(E_cross_pix[r[i].astype(bool)])
    
    else: 
        E_cross_pix_n = np.zeros((0))
    
    eps[j,:,:] = summary_distribution(epss)
    t[j,:] = summary_distribution(ts)
    E_cross[j] = E_cross_pix_n
    n_cross[j,:] = summary_distribution(n_cross_pix)
    E_band[j,:] = summary_distribution(E_bands)
    b[j,:] = summary_distribution(bs)


    
# np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_0", save_array)
# print(save_array)

save_dict = {"eps":eps, "t":t, "E_cross":E_cross, "n_cross":n_cross, "E_band":E_band, "b":b}
# comm.send(send_dict, dest=0)
path_to_save += "row_dicts/"
if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)
with open(path_to_save + 'row_dict_' + str(row) + '.p', 'wb') as fp:
    pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)






