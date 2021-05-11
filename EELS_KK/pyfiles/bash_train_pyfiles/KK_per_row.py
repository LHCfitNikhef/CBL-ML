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

def bandgap_b(x, amp, BG):
    b = 1.5 #standard value according to ..... #TODO
    result = np.zeros(x.shape)
    result[x<BG] = 1
    result[x>=BG] = amp * (x[x>=BG] - BG)**(b)
    return result

def find_pixel_cordinates(idx, n_y):
    y = int(idx%n_y)
    x = int((idx-y)/n_y)
    return [x,y]

def pool(im, n_p):
    pooled = np.zeros((im.image_shape[0]-n_p, im.image_shape[1]-n_p, im.shape[2]))
    for i in range(im.image_shape[0]-n_p+1):
        for j in range(im.image_shape[1]-n_p+1):
            pooled[i,j] = np.average(np.average(im.ieels[i:i+n_p,j:j+n_p,0],axis=1), axis=0)
    return pooled


path_to_models = sys.argv[1]
path_to_save = sys.argv[2]
row = int(sys.argv[3])
path_to_save += (path_to_save[-1] != '/')*'/'

if not os.path.exists(path_to_save):
    try:
        os.mkdir(path_to_save)
    except:
        pass


path = '/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4'
im = Spectral_image.load_data(path)
[n_x, n_y] = im.image_shape

dia_pooled = 5


if row >= n_x:
    sys.exit()


# im.cluster(5)
im.load_ZLP_models_smefit(path_to_models)
im.dE1[1,0] -= 0.2
im.set_n(4.1462, n_vac = 2.1759)
im.e0 = 200 #keV
im.beta = 67.2 #mrad


ieels = np.zeros((im.image_shape[1],3,im.l))
ieels_p = np.zeros((im.image_shape[1],3,im.l))
eps = (1+1j)*np.zeros((im.image_shape[1],3, np.sum(im.deltaE>0)))
t = np.zeros((im.image_shape[1],3))
E_cross = np.zeros(im.image_shape[1], dtype = 'object')
# E_cross = np.zeros((im.image_shape[1],1,3))
n_cross = np.zeros((im.image_shape[1],3))
E_band = np.zeros((im.image_shape[1],3))
# b = np.zeros((im.image_shape[1],3))
A = np.zeros((im.image_shape[1],3))
max_ieels = np.zeros((im.image_shape[1],3))

# n_model = len(im.ZLP_models)
n_fails = 0

im.pool(dia_pooled)

#TOD max vinden 
#TODO opslaan fiksen nu geen zin in

for j in range(n_y):
    #epss, ts, S_Es, IEELSs = im.KK_pixel(row, j, signal = "pooled")
    [ts, IEELSs, max_ieelss], [epss, ts_p, S_ss_p, IEELSs_p, max_ieels_p] = im.KK_pixel(row, j, signal = "pooled")
    n_model = len(IEELSs_p)
    E_bands = np.zeros(n_model)
    # bs = np.zeros(n_model)
    As = np.zeros(n_model)

    E_cross_pix = np.empty(0)
    n_cross_pix = np.zeros(n_model)
    for i in range(n_model):
        IEELS = IEELSs_p[i]
        try:
            cluster = im.clustered[row,j]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+0.2
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            # popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)], p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
            As[i] = popt[0]
            E_bands[i] = popt[1]
            # bs[i] = popt[2]
        except:
            n_fails += 1
            print("fail nr.: ", n_fails, "failed curve-fit, row: ", row, ", pixel: ", j, ", model: ", i)
            E_bands[i] = 0
            # bs[i] = 0
        
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
    ieels[j,:,:] = summary_distribution(IEELSs)
    ieels_p[j,:,:] = summary_distribution(IEELSs_p)
    eps[j,:,:] = summary_distribution(epss)
    t[j,:] = summary_distribution(ts)
    E_cross[j] = E_cross_pix_n
    n_cross[j,:] = summary_distribution(n_cross_pix)
    E_band[j,:] = summary_distribution(E_bands)
    # b[j,:] = summary_distribution(bs)
    A[j,:] = summary_distribution(As)
    max_ieels[j,:] = summary_distribution(max_ieelss)


    
# np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_0", save_array)
# print(save_array)

save_dict = {"ieels": ieels, "ieels_p": ieels_p, "eps":eps, "t":t, "E_cross":E_cross, \
             "n_cross":n_cross, "E_band":E_band, "A":A, "max_ieels": max_ieels}
# comm.send(send_dict, dest=0)
path_to_save += "row_dicts/"
if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)
with open(path_to_save + 'row_dict_' + str(row) + '.p', 'wb') as fp:
    pickle.dump(save_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)






