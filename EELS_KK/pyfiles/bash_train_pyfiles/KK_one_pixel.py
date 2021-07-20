#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 22:59:40 2021

@author: isabel
"""
import numpy as np
import sys
import os
from mpi4py import MPI
from scipy.optimize import curve_fit
from k_means_clustering import k_means
import matplotlib.pyplot as plt


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
    result[x<BG] = 1
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

"""
path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/models/train_004"
# im = im
im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
im.cluster(5)

#rank = 84
rank = 64
j = 80
im.load_zlp_models(path_to_models=path_to_models, n_rep = 500)
ZLPs = im.calc_zlps(rank,j,path_to_models=path_to_models, n_rep = 500)
n_model = len(im.ZLP_models)
epss, ts, S_Es, IEELSs = im.KK_pixel(rank, j)

E_cross = np.zeros(im.image_shape[1], dtype = 'object')


E_bands = np.zeros(n_model)
bs = np.zeros(n_model)
E_cross_pix = np.empty(0)
n_cross_pix = np.zeros(n_model)
for i in range(n_model):
    IEELS = IEELSs[i]
    popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>0.5) & (im.deltaE<3.7)], IEELS[(im.deltaE>0.5) & (im.deltaE<3.7)], p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
    E_bands[i] = popt[1]
    bs[i] = popt[2]
    
    crossing = np.concatenate((np.array([0]),(smooth_1D(np.real(epss[i]),50)[:-1]<0) * (smooth_1D(np.real(epss[i]),50)[1:] >=0)))
    deltaE_n = im.deltaE[im.deltaE>0]
    #deltaE_n = deltaE_n[50:-50]
    crossing_E = deltaE_n[crossing.astype('bool')]
    
    E_cross_pix = np.append(E_cross_pix, crossing_E)
    n_cross_pix[i] = len(crossing_E)

n_cross_lim = round(np.nanpercentile(n_cross_pix, 90))
# if n_cross_lim > E_cross.shape[1]:
#     E_cross_new = np.zeros((im.image_shape[1],n_cross_lim,3))
#     E_cross_new[:,E_cross.shape[1],:] = E_cross
#     E_cross = E_cross_new
#     del E_cross_new
E_cross_pix_n, r = k_means(E_cross_pix, n_cross_lim)
E_cross_pix_n = np.zeros((n_cross_lim, 3))
for i in range(n_cross_lim):
    E_cross_pix_n[i, :] = summary_distribution(E_cross_pix[r[i].astype(bool)])
    # E_cross[j,i,:] = summary_distribution(E_cross_pix[r[i].astype(bool)])


# eps[j,:,:] = summary_distribution(epss)
# t[j,:] = summary_distribution(ts)
E_cross[j] = E_cross_pix_n
# n_cross[j,:] = summary_distribution(n_cross_pix)
# E_band[j] = summary_distribution(E_bands)




"""

path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/models/train_004"
path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/dE1/E1_05/"
path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/models/dE2_8_times_dE1/train_004_not_pooled"
path_to_models = 'models/dE2_3_times_dE1/train_004_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_10/'
path_to_models = 'models/dE2_3_times_dE1/train_lau_pooled_5_CI_1_dE1_times_07_epochs_1e6_scale_on_pooled_clu_log_5/'

path_to_models = 'models/dE_n10-inse_SI-003/E1_05'
path_to_models = 'models/report/004_clu10_p5_final_35dE1_06dE1/'

# path_to_models = "/Users/isabel/Documents/Studie/MEP/CBL-ML/EELS_KK/pyfiles/bash_train_pyfiles/models/train_lau_log"


# im = im
im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
# im = Spectral_image.load_data('../../dmfiles/area03-eels-SI-aligned.dm4')
# im = Spectral_image.load_data('../../dmfiles/10n-dop-inse-B1_stem-eels-SI-processed_003.dm4')

name = ""
# name = "Lau's sample, "

[n_x, n_y] = im.image_shape

row = 50
j_b = 50

if row >= n_x:
    sys.exit()


im.load_zlp_models(path_to_models, name_in_path = False)

im.set_n(4.1462, n_background = 2.1759)
im.set_n(3.0)
im.e0 = 200 #keV
im.beta = 30#67.2 #mrad
#%%
try_pixels = [30,60,90,120]

ieels = np.zeros((len(try_pixels),3,im.l))
eps = (1+1j)*np.zeros((len(try_pixels),3, np.sum(im.deltaE>0)))
ss = np.zeros((len(try_pixels),3, np.sum(im.deltaE>0)))
ssratio = np.zeros((len(try_pixels),3))
t = np.zeros((len(try_pixels),3))
E_cross = np.zeros(len(try_pixels), dtype = 'object')
n_cross = np.zeros((len(try_pixels),3))
max_ieels = np.zeros((len(try_pixels),3))


E_band_08 = np.zeros((len(try_pixels),3))
A_08 = np.zeros((len(try_pixels),3))

E_band_12 = np.zeros((len(try_pixels),3))
A_12 = np.zeros((len(try_pixels),3))

E_band_02 = np.zeros((len(try_pixels),3))
A_02 = np.zeros((len(try_pixels),3))

E_band_04 = np.zeros((len(try_pixels),3))
A_04 = np.zeros((len(try_pixels),3))

E_band_06 = np.zeros((len(try_pixels),3))
A_06 = np.zeros((len(try_pixels),3))


n_model = len(im.ZLP_models)
n_fails = 0
#%%
dia_pooled = 5
im.pool(dia_pooled)

#%%



for j in range(len(try_pixels)):# n_y):
    col = try_pixels[j]
    [ts, IEELSs, max_ieelss], [epss, ts_p, S_ss_p, IEELSs_p, max_ieels_p] = im.KK_pixel(col,row,  signal = "pooled")
    
    
    
    
    
    
    
    
    n_model = len(IEELSs_p)
    E_bands_08 = np.zeros(n_model)
    E_bands_12 = np.zeros(n_model)
    E_bands_02 = np.zeros(n_model)
    E_bands_04 = np.zeros(n_model)
    E_bands_06 = np.zeros(n_model)

    As_08 = np.zeros(n_model)
    As_12 = np.zeros(n_model)
    As_02 = np.zeros(n_model)
    As_04 = np.zeros(n_model)
    As_06 = np.zeros(n_model)

    E_cross_pix = np.empty(0)
    n_cross_pix = np.zeros(n_model)
    for i in range(n_model):
        IEELS = IEELSs_p[i]
        boundery_onset = np.max(IEELS)/30
        first_not_onset = np.min(np.argwhere(IEELS>boundery_onset))
        range2 = im.deltaE[first_not_onset]
        if i%100 == 0:
            print("pixel:",j, ", model:", i, ", boundery onset:", boundery_onset,", range2:", range2)
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            # range2 = dE1+0.8
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,range2-0.3], bounds=([0, 0.5],np.inf))
            E_bands_08[i] = popt[1]
            As_08[i] = popt[0]
        except:
            n_fails += 1
        # try:
        #     cluster = im.clustered[col,row]
        #     dE1 = im.dE1[1,int(cluster)]
        #     range1 = dE1-1
        #     range2 = dE1+1.2
        #     baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
        #     popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
        #     E_bands_12[i] = popt[1]
        #     As_12[i] = popt[0]
        # except:
        #     n_fails += 1
        # try:
        #     cluster = im.clustered[col,row]
        #     dE1 = im.dE1[1,int(cluster)]
        #     range1 = dE1-1
        #     range2 = dE1+0.2
        #     baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
        #     popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
        #     As_02[i] = popt[0]
        #     E_bands_02[i] = popt[1]
        # except:
        #     n_fails += 1
        # try:
        #     cluster = im.clustered[col,row]
        #     dE1 = im.dE1[1,int(cluster)]
        #     range1 = dE1-1
        #     range2 = dE1+0.4
        #     baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
        #     popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
        #     As_04[i] = popt[0]
        #     E_bands_04[i] = popt[1]
        # except:
        #     n_fails += 1
        # try:
        #     cluster = im.clustered[col,row]
        #     dE1 = im.dE1[1,int(cluster)]
        #     range1 = dE1-1
        #     range2 = dE1+0.6
        #     baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
        #     popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
        #     As_06[i] = popt[0]
        #     E_bands_06[i] = popt[1]
        # except:
        #      n_fails += 1
        
        crossing = np.concatenate((np.array([0]),(smooth_1D(np.real(epss[i]),50)[:-1]<0) * (smooth_1D(np.real(epss[i]),50)[1:] >=0)))
        deltaE_n = im.deltaE[im.deltaE>0]
        crossing_E = deltaE_n[crossing.astype('bool')]
        
        E_cross_pix = np.append(E_cross_pix, crossing_E)
        n_cross_pix[i] = len(crossing_E)
    
    n_cross_lim = round(np.nanpercentile(n_cross_pix, 90))
    
    if n_cross_lim > 0:
        E_cross_pix_n, r = k_means(E_cross_pix, n_cross_lim)
        E_cross_pix_n = np.zeros((n_cross_lim, 3))
        for i in range(n_cross_lim):
            E_cross_pix_n[i, :] = summary_distribution(E_cross_pix[r[i].astype(bool)])
    
    else: 
        E_cross_pix_n = np.zeros((0))
        
        
    ieelssums = np.sum(IEELSs_p[:,im.deltaE>0], axis = 1)
    sssums = np.sum(S_ss_p, axis = 1)
    s_ratios = sssums/ieelssums
        
    
    ieels[j,:,:] = summary_distribution(IEELSs_p)
    ss[j,:,:] = summary_distribution(S_ss_p)
    ssratio[j,:] = summary_distribution(s_ratios)
    eps[j,:,:] = summary_distribution(epss)
    t[j,:] = summary_distribution(ts)
    E_cross[j] = E_cross_pix_n
    n_cross[j,:] = summary_distribution(n_cross_pix)
    max_ieels[j,:] = summary_distribution(max_ieelss)
    E_band_08[j,:] = summary_distribution(E_bands_08)
    A_08[j,:] = summary_distribution(As_08)
    E_band_12[j,:] = summary_distribution(E_bands_12)
    A_12[j,:] = summary_distribution(As_12)
    E_band_02[j,:] = summary_distribution(E_bands_02)
    A_02[j,:] = summary_distribution(As_02)
    E_band_04[j,:] = summary_distribution(E_bands_04)
    A_04[j,:] = summary_distribution(As_04)
    E_band_06[j,:] = summary_distribution(E_bands_06)
    A_06[j,:] = summary_distribution(As_06)


#%%
CI_E_band08 = (E_band_08[:,2]-E_band_08[:,1])/E_band_08[:,0]
CI_E_band12 = (E_band_12[:,2]-E_band_12[:,1])/E_band_12[:,0]
CI_E_band02 = (E_band_02[:,2]-E_band_02[:,1])/E_band_02[:,0]
CI_E_band04 = (E_band_04[:,2]-E_band_04[:,1])/E_band_04[:,0]
CI_E_band06 = (E_band_06[:,2]-E_band_06[:,1])/E_band_06[:,0]

E_bands = np.vstack((E_band_08[:,0],E_band_12[:,0],E_band_02[:,0],E_band_04[:,0],E_band_06[:,0]))
As = np.vstack((A_08[:,0],A_12[:,0],A_02[:,0],A_04[:,0],A_06[:,0]))


CI_E_band = np.vstack((CI_E_band08, CI_E_band12, CI_E_band02, CI_E_band04, CI_E_band06 ))


plt.figure()
plt.title(name + "relative CI E_band, dia_pool = " +str(dia_pooled))
plt.imshow(CI_E_band)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

#%%
plt.figure()
plt.title(name + "relative CI E_band, capped at 1, dia_pool = " +str(dia_pooled))
plt.imshow(np.minimum(CI_E_band,1))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()
#%%

plt.figure()
plt.title(name + "E_band, dia_pool = " +str(dia_pooled))
plt.imshow(E_bands)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()


plt.figure()
plt.title(name + "E_band, dia_pool = " +str(dia_pooled) + ", capped at 3")
plt.imshow((np.minimum(E_bands, 3)))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

#%%

ranges = [0.8,1.2,0.2,0.4,0.6]

for j in range(len(try_pixels)):
    cluster = im.clustered[try_pixels[j],row]
    xmin = im.dE1[1,cluster] - 1
    xmax = im.dE1[1,cluster] + 3.5
    baseline = np.average(ieels[j,0,(im.deltaE>xmin -0.1) & (im.deltaE<xmin)])

    ymax = np.max(ieels[j,0,(im.deltaE>xmin)&(im.deltaE<xmax)]) + 100
    x_values = im.deltaE[(im.deltaE>xmin)&(im.deltaE<xmax)]
    plt.figure()
    plt.title(name + "bandgap fits pixel [64," + str(try_pixels[j]) + "], pooled with diameter " + str(dia_pooled))
    plt.plot(x_values, ieels[j,0,(im.deltaE>xmin)&(im.deltaE<xmax)]-baseline, lw = 1.5)
    plt.fill_between(x_values, ieels[j,1,(im.deltaE>xmin)&(im.deltaE<xmax)] - baseline, ieels[j,2,(im.deltaE>xmin)&(im.deltaE<xmax)]- baseline, alpha=0.3)
    plt.vlines(im.dE1[1,cluster],0,500, linestyle = '--', color= 'black', label = 'dE1')
    for i in range(len(E_bands)):
        plt.plot(x_values, bandgap_b(x_values, As[i,j], E_bands[i,j]), label = str(ranges[i]))
    plt.legend()
    plt.ylim(-50,ymax)





#%% DEEL TWEE: met b fitten
E_band_08 = np.zeros((len(try_pixels),3))
b_08 = np.zeros((len(try_pixels),3))
A_08 = np.zeros((len(try_pixels),3))

E_band_12 = np.zeros((len(try_pixels),3))
b_12 = np.zeros((len(try_pixels),3))
A_12 = np.zeros((len(try_pixels),3))

E_band_02 = np.zeros((len(try_pixels),3))
b_02 = np.zeros((len(try_pixels),3))
A_02 = np.zeros((len(try_pixels),3))

E_band_04 = np.zeros((len(try_pixels),3))
b_04 = np.zeros((len(try_pixels),3))
A_04 = np.zeros((len(try_pixels),3))

E_band_06 = np.zeros((len(try_pixels),3))
b_06 = np.zeros((len(try_pixels),3))
A_06 = np.zeros((len(try_pixels),3))




for j in range(len(try_pixels)):# n_y):
    # epss, ts, S_Es, IEELSs = im.KK_pixel(row, j, signal = "pooled")
    col = try_pixels[j]
    [ts, IEELSs, max_ieelss], [epss, ts_p, S_ss_p, IEELSs_p, max_ieels_p] = im.KK_pixel(col,row,  signal = "pooled", iterations = 5)
    
    n_model = len(IEELSs_p)
    E_bands_08 = np.zeros(n_model)
    E_bands_12 = np.zeros(n_model)
    E_bands_02 = np.zeros(n_model)
    E_bands_04 = np.zeros(n_model)
    E_bands_06 = np.zeros(n_model)

    bs_08 = np.zeros(n_model)
    bs_12 = np.zeros(n_model)
    bs_02 = np.zeros(n_model)
    bs_04 = np.zeros(n_model)
    bs_06 = np.zeros(n_model)
    
    As_08 = np.zeros(n_model)
    As_12 = np.zeros(n_model)
    As_02 = np.zeros(n_model)
    As_04 = np.zeros(n_model)
    As_06 = np.zeros(n_model)

    E_cross_pix = np.empty(0)
    n_cross_pix = np.zeros(n_model)
    for i in range(n_model):
        IEELS = IEELSs_p[i]
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+0.8
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands_08[i] = popt[1]
            bs_08[i] = popt[2]
            # popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
            # E_bands_08[i] = popt[1]
            As_08[i] = popt[0]
        except:
            n_fails += 1
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+1.2
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands_12[i] = popt[1]
            bs_12[i] = popt[2]
            # popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
            # E_bands_12[i] = popt[1]
            As_12[i] = popt[0]
        except:
            n_fails += 1
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+0.2
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands_02[i] = popt[1]
            bs_02[i] = popt[2]
            # popt, pcov = curve_fit(bandgap_b, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5], bounds=([0, 0.5],np.inf))
            As_02[i] = popt[0]
            E_bands_02[i] = popt[1]
        except:
            n_fails += 1
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+0.4
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands_04[i] = popt[1]
            bs_04[i] = popt[2]
            As_04[i] = popt[0]
        except:
            n_fails += 1
        try:
            cluster = im.clustered[col,row]
            dE1 = im.dE1[1,int(cluster)]
            range1 = dE1-1
            range2 = dE1+0.6
            baseline = np.average(IEELS[(im.deltaE>range1 -0.1) & (im.deltaE<range1)])
            popt, pcov = curve_fit(bandgap, im.deltaE[(im.deltaE>range1) & (im.deltaE<range2)], IEELS[(im.deltaE>range1) & (im.deltaE<range2)]-baseline, p0 = [400,1.5,0.5], bounds=([0, 0.5, 0],np.inf))
            E_bands_06[i] = popt[1]
            bs_06[i] = popt[2]
            As_06[i] = popt[0]
            E_bands_06[i] = popt[1]
        except:
             n_fails += 1
        
        crossing = np.concatenate((np.array([0]),(smooth_1D(np.real(epss[i]),50)[:-1]<0) * (smooth_1D(np.real(epss[i]),50)[1:] >=0)))
        deltaE_n = im.deltaE[im.deltaE>0]
        crossing_E = deltaE_n[crossing.astype('bool')]
        
        E_cross_pix = np.append(E_cross_pix, crossing_E)
        n_cross_pix[i] = len(crossing_E)
    
    n_cross_lim = round(np.nanpercentile(n_cross_pix, 90))
    if n_cross_lim > 0:
        E_cross_pix_n, r = k_means(E_cross_pix, n_cross_lim)
        E_cross_pix_n = np.zeros((n_cross_lim, 3))
        for i in range(n_cross_lim):
            E_cross_pix_n[i, :] = summary_distribution(E_cross_pix[r[i].astype(bool)])
    
    else: 
        E_cross_pix_n = np.zeros((0))
    ieels[j,:,:] = summary_distribution(IEELSs_p)
    eps[j,:,:] = summary_distribution(epss)
    t[j,:] = summary_distribution(ts)
    E_cross[j] = E_cross_pix_n
    n_cross[j,:] = summary_distribution(n_cross_pix)
    max_ieels[j,:] = summary_distribution(max_ieelss)
    E_band_08[j,:] = summary_distribution(E_bands_08)
    b_08[j,:] = summary_distribution(bs_08)
    A_08[j,:] = summary_distribution(As_08)
    E_band_12[j,:] = summary_distribution(E_bands_12)
    b_12[j,:] = summary_distribution(bs_12)
    A_12[j,:] = summary_distribution(As_12)
    E_band_02[j,:] = summary_distribution(E_bands_02)
    b_02[j,:] = summary_distribution(bs_02)
    A_02[j,:] = summary_distribution(As_02)
    E_band_04[j,:] = summary_distribution(E_bands_04)
    b_04[j,:] = summary_distribution(bs_04)
    A_04[j,:] = summary_distribution(As_04)
    E_band_06[j,:] = summary_distribution(E_bands_06)
    b_06[j,:] = summary_distribution(bs_06)
    A_06[j,:] = summary_distribution(As_06)


#%%
CI_b08 = (b_08[:,2]-b_08[:,1])/b_08[:,0]
CI_b12 = (b_12[:,2]-b_12[:,1])/b_12[:,0]
CI_b02 = (b_02[:,2]-b_02[:,1])/b_02[:,0]
CI_b04 = (b_04[:,2]-b_04[:,1])/b_04[:,0]
CI_b06 = (b_06[:,2]-b_06[:,1])/b_06[:,0]

CI_E_band08 = (E_band_08[:,2]-E_band_08[:,1])/E_band_08[:,0]
CI_E_band12 = (E_band_12[:,2]-E_band_12[:,1])/E_band_12[:,0]
CI_E_band02 = (E_band_02[:,2]-E_band_02[:,1])/E_band_02[:,0]
CI_E_band04 = (E_band_04[:,2]-E_band_04[:,1])/E_band_04[:,0]
CI_E_band06 = (E_band_06[:,2]-E_band_06[:,1])/E_band_06[:,0]

bs = np.vstack((b_08[:,0],b_12[:,0],b_02[:,0],b_04[:,0],b_06[:,0]))
E_bands = np.vstack((E_band_08[:,0],E_band_12[:,0],E_band_02[:,0],E_band_04[:,0],E_band_06[:,0]))
As = np.vstack((A_08[:,0],A_12[:,0],A_02[:,0],A_04[:,0],A_06[:,0]))


CI_b = np.vstack((CI_b08, CI_b12, CI_b02, CI_b04, CI_b06 ))
CI_E_band = np.vstack((CI_E_band08, CI_E_band12, CI_E_band02, CI_E_band04, CI_E_band06 ))


plt.figure()
plt.title(name + "relative CI b, dia_pool = " +str(dia_pooled))
plt.imshow(CI_b)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()


plt.figure()
plt.title(name + "relative CI E_band, dia_pool = " +str(dia_pooled))
plt.imshow(CI_E_band)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

#%%
plt.figure()
plt.title(name + "relative CI b, capped at 1, dia_pool = " +str(dia_pooled))
plt.imshow(np.minimum(CI_b,1))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()


plt.figure()
plt.title(name + "relative CI E_band, capped at 1, dia_pool = " +str(dia_pooled))
plt.imshow(np.minimum(CI_E_band,1))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()
#%%

plt.figure()
plt.title(name + "b, dia_pool = " +str(dia_pooled))
plt.imshow(bs)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

plt.figure()
plt.title(name + "b, dia_pool = " +str(dia_pooled) + ", capped at 1")
plt.imshow((np.minimum(bs,1)))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

plt.figure()
plt.title(name + "E_band, dia_pool = " +str(dia_pooled) + ", capped at 3")
plt.imshow(E_bands)
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()


plt.figure()
plt.title(name + "E_band, dia_pool = " +str(dia_pooled) + ", capped at 3")
plt.imshow((np.minimum(E_bands, 3)))
plt.yticks([0,1,2,3,4],[0.8,1.2,0.2,0.4,0.6])
plt.xticks([0,1,2],["[64,30]","[64,60]","[64,90]"])
plt.xticks([0,1,2,3],["[64,0]","[64,30]","[64,60]","[64,90]"])
plt.ylabel("upperlimit fittingrange = dE1 + ... [eV]")
plt.xlabel("pixel")
plt.colorbar()

#%%

ranges = [0.8,1.2,0.2,0.4,0.6]

for j in range(len(try_pixels)):
    cluster = im.clustered[try_pixels[j],row]
    xmin = im.dE1[1,cluster] - 1
    xmax = im.dE1[1,cluster] + 3.5
    baseline = np.average(ieels[j,0,(im.deltaE>xmin -0.1) & (im.deltaE<xmin)])

    ymax = np.max(ieels[j,0,(im.deltaE>xmin)&(im.deltaE<xmax)]) + 100
    x_values = im.deltaE[(im.deltaE>xmin)&(im.deltaE<xmax)]
    plt.figure()
    plt.title(name + "bandgap fits pixel [64," + str(try_pixels[j]) + "], pooled with diameter " + str(dia_pooled))
    plt.plot(x_values, ieels[j,0,(im.deltaE>xmin)&(im.deltaE<xmax)]-baseline, lw = 1.5)
    plt.fill_between(x_values, ieels[j,1,(im.deltaE>xmin)&(im.deltaE<xmax)] - baseline, ieels[j,2,(im.deltaE>xmin)&(im.deltaE<xmax)]- baseline, alpha=0.3)
    plt.vlines(im.dE1[1,cluster],0,500, linestyle = '--', color= 'black', label = 'dE1')
    for i in range(len(E_bands)):
        plt.plot(x_values, bandgap(x_values, As[i,j], E_bands[i,j], bs[i,j]), label = str(ranges[i]))
        # plt.plot(x_values, bandgap_b(x_values, As[i,j], E_bands[i,j]), label = str(ranges[i]))
    plt.legend()
    plt.ylim(-50,ymax)

