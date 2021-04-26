#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 15:03:28 2021

@author: isabel
"""
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from image_class_bs import Spectral_image
import os
from datetime import datetime
import sys

#print(os.listdir())
path_to_KK = 'KK_results/'

#path_to_KK = sys.argv[1]
  

name_tickness = 'thickness_'
files_t = [filename for filename in os.listdir(path_to_KK) if filename.startswith(name_tickness)]

name_eps = 'diel_fun_'
files_eps = [filename for filename in os.listdir(path_to_KK) if filename.startswith(name_eps)]


#im = Spectral_image.load_data('../../dmfiles/h-ws2_eels-SI_004.dm4')
#

im = Spectral_image.load_data('/data/theorie/ipostmes/cluster_programs/EELS_KK/dmfiles/h-ws2_eels-SI_004.dm4')
#im.cluster(5)
#im= im

t = np.zeros(np.append(len(files_t),im.image_shape))
eps = (1+1j)*np.zeros(np.append(np.append(len(files_eps),im.image_shape),len(im.deltaE[im.deltaE>0])))

for i in range(len(files_t)):
    file = files_t[i]
    t[i] = np.load(path_to_KK + file)


t_med = np.nanpercentile(t, 50, axis=0)
t_low = np.nanpercentile(t, 16, axis=0)
t_high = np.nanpercentile(t, 84, axis=0)


"""
with open("summary/thickness_median.npy", 'wb') as f:
    np.save(f, t_med)
with open("summary/thickness_low.npy", 'wb') as f:
    np.save(f, t_low)
with open("summary/thickness_high.npy", 'wb') as f:
    np.save(f, t_med)
"""


"""
baseline = np.nanpercentile(t[:,im.clustered == 0],50)

pix_max_t = np.unravel_index(t.argmax(), t.shape)

cmap = sns.cm.rocket_r
plt.figure()
plt.title("thickness of sample")
ax = sns.heatmap(np.nanpercentile(t, 50, axis=0), cmap = cmap)
plt.show()
cmap = sns.cm.rocket_r
plt.figure()
plt.title("thickness of sample")
ax = sns.heatmap(np.nanpercentile(t, 50, axis=0), cmap = cmap, vmax = 150)
plt.show()

cmap = sns.cm.rocket_r
plt.figure()
plt.title("thickness of sample")
ax = sns.heatmap(np.nanpercentile(t, 50, axis=0)-baseline, cmap = cmap)
plt.show()
cmap = sns.cm.rocket_r
plt.figure()
plt.title("thickness of sample")
ax = sns.heatmap(np.nanpercentile(t, 50, axis=0)-baseline, cmap = cmap, vmax = 150)
plt.show()


im.plot_sum()
"""

for i in range(len(files_eps)):
    file = files_eps[i]
    eps[i] = np.load(path_to_KK + file)
del file

"""
then = datetime.now()
print("before finding eps median and stuff: ", then)
eps_med = np.nanpercentile(eps, 50, axis=0)
now1 = datetime.now()
print("eps med gevonden ", now1)
print("time spend in med: ", now1-then)
eps_low = np.nanpercentile(eps, 16, axis=0)
now2 = datetime.now()
print("eps low gevonden ", now2)
print("time spend in low: ", now2-now1)
eps_high = np.nanpercentile(eps, 84, axis=0)
now3 = datetime.now()
print("eps high gevonden ", now3)
print("time spend in high: ", now3-now2)
print("tot tijd in dit: ", now3-then)
del eps


with open("summary/die_fun_median.npy", 'wb') as f:
    np.save(f, eps_med)
with open("summary/die_fun_low.npy", 'wb') as f:
    np.save(f, eps_low)
with open("summary/die_fun_high.npy", 'wb') as f:
    np.save(f, eps_high)
"""


"""
plt.figure()
plt.title("dielectric function mixel [50,60]")
plt.ylabel("eps")
plt.xlabel("energy loss [eV]")
plt.plot(im.deltaE[im.deltaE>0],np.real(eps_med[50,50,:]), label = "real")
plt.fill_between(im.deltaE[im.deltaE>0], np.real(eps_low[50,50,:]),np.real(eps_high[50,50,:]), alpha=0.3)
plt.plot(im.deltaE[im.deltaE>0],np.imag(eps_med[50,50,:]), label = "imag")
plt.fill_between(im.deltaE[im.deltaE>0], np.imag(eps_low[50,50,:]),np.imag(eps_high[50,50,:]), alpha=0.3)
plt.savefig("diel_fun_50_50.jpg")

"""


















