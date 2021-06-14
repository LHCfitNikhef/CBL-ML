#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:05:30 2021

@author: isabel
"""
import numpy as np
import matplotlib.pyplot as plt

im = im
spectra = im.data[0,:]
logNtot = np.log(np.sum(spectra,axis=1))
spectra[spectra<1] = 1
spectra = np.log(spectra[:, im.deltaE<1])
specshape = spectra.shape
spectra = spectra.flatten()
plt.plot(spectra)

var = np.zeros([3, spectra.size])
var[0,:] = spectra
del spectra

for i in range(specshape[0]):
    vari = np.ones((2,specshape[1]))
    vari[0,:] = logNtot[i]
    vari[1,:] = im.deltaE[im.deltaE<1]
    var[1:3,i*specshape[1]:(i+1)*specshape[1]] = vari


plt.figure()
plt.plot(var[0])
plt.figure()
plt.plot(var[1])
plt.figure()
plt.plot(var[2])

cov = np.cov(var)
print(cov)

#%%

# pixx = 0
# pixy = 0
# sig = 'pooled'

# zlps = im.calc_gen_ZLPs(pixx,pixy, signal = sig)

# zlp_mod = np.log(zlps[0,:specshape[1]])

#im.pool(11)


sig = 'pooled'

pixx = 30
pixy = 30

ylim = 7000

zlps = im.calc_ZLPs(pixx,pixy, signal = sig)
zlp = np.nanpercentile(zlps, 50, axis = 0)


plt.figure(figsize=(4, 3), dpi=200)
plt.title('Characteristic EELS')
plt.ylabel("intensity [arb. units]")
plt.xlabel('energy loss [eV]')
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'grey', lw =1.5, label = 'complete EELS')
plt.legend()
#plt.plot([min(im.deltaE), max(im.deltaE)], [ylim,ylim], color = 'grey', lw = 0.75)

# plt.figure()
# plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'black', lw =3, label = 'EELS')
# plt.plot(im.deltaE, zlp, '-.', color = 'lb', label = 'possible ZLP')
# plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig)-zlp, '--', label = r'possible I_{inel}')
# plt.ylim([-100,ylim])
# plt.title('Characteristic electron energy loss spectrum, zoomed')
# plt.ylabel("intensity [arb. units]")
# plt.xlabel('energy loss [eV]')
# plt.legend()



plt.figure()
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'grey', lw =1.5, label = 'complete EELS')
plt.plot(im.deltaE, zlp, linestyle = (0, (3, 5, 1, 6)), color = 'blue', lw = 2, label = 'possible ZLP')
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig)-zlp, lw =2, linestyle = (0,(1, 2.5)), color = 'red', label = r'possible I_{inel}')
plt.ylim([-100,ylim])
plt.title('Characteristic EELS, zoomed')
plt.ylabel("intensity [arb. units]")
plt.xlabel('energy loss [eV]')
plt.legend()

#%%

ylim2 = 1000
xlim2 = 5
plt.figure(figsize=(4, 3), dpi=200)
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'grey', lw =1.5, label = 'complete EELS')
plt.plot(im.deltaE, zlp, linestyle = (0, (3, 4, 1, 4)), color = 'blue', lw = 2, label = 'possible ZLP')
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig)-zlp, lw =2, linestyle = (0,(1.5, 2.5)), color = 'red', label = r'possible $I_{inel}$')
plt.ylim([-200,ylim])
plt.title('Characteristic EELS, zoomed')
plt.ylabel("intensity [arb. units]")
plt.xlabel('energy loss [eV]')
plt.legend()
plt.plot([0,0], [-20,ylim2], color = 'black', lw = 0.75)
plt.plot([0,xlim2], [ylim2,ylim2], color = 'black', lw = 0.75)
plt.plot([xlim2,xlim2], [-20,ylim2], color = 'black', lw = 0.75)
plt.plot([0,xlim2], [-20,-20], color = 'black', lw = 0.75)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))



#%%

ylim2 = 1000
xlim2 = 5
plt.figure(figsize=(4, 3), dpi=200)
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'grey', lw =1.5, label = 'complete EELS')
plt.plot(im.deltaE, zlp, linestyle = (0, (3, 4, 1, 4)), color = 'blue', lw = 2, label = 'possible ZLP')
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig)-zlp, lw =2, linestyle = (0,(2, 3.5)), color = 'red', label = r'possible $I_{inel}$')
plt.ylim([-20,ylim2])
plt.xlim([0,5])
plt.title('Characteristic EELS, zoomed')
plt.ylabel("intensity [arb. units]")
plt.xlabel('energy loss [eV]')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()


#%%
ylim2 = 800
xlim2 = 5
plt.figure(figsize=(7, 3), dpi=200)
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig), color = 'grey', lw =1.5, label = 'complete EELS')
plt.plot(im.deltaE, zlp, linestyle = (0, (3, 4, 1, 4)), color = 'blue', lw = 2, label = 'possible ZLP')
plt.plot(im.deltaE, im.get_pixel_signal(pixx,pixy, signal = sig)-zlp, lw =2, linestyle = (0,(2, 3.5)), color = 'red', label = r'possible  $I_{inel}$')
plt.ylim([-20,ylim2])
plt.xlim([0,5])
plt.title('Characteristic EELS, zoomed')
plt.ylabel("intensity [arb. units]")
plt.xlabel('energy loss [eV]')
plt.legend()
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))








