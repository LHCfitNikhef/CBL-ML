#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:37:18 2021

@author: isabel
"""
from image_class_bs import Spectral_image
import numpy as np
import matplotlib.pyplot as plt
import logging
import math
from scipy.fftpack import next_fast_len



_logger = logging.getLogger(__name__)

def kramers_kronig_hs(self, I_EELS,
                        N_ZLP=None,
                        iterations=1,
                        n=None,
                        t=None,
                        delta=0.5, correct_S_s = False,
                        Srfint_start = None , initial_S_s = None):
    
    # if initial_S_s is not None:
    #     I_EELS = I_EELS - initial_S_s
    
    output = {}
    # Constants and units
    me = 511.06

    e0 = 200 #keV
    beta =30 #mrad
    
    e0 = self.e0
    beta = self.beta

    eaxis = self.deltaE[self.deltaE>0] #axis.axis.copy()
    S_E = I_EELS[self.deltaE>0] # - initial_S_s[self.deltaE>0]
    if initial_S_s is not None:
        y = I_EELS[self.deltaE>0] - initial_S_s[self.deltaE>0]
    else:
        y = I_EELS[self.deltaE>0]
    l = len(eaxis)
    i0 = N_ZLP
    
    # Kinetic definitions
    ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
    tgt = e0 * (2 * me + e0) / (me + e0)
    rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

    for io in range(iterations):
        # Calculation of the ELF by normalization of the SSD
        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE#axis.scale
        if n is None and t is None:
            raise ValueError("The thickness and the refractive index are "
                             "not defined. Please provide one of them.")
        elif n is not None and t is not None:
            raise ValueError("Please provide the refractive index OR the "
                             "thickness information, not both")
        elif n is not None:
            # normalize using the refractive index.
            K = np.sum(Im/eaxis)*self.ddeltaE 
            K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
            te = (332.5 * K * ke / i0)
        elif t is not None:
            if N_ZLP is None:
                raise ValueError("The ZLP must be provided when the  "
                                 "thickness is used for normalization.")
            # normalize using the thickness
            K = t * i0 / (332.5 * ke)
            te = t
        Im = Im / K


        esize = next_fast_len(2*l)#2**math.floor(math.log2(l)+1)*4)# #HIER AANGEPAST, omgedraaid -> helpt niet
        q = -2 * np.fft.fft(Im, esize).imag / esize

        q[:l] *= -1
        q = np.fft.fft(q)
        # Final touch, we have Re(1/eps)
        Re = q[:l].real + 1
       
        e1 = Re / (Re ** 2 + Im ** 2)
        e2 = Im / (Re ** 2 + Im ** 2)

        if iterations > 0 and N_ZLP is not None:
            Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
            adep = (tgt / (eaxis + delta) *
                    np.arctan(beta * tgt / eaxis) -
                    beta / 1000. /
                    (beta ** 2 + eaxis ** 2. / tgt ** 2))
            Srfint = 2000 * K * adep * Srfelf / rk0 / te * self.ddeltaE #axis.scale #TODO: HIER - toegevoegd om effect te zien
            if correct_S_s == True:
                print("correcting S_s")
                Srfint[Srfint<0] = 0
                Srfint[Srfint>S_E] = S_E[Srfint>S_E]
            y = S_E - Srfint 
            _logger.debug('Iteration number: %d / %d', io + 1, iterations)
            

    eps = (e1 + e2 * 1j)
    del y
    del I_EELS
    if 'thickness' in output:
        # As above,prevent errors if the signal is a single spectrum
        output['thickness'] = te
    
    return eps, te, Srfint


def kramers_kronig_hs_verlengd(self, I_EELS,
                        N_ZLP=None,
                        iterations=1,
                        n=None,
                        t=None,
                        delta=0.5, correct_S_s = False,
                        Srfint_start = None , initial_S_s = None):
    
    # if initial_S_s is not None:
    #     I_EELS = I_EELS - initial_S_s
    
    output = {}
    # Constants and units
    me = 511.06

    e0 = 200 #keV
    beta =30 #mrad
    
    e0 = self.e0
    beta = self.beta

    l_i = len(self.deltaE[self.deltaE>0])
    l = l_i*2
    
    eaxis =  np.linspace(im.ddeltaE,(l)*im.ddeltaE, l)#axis.axis.copy()
    
    S_E = np.zeros(l)
    S_E[:l_i] = I_EELS[self.deltaE>0] # - initial_S_s[self.deltaE>0]
    S_E[l_i:] = I_EELS[-1]*np.power(np.linspace(1, 1+(l-l_i)*im.ddeltaE, l-l_i),-3)
    
    
    # S_s_e = np.zeros(len(S_s_norm)*2)
    # S_s_e[:len(S_s_norm)] = S_s_norm
    # S_s_e[im.l:] = S_s_norm[-1]*np.power(np.linspace(1, im.l*im.ddeltaE, im.l),-3)
    
    # S_b_e = np.zeros(len(S_b_norm)*2)
    # S_b_e[:len(S_b_norm)] = S_b_norm
    # S_b_e[im.l:] = S_b_norm[-1]*np.power(np.linspace(1, im.l*im.ddeltaE, im.l),-3)
    
    # S_s_norm = S_s_e
    # S_b_norm = S_b_e

    
    
    if initial_S_s is not None:
        S_s_E = np.zeros(l)
        S_s_E[:l_i] = initial_S_s[self.deltaE>0] # - initial_S_s[self.deltaE>0]
        S_s_E[l_i:] = initial_S_s[-1]*np.power(np.linspace(1, 1+(l-l_i)*im.ddeltaE, l-l_i),-3)
        y = S_E - S_s_E
    else:
        y = S_E
    
    i0 = N_ZLP
    
    # Kinetic definitions
    ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
    tgt = e0 * (2 * me + e0) / (me + e0)
    rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

    for io in range(iterations):
        # Calculation of the ELF by normalization of the SSD
        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE#axis.scale
        if n is None and t is None:
            raise ValueError("The thickness and the refractive index are "
                             "not defined. Please provide one of them.")
        elif n is not None and t is not None:
            raise ValueError("Please provide the refractive index OR the "
                             "thickness information, not both")
        elif n is not None:
            # normalize using the refractive index.
            K = np.sum(Im/eaxis)*self.ddeltaE 
            K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
            te = (332.5 * K * ke / i0)
        elif t is not None:
            if N_ZLP is None:
                raise ValueError("The ZLP must be provided when the  "
                                 "thickness is used for normalization.")
            # normalize using the thickness
            K = t * i0 / (332.5 * ke)
            te = t
        Im = Im / K


        esize = next_fast_len(2*l)#2**math.floor(math.log2(l)+1)*4)# #HIER AANGEPAST, omgedraaid -> helpt niet
        q = -2 * np.fft.fft(Im, esize).imag / esize

        q[:l] *= -1
        q = np.fft.fft(q)
        # Final touch, we have Re(1/eps)
        Re = q[:l].real + 1
       
        e1 = Re / (Re ** 2 + Im ** 2)
        e2 = Im / (Re ** 2 + Im ** 2)

        if iterations > 0 and N_ZLP is not None:
            Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
            adep = (tgt / (eaxis + delta) *
                    np.arctan(beta * tgt / eaxis) -
                    beta / 1000. /
                    (beta ** 2 + eaxis ** 2. / tgt ** 2))
            Srfint = 2000 * K * adep * Srfelf / rk0 / te * self.ddeltaE #axis.scale #TODO: HIER - toegevoegd om effect te zien
            if correct_S_s == True:
                print("correcting S_s")
                Srfint[Srfint<0] = 0
                Srfint[Srfint>S_E] = S_E[Srfint>S_E]
            y = S_E - Srfint 
            _logger.debug('Iteration number: %d / %d', io + 1, iterations)
            

    eps = (e1 + e2 * 1j)
    del y
    del I_EELS
    if 'thickness' in output:
        # As above,prevent errors if the signal is a single spectrum
        output['thickness'] = te
    
    return eps, te, Srfint


# path_to_results = "../../KK_results/image_KK_004_p_5.pkl"
# im = Spectral_image.load_Spectral_image(path_to_results)
# # # im.pixelsize *=1E6
# im.calc_axes()
# im.cluster(5, based_upon = 'log')

# im.n = np.ones(5)*4.6

im = im

S_s = im.ieels_p[im.clustered == 1]

S_b = im.ieels_p[im.clustered == 4]

avg_S_b = np.average(im.ieels_p[im.clustered == 4][:,0,:], axis=0)
avg_S_s = np.average(im.ieels_p[im.clustered == 1][:,0,:], axis=0)
#im.S_b = pass
S_b_avg =  np.average(S_b[:,0,:],axis=0) - (np.average(S_s[:,0,:], axis=0)-np.average(S_b[:,0,:],axis=0))
S_b_avg += 100
S_b_avg[:250] = 0
S_b_avg[S_b_avg<0] = 0

N_ZLP_s = np.sum(np.average(im.data[im.clustered == 1], axis=0)) - np.sum(np.average(im.ieels_p[im.clustered == 1][:,0,:], axis=0))
N_ZLP_b = np.sum(np.average(im.data[im.clustered == 4], axis=0)) - np.sum(np.average(im.ieels_p[im.clustered == 4][:,0,:], axis=0))

S_s_norm = avg_S_s/N_ZLP_s
S_b_norm = avg_S_b/N_ZLP_b
S_b_norm -= np.average(S_b_norm[:20])




#%%
N_ZLP = 1

im.set_n(4.1462, n_vac = 2.1759)
im.e0 = 200 #keV
im.beta = 67.2 #mrad

# eps_1, t_1, S_s_1 = kramers_kronig_hs(im, S_b_norm, N_ZLP, n=im.n[4], correct_S_s=True, initial_S_s=S_s_norm)

gebruik_fun = kramers_kronig_hs_verlengd

eps_1, t_1, S_s_1 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], initial_S_s=S_s_norm)

#%%
eps_2, t_2, S_s_2 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 2, initial_S_s=S_s_norm)
#%%
eps_3, t_3, S_s_3 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 3, initial_S_s=S_s_norm)

eps_5, t_5, S_s_5 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 5, initial_S_s=S_s_norm)

eps_30, t_30, S_s_30 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 30, initial_S_s=S_s_norm)

eps_200, t_200, S_s_200 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 200, initial_S_s=S_s_norm)


#%%
x_val = np.linspace(0, (len(S_s_1)-1)*im.ddeltaE, len(S_s_1))

plt.figure()
plt.title("surface scatterings over iterations")
plt.plot(im.deltaE[im.deltaE>0], S_s_norm[im.deltaE>0], label = "initial S_s")
plt.plot(x_val, S_s_1, label = "1 iteration")
plt.plot(x_val, S_s_2, label = "2 iteration")
plt.plot(x_val, S_s_3, label = "3 iteration")
plt.plot(x_val, S_s_5, label = "5 iteration")
plt.plot(x_val, S_s_30, label = "30 iteration")
plt.plot(x_val, S_s_200, label = "200 iteration")
plt.legend()



eps_1, t_1, S_s_1 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], correct_S_s=True, initial_S_s=S_s_norm)

eps_2, t_2, S_s_2 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 2, correct_S_s=True, initial_S_s=S_s_norm)

eps_3, t_3, S_s_3 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 3, correct_S_s=True, initial_S_s=S_s_norm)

eps_5, t_5, S_s_5 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 5, correct_S_s=True, initial_S_s=S_s_norm)

eps_30, t_30, S_s_30 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 30, correct_S_s=True, initial_S_s=S_s_norm)

eps_200, t_200, S_s_200 = gebruik_fun(im, S_b_norm, N_ZLP, n=im.n[4], iterations = 200, correct_S_s=True, initial_S_s=S_s_norm)


#%%
x_val = np.linspace(0, (len(S_s_1)-1)*im.ddeltaE, len(S_s_1))

plt.figure()
plt.title("surface scatterings over iterations, \nnegative/too positive corrected")
plt.plot(im.deltaE[im.deltaE>0], S_s_norm[im.deltaE>0], label = "initial S_s")
plt.plot(x_val, S_s_1, label = "1 iteration")
plt.plot(x_val, S_s_2, label = "2 iteration")
plt.plot(x_val, S_s_3, label = "3 iteration")
plt.plot(x_val, S_s_5, label = "5 iteration")
plt.plot(x_val, S_s_30, label = "30 iteration")
plt.plot(x_val, S_s_200, label = "200 iteration")
plt.legend()

#%%
plt.figure()
plt.title("comparing surface scatterings")
# plt.plot(im.deltaE, np.average(im.ieels_p[im.clustered == 4][:,0,:], axis=0), label = "IEELS")
plt.plot(im.deltaE, S_s_norm, label = "input surface")
plt.plot(im.deltaE, S_b_norm, label = "input bulk")
#plt.plot(im.deltaE, np.average(S_s[:,0,:], axis=0)-np.average(S_b[:,0,:],axis=0), label = "est. S_s")
plt.plot(x_val, S_s_1, label = "calc. S_s")
# plt.plot(im.deltaE[im.deltaE>0], S_b_avg[im.deltaE>0] + S_s_1, label = "calc. S_s + input bulk")
plt.legend()


