#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:27:56 2020

Kramer-Kronig analysis including iteration


@author: isabel
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
#import start_cleaning_lau
import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/revise_Lau')
from revise_Lau.functions_revised import *




def kramer_kronig(x, y, plot = False):
    #TODO: change variables to correct values
    beta = 30E-3
    m_0 = 1
    v = 0.5 #needs to be smaller than c
    c = 1 #natural units?
    gamma = (1-v**2/c**2)**-0.5
    
    deltaE = x
    deltaE[deltaE == 0] = 1e-14 #some very small number
    
    theta_E = deltaE/(gamma*m_0*v**2)
    log_term = np.log(1+(beta/theta_E)**2)
    
    
    
    EELsample = y
    EELsample_ac = EELsample/log_term
    
    #step 3: normalisation and retreiving Im[1/eps(E)]
    
    Re_eps0 = 0 #value of Re[1/eps(0)]
    int_EELsample_over_deltaE = np.sum(EELsample_ac/deltaE)*ddeltaE
    K = 2*int_EELsample_over_deltaE/(math.pi*(1-Re_eps0))
    
    
    Im_eps = EELsample/K #Im[-1/eps(E)]
    
    if plot:
        plt.figure()
        plt.plot(np.real(Im_eps[:100]), label = "real")
        plt.plot(np.imag(Im_eps[:100]), label = "imag")
        plt.legend()
        plt.title("real and imag part Im_eps")
    
    #step 4: retreiving Re[1/eps(E)]
    
   
    
    deltaE_Re = deltaE
    Re_eps = np.zeros(sem_inf) 
    
    #integrate around each energy (to avoid singularities)
    sgn = np.ones(Im_eps.shape)
    half = math.floor(Im_eps.size/2)
    sgn[-half:] *= -1
    
        
    #TODO: evaluate possible influence discrete approximation sine and cosine transform 
    q_t = scipy.fft.idst(Im_eps)
    p_t = sgn*q_t
    Re_eps =  scipy.fft.dct(p_t)
        
    
    eps1 = Re_eps / (Re_eps**2 + Im_eps**2)
    eps2 = Im_eps / (Re_eps**2 + Im_eps**2)
    eps = eps1 + 1j*eps2
    
    
    if plot:
        plt.figure()
        plt.plot(deltaE[:2*l], eps1[:2*l], label = r"$\varepsilon_1$")
        plt.plot(deltaE[:2*l], eps2[:2*l], label = r"$\varepsilon_2$")
        plt.plot(deltaE[:2*l], np.absolute(eps[:2*l]), label = r"$|\varepsilon|$")
        plt.title(r"dielectric function")
        plt.legend()
        plt.xlabel(r"$\Delta E$")
        plt.ylabel(r"$\varepsilon$")
        
        plt.figure()
        plt.plot(deltaE[:l], eps1[:l], label = r"$\varepsilon_1$")
        plt.plot(deltaE[:l], eps2[:l], label = r"$\varepsilon_2$")
        plt.plot(deltaE[:l], np.absolute(eps[:l]), label = r"$|\varepsilon|$")
        plt.title(r"dielectric function")
        plt.legend()
        plt.xlabel(r"$\Delta E$")
        plt.ylabel(r"$\varepsilon$")
    
    
    return eps, K


def calculate_S_s(x, y, eps, K):
    #TODO: fix variables!!!!
    t = 1000#E-6 
    S_s = np.zeros(x.shape)
    
    beta = 30E-3
    m_0 = 1
    v = 0.5 #needs to be smaller than c
    c = 1 #natural units?
    h_bar = 1
    deltaE = x
    deltaE[deltaE == 0] = 1e-14 #some very small number

    gamma = (1-v**2/c**2)**-0.5
    theta_E = deltaE/(gamma*m_0*v**2)

    k_0 = gamma*m_0*v/h_bar #gamma*m0*v/h_bar
    
    term = 2*K/(math.pi*t*k_0) * (np.arctan(beta/theta_E)/theta_E - beta/(beta**2-theta_E**2))
    eps_term = np.imag(-4/(1+eps)) - np.imag(-1/eps)
    
    S_s = term*eps_term
    
    return S_s


def calculate_Kroger(x, y, eps):
    S = np.zeros(x.shape)
    
    return S


xs = total_replicas['x14'].values
Nx = np.sum(total_replicas['x14']==total_replicas['x14'][0])
nx = int(len(total_replicas)/Nx)

x_14 = df_sample.iloc[5].x_shifted
ys_14 = smooth(df_sample.iloc[5].y, 50)

ys_ZLPs = total_replicas.match14.values


r = 3 #Drude model, can also use estimation from exp. data
n_times_extra = 20
l = len(x_14)
sem_inf = l*(n_times_extra+1)
ddeltaE = (x_14[-1]-x_14[0])/l


A = ys_14[-1]
ys_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x_14[0], (sem_inf-1)*ddeltaE+x_14[0], sem_inf)

ys_extrp[:l] = ys_14
x_extrp[:l] = x_14
ys_extrp[l:] = A*np.power(1+x_extrp[l:]-x_extrp[l],-r)
x_14 = x_extrp
ys_14 = ys_extrp
ZLPs_14 = np.zeros((Nx, sem_inf))

for i in range(Nx):
    ys_ZLP = smooth(ys_ZLPs[i*nx:(i+1)*nx],50)
    ys_ZLP_extrp = np.zeros(sem_inf)
    ys_ZLP_extrp[:nx] = ys_ZLP
    ys_ZLP_extrp[nx:] = ys_ZLP[-1]*np.power(1+x_extrp[nx:]-x_extrp[nx],-r)
    ys_ZLP = ys_ZLP_extrp
    ZLPs_14[i,:] = ys_ZLP


EELsample = ys_14-np.average(ZLPs_14, axis = 0)



"""
x_14 = df_sample.iloc[5].x_shifted
#y_14 = df_sample.iloc[5].y_smooth
ys_14 = smooth(df_sample.iloc[5].y, 50)

#nx_2 = len(x_14)

#x_14 = xs[:nx]
#y_14 = total_replicas['data y14'].values[:nx]
ys_ZLPs = total_replicas.match14.values


r = 3 #Drude model, can also use estimation from exp. data
n_times_extra = 2
l = len(x_14)
sem_inf = l*(n_times_extra+1)
ddeltaE = (x_14[-1]-x_14[0])/l


A = ys_14[-1]
ys_extrp = np.zeros(sem_inf)
x_extrp = np.linspace(x_14[0], (sem_inf-1)*ddeltaE+x_14[0], sem_inf)

ys_extrp[:l] = ys_14
x_extrp[:l] = x_14
ys_extrp[l:] = A*np.power(1+x_extrp[l:]-x_extrp[l],-r)
x_14 = x_extrp
ys_14 = ys_extrp


ZLPs_14 = np.zeros((Nx, sem_inf))

for i in range(Nx):
    ys_ZLP = smooth(ys_ZLPs[i*nx:(i+1)*nx],50)
    ys_ZLP_extrp = np.zeros(sem_inf)
    ys_ZLP_extrp[:nx] = ys_ZLP
    ys_ZLP_extrp[nx:] = ys_ZLP[-1]*np.power(1+x_extrp[nx:]-x_extrp[nx],-r)
    ys_ZLP = ys_ZLP_extrp
    ZLPs_14[i,:] = ys_ZLP

"""













#TODO define
x = x_14
y = ys_14
I_tot = EELsample * ddeltaE/np.sum(ys_14)
dE = x











err_th = 1E6 * np.ones(l)
err = 1E8 * np.ones(l)
I_new = I_tot
eps_old, K = kramer_kronig(dE, I_new)
i = 0
max_iter = 100
plt.figure()
plt.plot(dE[:l], eps_old[:l], label = i)
while((err_th < err).any() and i<100):
    S_s = calculate_S_s(dE, I_new, eps_old, K)
    Kroger = calculate_Kroger(dE, I_new, eps_old)
    I_new = I_new - S_s #TODO ?
    #I_new = Kroger #TODO ?
    eps_new, K = kramer_kronig(dE, I_new)
    
    #TODO: make error relative to max value eps?
    err = np.absolute(eps_old[:l] - eps_new[:l])
    eps_old = eps_new
    i += 1
    plt.plot(dE[:l], eps_old[:l], label = i)


















