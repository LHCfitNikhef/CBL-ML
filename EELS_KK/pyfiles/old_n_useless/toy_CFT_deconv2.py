#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:03:09 2020

@author: isabel
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


def DFT(x_n, y_n):
    N = len(x_n)
    #deltax = (x_n[-1]-x_n[0])/(N-1)
    s_DFT = np.zeros(N) + 1j*np.zeros(N)

    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/(N-1)
    shift = x_min/deltax

    x_n = np.linspace(round(shift), round(shift) + N-1,N)
    omega_k = np.zeros(N)

    for k in range(N):
        #omg_k = k/N
        exp_factor = -2j*np.pi*k/N
        #omega_k[k] = -exp_factor
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)*deltax
        s_DFT[k] = s_DFT_k
    return s_DFT, omega_k

def iDFT(x_n, y_n, A = 1):
    N = len(x_n)

    s_DFT = np.zeros(N) + 1j*np.zeros(N)

    #x_n = np.linspace(0, N-1,N)
    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/(N-1)
    shift = x_min/deltax

    x_k = np.linspace(round(shift), round(shift) + N-1,N)

    omega_n = np.linspace(0, N-1,N)

    for k in range(N):
        x = x_k[k]
        exp_factor = 2j*np.pi*x/N
        exp = np.exp(exp_factor*omega_n)
        s_DFT_k = np.sum(y_n*exp)/(N*deltax)
        s_DFT[k] = s_DFT_k
    return s_DFT


def gauss(DeltaE, sigma, A, shift = 0):
    izlp = A*np.exp( -np.power((DeltaE-shift),2)/np.power(sigma,2) )
    return izlp


def gauss_der(DeltaE, sigma, A, shift = 0):
    izlp = -2*(DeltaE-shift)/np.power(sigma,2) * gauss(DeltaE, sigma, A, shift = shift)
    return izlp


a = np.array([0,0,0,1,1,0,0,0])
x_a = np.linspace(-3,4,8)

b = a
x_b = x_a/2

c = np.array([0,1,0,0,0,0,0,1])
x_c = x_a

A, omega_a = DFT(x_a,a)
B, omega_b = DFT(x_a,b)
C, omega_c = DFT(x_a,c)

a_a = iDFT(x_a, A*A)
a_b = iDFT(x_a, A*B)
a_c = iDFT(x_a, A*C)

plt.figure()
plt.title("convolutions")
plt.plot(a_a, label = "a*a")
#plt.plot(iDFT(x_a, DFT(x_a,a)*DFT(x_b, b)), label = "a*b")
#plt.plot(iDFT(x_a, DFT(x_a,a)*DFT(x_c, c)), label = "a*c")
plt.legend()

A = fft(a)
B = fft(x_b)
C = fft(x_c)

plt.figure()
plt.title("convolutions numpy pack")
plt.plot(ifft(A*A), label = "a*a")
plt.plot(ifft(A*B), label = "a*b")
plt.plot(ifft(A*C), label = "a*c")
plt.legend()


#%%GOING BACKWARDS

#deltaE
deltaE_min = -1 # eV, adjust for each spectrum
deltaE_max = 100 # eV, adjust for each spectrum
l=2000
deltaE = np.linspace(deltaE_min, deltaE_max, l)


#variables ZLP
sigma_ZLP = 0.3 # eV (variance)
A_ZLP = 1e3 # normalisation
EELZLP = gauss(deltaE,sigma_ZLP,A_ZLP)
EELZLP_der = gauss_der(deltaE, sigma_ZLP, A_ZLP)



sigma_sample = 0.6 # eV (variance)
A_sample = 0.3 # normalisation
shift_sample = 1.5
EELsample = gauss(deltaE, sigma_sample, A_sample, shift_sample)
EELsample_der = gauss_der(deltaE,  sigma_sample, A_sample, shift_sample)

zlp_is_delta = True
sample_is_delta = False
if zlp_is_delta:
    arg_zero = np.argmin(np.absolute(deltaE))
    EELZLP = np.zeros(l)
    EELZLP[arg_zero] = A_ZLP
    if sample_is_delta:
        arg_peak = np.argmin(np.absolute(deltaE-shift_sample))
        EELsample = np.zeros(l)
        EELsample[arg_peak] = A_sample
  
S_E = EELsample
        
N_ZLP = np.sum(EELZLP)#1 #arbitrary units??? np.sum(EELZLP)

s_nu = np.fft.fft(EELsample)
z_nu = np.fft.fft(EELZLP)
i_nu = z_nu*np.exp(s_nu/N_ZLP)

EELsample = np.fft.ifft(i_nu)

if True:
    s_nu = DFT(deltaE,EELsample)
    z_nu = DFT(deltaE,EELZLP)
    i_nu = z_nu*np.exp(s_nu/N_ZLP)
    
    EELsample = iDFT(deltaE,i_nu)


plt.figure()
plt.plot(deltaE, S_E, label = "S(E)")
plt.plot(deltaE, EELsample, label = "I_sample")
plt.plot(deltaE,iDFT(deltaE,np.exp(s_nu)))
plt.ylim((0,A_sample*1.5))
plt.xlim((0,5))
plt.legend()