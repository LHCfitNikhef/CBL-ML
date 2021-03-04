#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:04:07 2020

@author: isabel
"""
import numpy as np
import matplotlib.pyplot as plt



def semi_DFT(x_n, y_n):
    N = len(x_n)
    deltax = (x_n[-1]-x_n[0])/N
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    for k in range(N):
        omg_k = k/N
        s_DFT[k] = np.sum(y_n*np.exp(-2j*np.pi*omg_k*x_n)*deltax)
    return s_DFT

def semi_iDFT(x_n, y_n, A = 1):
    N = len(x_n)
    deltax = (x_n[-1]-x_n[0])/N
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    for k in range(N):
        omg_k = k/N
        s_DFT[k] = A*np.sum(y_n*np.exp(2j*np.pi*omg_k*x_n)*deltax)
    return s_DFT

#---------------
def DFT(x_n, y_n):
    N = len(x_n)
    #deltax = (x_n[-1]-x_n[0])/(N-1)
    s_DFT = np.zeros(N) + 1j*np.zeros(N)

    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/(N-1)
    shift = x_min/deltax

    x_n = np.linspace(round(shift), round(shift) + N-1,N)

    for k in range(N):
        #omg_k = k/N
        exp_factor = -2j*np.pi*k/N
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)*deltax
        s_DFT[k] = s_DFT_k
    return s_DFT 

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

def DFT(x, y):
    x_0 = np.min(x)
    x_max = np.max(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(2j*np.pi*N_0*k/N) #np.exp(-1j*(x_0)*k*delta_omg)*delta_x 
    F_k = cont_factor * np.fft.fft(y)
    return F_k *delta_x

def iDFT(x, Y_k):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(-2j*np.pi*N_0*k/N)
    f_n = np.fft.ifft(cont_factor*Y_k) ##np.exp(-2j*np.pi*x_0*k)
    return f_n



#-------------------



def DFT1(x_n, y_n):
    N = len(x_n)
    #deltax = (x_n[-1]-x_n[0])/(N-1)
    s_DFT = np.zeros(N) + 1j*np.zeros(N)

    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/(N-1)
    shift = x_min/deltax

    x_n = np.linspace(round(shift), round(shift) + N-1,N)

    for k in range(N):
        #omg_k = k/N
        exp_factor = -2j*np.pi*k/N
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)*deltax
        s_DFT[k] = s_DFT_k
    return s_DFT


def iDFT1(x_n, y_n, A = 1):
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



def DFT_oud(x_n, y_n):
    N = len(x_n)
    deltax = (x_n[-1]-x_n[0])/N
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    
    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/N
    shift = x_min/deltax
    
    x_n = np.linspace(round(shift), round(shift) + N-1,N)
    
    for k in range(N):
        #omg_k = k/N
        exp_factor = -2j*np.pi*k/N
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)/N
        s_DFT[k] = s_DFT_k
    return s_DFT

def iDFT_oud(x_n, y_n, A = 1):
    N = len(x_n)
    
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    
    #x_n = np.linspace(0, N-1,N)
    x_min = np.min(x_n)
    x_max = np.max(x_n)
    deltax = (x_max-x_min)/N
    shift = x_min/deltax
    
    x_k = np.linspace(round(shift), round(shift) + N-1,N)

    omega_n = np.linspace(0, N-1,N)

    for k in range(N):
        x = x_k[k]
        exp_factor = 2j*np.pi*x/N
        exp = np.exp(exp_factor*omega_n)
        s_DFT_k = np.sum(y_n*exp)
        s_DFT[k] = s_DFT_k
    return s_DFT


def DFT_true(x_n, y_n):
    N = len(x_n)
    deltax = (x_n[-1]-x_n[0])/N
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    x_n = np.linspace(0, N-1,N)
    for k in range(N):
        #omg_k = k/N
        exp_factor = -2j*np.pi*k/N
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)
        s_DFT[k] = s_DFT_k
    return s_DFT

def iDFT_true(x_n, y_n, A = 1):
    N = len(x_n)
    deltax = (x_n[-1]-x_n[0])/N
    s_DFT = np.zeros(N) + 1j*np.zeros(N)
    x_n = np.linspace(0, N-1,N)
    for k in range(N):
        exp_factor = 2j*np.pi*k/N
        exp = np.exp(exp_factor*x_n)
        s_DFT_k = np.sum(y_n*exp)/N
        s_DFT[k] = s_DFT_k
    return s_DFT


def gauss(DeltaE, sigma, A, shift = 0):
    izlp = A*np.exp( -np.power((DeltaE-shift),2)/np.power(sigma,2) )
    return izlp


def gauss_der(DeltaE, sigma, A, shift = 0):
    izlp = -2*(DeltaE-shift)/np.power(sigma,2) * gauss(DeltaE, sigma, A, shift = shift)
    return izlp


#deltaE
deltaE_min = -1 # eV, adjust for each spectrum
deltaE_max = 100 # eV, adjust for each spectrum
l=2000
deltaE = np.linspace(deltaE_min, deltaE_max, l)


#variables ZLP
sigma_ZLP = 0.3 # eV (variance)
A_ZLP = 1e8 # normalisation
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

single_scat = EELsample
n_scattering = 15
EELsample = 0
semi = False
for i in range(1, n_scattering+1):
    if semi:
        EELsample += iDFT(deltaE,np.power(DFT(deltaE,single_scat),i))
    else:
        EELsample += np.fft.ifft(np.power(np.fft.fft(single_scat),i))



EELtot = EELZLP+EELsample
EELtot_der = EELZLP_der+EELsample_der


norm = True
if norm:
    norm_factor = np.sum(EELZLP)#*(deltaE_max-deltaE_min)/l
    EELZLP /= norm_factor
    EELsample /= norm_factor
    EELtot /= norm_factor
    A_sample /= norm_factor


#semi = False
if semi:
    z_nu = DFT(deltaE, EELZLP)
    i_nu = DFT(deltaE, EELtot)
else:
    z_nu = np.fft.fft( EELZLP)
    i_nu = np.fft.fft( EELtot)



abs_i_nu = np.absolute(i_nu)
max_i_nu = np.max(abs_i_nu)
i_nu_copy = np.copy(i_nu)
#i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
N_ZLP = np.sum(EELZLP)#1 #arbitrary units??? np.sum(EELZLP)

#precautionary for nan values in s_nu and j1_nu
very_small_value = 1E-8
print(z_nu[z_nu == 0].size, i_nu[i_nu == 0].size)
z_nu[z_nu == 0] = very_small_value
i_nu[i_nu == 0] = very_small_value

s_nu = N_ZLP*np.log(i_nu/z_nu)
#s_nu[np.isnan(s_nu)] = 0
j1_nu = z_nu*s_nu/N_ZLP

#j1_nu[np.isnan(j1_nu)] = 0 #do something with nanvalues...



s_nu_2 = s_nu
s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf

if semi:
    S_E = iDFT(deltaE, s_nu_2)
    J1_E = np.real(iDFT(deltaE, j1_nu))
else:
    S_E = np.fft.ifft( s_nu_2)
    J1_E = np.real(np.fft.ifft( j1_nu))
    


plt.figure()
plt.title("z_nu")
plt.plot(np.absolute(z_nu))
plt.figure()
plt.title("i_nu")
plt.plot(np.absolute(i_nu))

plt.figure()
plt.title("i_nu/z_nu")
plt.plot(np.absolute(i_nu/z_nu))

plt.figure()
plt.title("$\sigma_{ZLP} = " + str(sigma_ZLP) + "$, $\sigma_{sample} = " + str(sigma_sample) + "$")
plt.plot(deltaE,S_E[:len(EELZLP)]*max(J1_E)/max(S_E),linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(deltaE,EELtot,linewidth=2.5,color="black",label=r"${\rm total}$")
plt.plot(deltaE,EELZLP,linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(deltaE,EELsample,linewidth=2.5,color="red",ls="dashdot",label=r"${\rm sample}$")
plt.plot(deltaE,J1_E[:len(EELZLP)],linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")
plt.plot(deltaE,iDFT(deltaE,z_nu)[:len(EELZLP)],linewidth=2.5,color="pink",ls="dashdot",label=r"${\rm S(-E)}$")

plt.legend()
# Now produce the plot        
plt.xlabel(r"${\rm Energy~loss~(eV)}$",fontsize=17)
plt.ylabel(r"${\rm Intensity~(a.u.)}$",fontsize=17)
plt.xlim(shift_sample-3*sigma_sample,shift_sample+6*sigma_sample)
plt.ylim(0,100.5*A_sample)

plt.figure()
plt.title("$\sigma_{ZLP} = " + str(sigma_ZLP) + "$, $\sigma_{sample} = " + str(sigma_sample) + "$")
plt.plot(deltaE[:2*l],S_E[:2*l]*max(J1_E)/max(S_E),linewidth=2.5,color="grey",ls="dotted",label=r"${\rm S(E)}$")
plt.plot(deltaE[:2*l],EELZLP[:2*l],linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(deltaE[:2*l],EELsample[:2*l],linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.plot(deltaE[:2*l],J1_E[:2*l],linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")
#plt.plot(deltaE[:2*l],iDFT(deltaE,z_nu)[:2*l],linewidth=2.5,color="pink",ls="dotted",label=r"${\rm ZLP}$")
plt.legend()
plt.ylim(0,200*A_sample)

plt.figure()
plt.plot(deltaE[:2*l],EELZLP[:2*l],linewidth=2.5,color="blue",ls="dashed",label=r"${\rm ZLP}$")
plt.plot(deltaE[:2*l],EELsample[:2*l]*A_ZLP,linewidth=2.5,color="red",label=r"${\rm sample}$")
plt.plot(deltaE[:2*l],J1_E[:2*l]*A_ZLP,linewidth=2.5,color="green",ls="dotted",label=r"${\rm J^1(E)}$")
plt.legend()
