#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:47:05 2020

@author: isabel
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def CFT(x, y):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_max = np.max(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(2j*np.pi*N_0*k/N)*delta_x #np.exp(-1j*(x_0)*k*delta_omg)*delta_x 
    F_k = cont_factor * np.fft.fft(y)
    return F_k

def iCFT(x, Y_k):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    k = np.linspace(0, N-1, N)
    cont_factor = np.exp(-2j*np.pi*N_0*k/N)
    f_n = np.fft.ifft(cont_factor*Y_k)/delta_x # 2*np.pi ##np.exp(-2j*np.pi*x_0*k)
    return f_n

def convolute(xf, yf, xg, yg):
    Nf = len(xf)
    Ng = len(xg)
    Nh = Nf+Ng-1
    xf0 = np.min(xf)
    xf1 = np.max(xf)
    deltaxf = (xf1-xf0)/(Nf-1)
    xg0 = np.min(xg)
    xg1 = np.max(xg)
    deltaxg = (xg1-xg0)/(Ng-1)
    h_n = np.zeros(Nh)
    x_n = np.zeros(Nh)
    if not deltaxf == deltaxg:
        print(deltaxf,  deltaxg)
        print("delta_x_f and delta_x_g not equal, unable to convolute")
        return x_n, h_n
    
    x_n = np.linspace(xf0+xg0, xf1+xg1, Nh)
    yg_expanded = np.zeros(Ng + 2*(Nf-1))
    yg_expanded[Nf-1:Ng+Nf-1] = yg
    
    for n in range(Nh):
        h_n[n] = np.sum(yf[::-1] * yg_expanded[n:Nf+n])*deltaxf
    
    return x_n, h_n


def gauss(DeltaE, sigma, A, shift = 0):
    g_x = A / (2**0.5*math.pi**0.5*sigma_Z) *np.exp( -np.power((DeltaE-shift),2)/(2*np.power(sigma,2)) )
    return g_x


N = 2000
x_0 = -20
x_1 = 50
x = np.linspace(x_0,x_1,N)
dx = (x_1-x_0)/N
N_0 = np.argmin(np.absolute(x))


#%%
#DECONVOLUTION:
ZLP = (1+1j) * np.zeros(len(x))
ZLP[N_0] = 2E4
sigma_Z = 0.3
A_Z = 4E3
ZLP = gauss(x,sigma_Z,A_Z)
z_nu = CFT(x,ZLP)
N_ZLP = np.sum(ZLP)*dx


A_S = 500
sigma_S = 0.5
mu_S = 2
S_E = gauss(x, sigma_S, A_S, mu_S)
#I_E = S_E
i_nu = (1+1j)*np.zeros(len(S_E))
i_nu += z_nu
s_nu = CFT(x,S_E)
scatterings = 6
plt.figure()
for i in range(1, scatterings):
    add = z_nu*np.power(s_nu, i)/(math.factorial(i)*N_ZLP**i)
    i_nu += add
    plt.plot(x, iCFT(x,add), color = np.array([0.8,0.8,1])*(1.0-i/scatterings), label = "J" + str(i) + "(E)")
I_E = np.real(iCFT(x, i_nu))
z_nu[z_nu == 0] = 1E-14
i_nu[i_nu == 0] = 1E-14
deconv = N_ZLP*np.log(i_nu/z_nu)
S_Ec = iCFT(x,deconv)


plt.plot(x,ZLP, label = "I_ZLP(E)")
plt.plot(x,I_E, label = "I(E)")
plt.plot(x,S_Ec, linewidth = 2.5,label = "calculated S(E)")
plt.plot(x,S_E, '--', linewidth = 1.5, label = "original S(E)")
plt.xlim(0,10)
plt.ylim(0,A_S*1.5)
plt.legend()
plt.title("decovolution of convoluted gaussian")


plt.figure()
I_E2 = np.copy(ZLP)
scatterings = 5
plt.figure()
for n in range(1,scatterings):
    A_n = A_S**n / (math.factorial(n)*A_Z**(n-1))
    sigma_n = (sigma_Z**2 + n*sigma_S**2)**0.5
    mu_n = n*mu_S
    I_E2 += gauss(x, sigma_n, A_n, mu_n)
    plt.plot(x, gauss(x, sigma_n, A_n, mu_n), color = np.array([1,1,1])*n/scatterings, label = "J" + str(i) + "(E)")

plt.plot(x, I_E, label="I(E)")
plt.plot(x,I_E, label = "convoluted I(E)")
plt.plot(x,I_E2, label = "calculated I(E)")
plt.ylim(0,A_S*1.5)
plt.xlim(0,10)
plt.legend()
plt.title("comparison of calculated I(E) and convoluted I(E) for gaussian S(E)")





#%% DECONVOLUTION OF I_E IS GAUSSIAN
A_EEL = 0.8
sigma_EEL = 0.5
mu_EEL = 3
y5 = gauss(x, sigma_EEL, A_EEL, mu_EEL)
Y5= CFT(x,y5)
N_ZLP = 3
A_ZLP = 3
sigma_ZLP = 0.2
ZLP = gauss(x,sigma_ZLP,A_ZLP)
z_nu = CFT(x,ZLP)
Y6 = Y5 + z_nu
z_nu[z_nu == 0] = 1E-14
Y6[Y6 == 0] = 1E-14

deconv = N_ZLP*np.log(Y6/z_nu)
deconv[250:-250] = 0 #SUPPRESS NOISE
plt.figure()
y6 = iCFT(x, Y6)
S6_E = iCFT(x,deconv)
plt.plot(x,y6, label = "J(E)")
plt.plot(x,S6_E, label = "S(E)")
plt.xlim(-3,30)
plt.ylim(-A_EEL*0.8,A_EEL*2.5)
plt.title("decovolution of  gaussian")
scatterings = 5
for i in range(2,scatterings):
    plt.plot(x, iCFT(x,np.power(deconv, i)/math.factorial(i)), color = np.array([1,1,1])*i/scatterings, label = "J" + str(i) + "(E)")



#ANALYTICAL RESULT
S_Ec = np.zeros(len(x))

for n in range(1,10):
    A_n = (-1)**(n+1) * A_EEL**n/(n*A_ZLP**(n-1))
    mu_n = n*mu_EEL
    sigma_n = (n*(sigma_EEL**2 - sigma_ZLP**2))**0.5
    
    S_Ec += gauss(x,sigma_n, A_n, mu_n)

plt.plot(x,S_Ec, label = "analytical S(E)")
plt.legend()
    
    
    
    

