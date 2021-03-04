#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 01:07:53 2020

@author: isabel
"""
#ANOTHER TRY TO DO IT MYSELF

import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import math

def fix_phi(phi):
    lim_fix = np.pi
    check = (phi[1:] - phi[:-1] > lim_fix)*-1
    check2 = (phi[1:] - phi[:-1] < -lim_fix)*1
    check3 = np.cumsum(check+check2)*2*np.pi
    fix = np.insert(phi[1:]+check3,0, phi[0])
    return fix

def deconvolute(Y):
    rY = np.absolute(Y)
    phiY = np.angle(Y)
    phiY = fix_phi(phiY)
    rYc = np.power(rY,0.5)
    phiYc = phiY/2
    return rYc*np.exp(1j*phiYc)

def CFT_approx2(x, y):
    x_0 = np.min(x)
    #print("CFT x_0", x_0)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    delta_omg = 2*np.pi/(N*delta_x)
    F_k = (1 + 1j) * np.zeros(N)
    n = np.linspace(0, N-1, N)
    for k in range(N):
        cont_factor = np.exp(-1j*x_0*k*delta_omg)*delta_x#(2*np.pi)**-0*
        #cont_factor = np.exp(-1j*x_0*k*delta_omg)*delta_x
        exp_factor = -2j*np.pi*n*k/N
        F_k[k] = cont_factor * np.sum(np.exp(exp_factor)*y)
    return F_k

def iCFT_approx2(x, Y_k):
    x_0 = np.min(x)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    delta_omg = 2*np.pi/(N*delta_x)
    f_n = (1 + 1j) * np.zeros(N)
    k = np.linspace(0, N-1, N)
    for n in range(N):
        x_n = x[n]
        cont_factor = delta_omg/(2*np.pi)
        exp_factor = 1j*k*delta_omg*x_0 + 2j*np.pi*n*k/N
        f_n[n] = cont_factor * np.sum(np.exp(exp_factor)*Y_k)
    return f_n


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


"""   
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    delta_omg = 2*np.pi/(N*delta_x)
    F_k = (1 + 1j) * np.zeros(N)




    delta_omg = 2*np.pi/(N*delta_x)
    f_n = (1 + 1j) * np.zeros(N)
    #n = np.linspace(0, N-1, N)
    
    G_k = np.exp(1j*k*delta_omg)*x_0*Y_k 
    f_n = delta_omg/(2*np.pi) * np.fft.ifft(G_k)
"""



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

def ft(x, samples):
    """Approximate the Fourier Transform of a time-limited 
    signal by means of the discrete Fourier Transform.
    
    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal
    """
    x_0 = np.min(x)
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max-x_0)/N
    Fs = 1/delta_x
    t0 = -x_0
    f = np.linspace(-Fs/2, Fs/2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples)/Fs * np.exp(-2j*np.pi*f*t0))

def gauss(DeltaE, sigma, A, shift = 0):
    g_x = A / (2**0.5*math.pi**0.5*sigma_Z) *np.exp( -np.power((DeltaE-shift),2)/(2*np.power(sigma,2)) )
    return g_x

#try multiples:
N = 2000
x_0 = -3
x_1 = 6
xi = np.linspace(x_0,x_1,N)
dx = (x_1-x_0)/N
N_0 = np.argmin(np.absolute(xi))


y1 = np.zeros(N)
y_0 = 700
dy = 200
y1[y_0:y_0+dy] = 1
x1, y1c = convolute(xi,y1,xi,y1)

Y1 = CFT(xi,y1)
Y1T = np.power(Y1, 2)#(6.2830)**0.5*
Y1C = CFT(x1,y1c)
Y1Csq = deconvolute(Y1C)#np.power(Y1C, 0.5)
y1t = iCFT(xi, Y1T)

#add zeros at end to match N_conv
xee = np.linspace(x_0,2*x_1-x_0,2*N-1)
y1ee = np.zeros(2*N-1)
y1ee[:N] = y1
Y1ee = CFT(xee,y1ee)
Y1eeT = np.power(Y1ee, 2)#(6.2830)**0.5*
#Y1eeT = Y1ee*np.conj(Y1ee)
y1eet = iCFT(xee, Y1eeT)

#add zeros at begin and end to match domain conv
xe = np.linspace(x_0+x_0,2*x_1,2*N-1)
y1e = np.zeros(2*N-1)
arg_0 = np.argmin(np.absolute(xe))
y1e[arg_0:N+arg_0] = y1
Y1e = CFT(xe,y1e)
Y1eT = np.power(Y1e, 2)#(6.2830)**0.5*
#Y1eT = Y1e*np.conj(Y1e)
y1et = iCFT(xe, Y1eT)

y1DFTc = np.fft.ifft(np.power(np.fft.fft(y1),2))*dx  

#np.exp(-2j*np.pi*N_0/N)*
#y1DFTcs = np.concatenate((y1DFTc[N_0:], y1DFTc[:N_0]))


#add zeros at begin and end to match 
xem = np.linspace(x_0-x_1,x_1-x_0,2*N-1)
y1em = np.zeros(2*N-1)
arg_0 = np.argmin(np.absolute(xem))
y1em[arg_0:N+arg_0] = y1
Y1emt = CFT(xem,y1em)
Y1emT = np.power(Y1emt, 2)#(6.2830)**0.5*
#Y1eT = Y1e*np.conj(Y1e)
y1emt = iCFT(xem, Y1emT)



"""
Y12 = np.zeros(len(Y1T)*2-1)
for i in range(len(Y12)):
    if i%2 == 0:
        #print(i)
        Y12[i] = Y1T[int(i/2)]
    else:
        Y12[i] = 0#(Y1T[int((i-1)/2)] + Y1T[int((i+1)/2)])/2
"""
"""
plt.figure()
#plt.plot(Y1, label = "Y1")
#plt.plot(Y1T, label = "Y1*Y1")
plt.plot(Y1C, label = "FT(y1c)")
plt.plot(Y12, 'o', label = "Y1*Y1, 2")
plt.legend()
plt.xlim(0,50)

plt.figure()
plt.plot(np.imag(Y1), label = "Y1")
plt.plot(np.imag(Y1T), label = "Y1*Y1")
plt.plot(np.imag(Y1C), label = "FT(y1c)")
#plt.plot(Y12, 'o', label = "Y1*Y1, 2")
plt.legend()
plt.xlim(0,50)
"""

og_max = np.max(y1c)

plt.figure()
plt.title("convolutions")
plt.plot(xi,y1, label ="OG")
plt.plot(x1, iCFT(x1,Y1Csq), label = "conv sq")
plt.plot(xi,y1t, label="FT conv")#*og_max/np.min(y1t)
plt.plot(xi,y1DFTc, label = "DFT conv, shift")
#plt.plot(xe,y1et ,'--', label="FT conv, ext dm")#*og_max/np.min(y1et)
#plt.plot(xem,y1emt, '--',label="FT conv, -a,a")#*og_max/np.min(y2et)
#plt.plot(xee,y1eet, label="FT conv, ext end")#*og_max/np.min(y1eet)
plt.plot(x1,y1c, '--', label ="official conv")

plt.legend()

#%%

#N = 1000
x_0 = -20
x_1 = 50
x = np.linspace(x_0,x_1,N)
y2 = gauss(x, 0.5,0.8,3)
x2, y2c = convolute(x,y2,x,y2)
dx = (x_1-x_0)/N
N_0 = np.argmin(np.absolute(x))



Y2 = CFT(x,y2)
Y2T = np.power(Y2, 2)#(6.2830)**0.5*np.power(Y2, 2)
Y2C = CFT(x2,y2c)
Y2Csq = deconvolute(Y2C)#np.power(Y1C, 0.5)
y2t = iCFT(x, Y2T)

"""
Y22 = np.zeros(len(Y2)*2-1)
for i in range(len(Y22)):
    if i%2 == 0:
        #print(i)
        Y22[i] = Y2T[int(i/2)]
    else:
        Y22[i] = 0#(Y2T[int((i-1)/2)] + Y2T[int((i+1)/2)])/2



plt.figure()
#plt.plot(Y2, label = "Y2")
plt.plot(Y2T, label = "Y2*Y2")
plt.plot(Y2C, label = "FT(y2c)")
plt.plot(Y22, 'o', label = "Y2*Y2, 2")
plt.legend()
plt.xlim(0,50)
"""

#add zeros at end to match N_conv
xee = np.linspace(x_0,2*x_1-x_0,2*N-1)
y2ee = np.zeros(2*N-1)
y2ee[:N] = y2
Y2ee = CFT(xee,y2ee)
Y2eeT = np.power(Y2ee, 2)#(6.2830)**0.5*np.power(Y2ee, 2)
y2eet = iCFT(xee, Y2eeT)

#add zeros at begin and end to match domain conv
xe = np.linspace(x_0+x_0,2*x_1,2*N-1)
y2e = np.zeros(2*N-1)
arg_0 = np.argmin(np.absolute(xe))
y2e[arg_0:N+arg_0] = y2
Y2e = CFT(xe,y2e)
Y2eT = np.power(Y2e, 2)#(6.2830)**0.5*np.power(Y2e, 2)
y2et = iCFT(xe, Y2eT)

y2DFTcs = np.fft.ifft(np.power(np.fft.fft(y2),2))*dx



#y2DFTcs = np.concatenate((y2DFTc[N_0:], y2DFTc[:N_0]))*dx


#add zeros at begin and end to match 
xem = np.linspace(x_0-x_1,x_1-x_0,2*N-1)
y2em = np.zeros(2*N-1)
arg_0 = np.argmin(np.absolute(xem))
y2em[arg_0:N+arg_0] = y2
Y2emt = CFT(xem,y2em)
Y2emT = np.power(Y2emt, 2)#(6.2830)**0.5*
#Y1eT = Y1e*np.conj(Y1e)
y2emt = iCFT(xem, Y2emT)

og_max = np.max(y2c)

plt.figure()
plt.title("convolutions")
plt.plot(x,y2, label ="OG")
plt.plot(x2, iCFT(x2,Y2Csq), label = "conv sq")
#plt.plot(xe,iCFT(xe,Y2e))
plt.plot(x,y2t, label="FT conv")#*og_max/np.min(y2t)
plt.plot(x,y2DFTcs, label = "DFT conv, shift")
#plt.plot(xe,y2et, '--',label="FT conv, ext dm")#*og_max/np.min(y2et)
#plt.plot(xem,y2emt, '--',label="FT conv, -a,a")#*og_max/np.min(y2et)
#plt.plot(xee,y2eet, label="FT conv, ext end")#*og_max/np.min(y2eet
plt.plot(x2,y2c, '--', label ="official conv")
plt.legend()
plt.xlim(0,60)

plt.figure()
plt.title("Fourier transforms")
plt.plot(np.imag(Y2C), label = "Im[ft conv]")
plt.plot(np.real(Y2C), label = "Re[ft conv]")
plt.plot(np.absolute(Y2C), label = "abs[conv]")
#plt.plot(np.imag(Y2ee), label = "Im[ext end]")
#plt.plot(np.real(Y2ee), label="Re[ext end]")
#plt.plot(np.absolute(Y2ee), label = "abs[ext end]")
plt.plot(np.imag(Y2e), label = "Im[ext dom]")
plt.plot(np.real(Y2e), label="Re[ext dom]")
plt.plot(np.absolute(Y2e), label = "abs[ext dom]")
plt.xlim((0,50))
plt.ylim((-0.65,1.4))
plt.legend()

plt.figure()
plt.title("Fourier transforms")
plt.plot(np.imag(Y2C), label = "Im[ft conv]")
plt.plot(np.real(Y2C), label = "Re[ft conv]")
plt.plot(np.absolute(Y2C), label = "abs[conv]")
#plt.plot(np.imag(Y2ee), label = "Im[ext end]")
#plt.plot(np.real(Y2ee), label="Re[ext end]")
#plt.plot(np.absolute(Y2ee), label = "abs[ext end]")
plt.plot(np.imag(Y2e), label = "Im[ext dom]")
plt.plot(np.real(Y2e), label="Re[ext dom]")
plt.plot(np.absolute(Y2e), label = "abs[ext dom]")
plt.xlim((len(Y2C)-50, len(Y2C)))
plt.ylim((-0.65,1.4))
plt.legend()

#%%

#DECONVOLUTION:
N_0 = np.argmin(np.absolute(x))
ZLP = (1+1j) * np.zeros(len(x))
ZLP[N_0] = 2E4
ZLP = gauss(x,0.3,4E3)
z4_nu = CFT(x,ZLP)
N_ZLP = np.sum(ZLP)*dx
plt.figure()
A = 500
y3 = gauss(x, 0.5, A, 2)
y4 = y3
Y4 = (1+1j)*np.zeros(len(y3))
Y4 += z4_nu
Y3 = CFT(x,y3)
scatterings = 6
for i in range(1, scatterings):
    add = z4_nu*np.power(Y3, i)/(math.factorial(i)*N_ZLP**i)
    Y4 += add
    plt.plot(x, iCFT(x,add), color = np.array([0.8,0.8,1])*(1.0-i/scatterings), label = "J" + str(i) + "(E)")
y4 = np.real(iCFT(x, Y4))
#y4 = I_E
#Y4 = CFT(x,y4)
#z4_nu = z_nu
z4_nu[z4_nu == 0] = 1E-14
Y4[Y4 == 0] = 1E-14
deconv = N_ZLP*np.log(Y4/z4_nu)
S_E = iCFT(x,deconv)
plt.plot(x,ZLP, label = "I_ZLP(E)")
plt.plot(x,y4, label = "I(E)")
#plt.plot(x,y3)
plt.plot(x,S_E, linewidth = 2.5,label = "calculated S(E)")
plt.plot(x,y3, '--', linewidth = 1.5, label = "original S(E)")
#xc,yc = convolute(x,y3,x,y3)
#plt.plot(xc,yc)
plt.xlim(0,10)
plt.ylim(0,A*1.2)
plt.legend()
plt.title("decovolution of convoluted gaussian")


#%% DECONVOLUTION OF NON FT CONVOLUTED SIGNAL
N_0 = np.argmin(np.absolute(x))
ZLP = (1+1j) * np.zeros(len(x))
ZLP[N_0] = 1E3
ZLP = gauss(x,0.3,4E3)

z4_nu = CFT(x,ZLP)
N_ZLP = np.sum(ZLP)*dx
plt.figure()
scale = 1
A = 0.01
y3 = gauss(x, 0.8, A, 3)
x32, y32 = convolute(x,ZLP,x,y3)
y32 = y32[N_0:N_0+N]
y7 = y32*scale +ZLP
scatterings = 10
for i in range(2, scatterings):
    x32, y32 = convolute(x, y3, x, y32)
    y32 = y32[N_0:N_0+N]
    y7 += y32*scale/math.factorial(i)
    plt.plot(x, y32*scale/math.factorial(i), color = np.array([1,1,1])*i/scatterings, label = "J" + str(i) + "(E)")
Y4 = CFT(x,y7*scale)
z4_nu[z4_nu == 0] = 1E-14
Y4[Y4 == 0] = 1E-14

#y4 = iCFT(x, Y4)
#z4_nu = np.power(Y3, 0)
deconv = np.log(Y4/z4_nu)*N_ZLP
plt.plot(x,y7, label = "J(E)")
#plt.plot(x,y3)
plt.plot(x,iCFT(x,deconv), label = "S(E)")
#xc,yc = convolute(x,y3,x,y3)
#plt.plot(xc,yc)
plt.xlim(0,10)
plt.ylim(0,A*1.1*N_ZLP)
plt.legend()
plt.title("decovolution of convoluted gaussian")


#%% deconvolution of the analytical signal
sigma_S = 0.5
A_S = 500
mu_S = 2

sigma_Z = 0.3
A_Z = 4E3


ZLP = (1 + 0j) * gauss(x,sigma_Z, A_Z)
S_E = (1 + 0j) * gauss(x,sigma_S, A_S, mu_S)


N_ZLP = np.sum(ZLP)*dx

I_E = np.copy(ZLP)
scatterings = 5
plt.figure()
for n in range(1,scatterings):
    A_n = A_S**n / (math.factorial(n)*A_Z**(n-1))
    sigma_n = (sigma_Z**2 + n*sigma_S**2)**0.5
    mu_n = n*mu_S
    I_E += gauss(x, sigma_n, A_n, mu_n)
    plt.plot(x, gauss(x, sigma_n, A_n, mu_n), color = np.array([1,1,1])*n/scatterings, label = "J" + str(i) + "(E)")

z_nu = CFT(x,ZLP)
i_nu = CFT(x, I_E)
deconv = np.log(i_nu/z_nu)*N_ZLP
S_Ec = iCFT(x, deconv)

plt.plot(x, I_E, label="I(E)")
plt.plot(x,S_E, label = "original S(E)")
plt.plot(x,S_Ec, label = "calculated S(E)")
plt.ylim(0,A_S*1.2)
plt.xlim(0,10)
plt.legend()


z_nu[z_nu == 0] = 1E-14
I_E[I_E == 0] = 1E-14
deconv = N_ZLP*np.log(I_E/z_nu)
S_E = iCFT(x,deconv)
plt.plot(x,ZLP, label = "I_ZLP(E)")
plt.plot(x,y4, label = "I(E)")
#plt.plot(x,y3)
plt.plot(x,S_E, linewidth = 2.5,label = "calculated S(E)")
plt.plot(x,y3, '--', linewidth = 1.5, label = "original S(E)")
#xc,yc = convolute(x,y3,x,y3)
#plt.plot(xc,yc)
plt.xlim(0,10)
plt.ylim(0,A*1.2)
plt.legend()
plt.title("decovolution of convoluted gaussian")
    

#%% DECONVOLUTION OF GAUSS
A = 0.8
y5 = gauss(x, 0.5, A, 3)
Y5= CFT(x,y5)
N_ZLP = 3
ZLP = gauss(x,0.2,3)
z_nu = CFT(x,ZLP)
Y6 = Y5 + z_nu#np.power(Y5,0)*N_ZLP
#z6_nu = np.power(Y5, 0)*N_ZLP
#z6_nu = z_nu*1E-4
z_nu[z_nu == 0] = 1E-14
Y6[Y6 == 0] = 1E-14

deconv = N_ZLP*np.log(Y6/z_nu)
deconv[250:-250] = 0
plt.figure()
y6 = iCFT(x, Y6)
S6_E = iCFT(x,deconv)
plt.plot(x,y6, label = "J(E)")
plt.plot(x,S6_E, label = "S(E)")
plt.xlim(-3,30)
plt.ylim(-A*0.8,A*2.5)
plt.title("decovolution of  gaussian")
scatterings = 5
for i in range(2,scatterings):
    plt.plot(x, iCFT(x,np.power(deconv, i)/math.factorial(i)), color = np.array([1,1,1])*i/scatterings, label = "J" + str(i) + "(E)")
plt.legend()

