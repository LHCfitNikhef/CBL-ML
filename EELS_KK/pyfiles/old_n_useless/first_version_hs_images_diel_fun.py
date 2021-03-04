#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:12:19 2020

@author: isabel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 01:05:56 2020

@author: isabel
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import tensorflow as tf 
import tensorflow.compat.v1 as tf
from tensorflow import keras

from matplotlib import rc
from copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import fnmatch
import csv
import pickle
import warnings
import scipy
from scipy import optimize
from scipy.optimize import leastsq
from datetime import datetime
from matplotlib import cm
#from lmfit import Model
from scipy.optimize import curve_fit
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm;

tf.get_logger().setLevel('ERROR')

_logger = logging.getLogger(__name__)




def im_dielectric_function(data, energies):
    data = data[:,:, energies>0]
    energies = energies[energies>0]
    ZLPs_gen, dE0, dE1, dE2 = calc_ZLPs_gen(energies)
    dielectric_function_im_avg = (1+1j)*np.zeros(data.shape)
    dielectric_function_im_std = (1+1j)*np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print(i,j)
            data_ij = data[i,j,:]
            ZLPs = calc_ZLPs(data_ij, energies,ZLPs_gen, dE0, dE1, dE2)
            dielectric_functions = np.zeros(ZLPs.shape)
            for k in range(ZLPs.shape[0]):
                ZLP_k = ZLPs[k,:]
                N_ZLP = np.sum(ZLPs)
                dielectric_functions[k,:] = kramers_kronig_hs(energies, data_ij-ZLP_k, N_ZLP = N_ZLP, n =3, full_output=False)
            dielectric_function_im_avg[i,j,:] = np.average(dielectric_functions, axis = 0)
            dielectric_function_im_std[i,j,:] = np.std(dielectric_functions, axis = 0)
    return dielectric_function_im_avg, dielectric_function_im_std


def kramers_kronig_hs(deltaE, I_EELS,
                            N_ZLP=None,
                            iterations=1,
                            n=None,
                            t=None,
                            delta=0.5,
                            full_output=True, prints = np.array([]), correct_S_s = False):
    r"""Calculate the complex
    dielectric function from a single scattering distribution (SSD) using
    the Kramers-Kronig relations.

    It uses the FFT method as in [1]_.  The SSD is an
    EELSSpectrum instance containing SSD low-loss EELS with no zero-loss
    peak. The internal loop is devised to approximately subtract the
    surface plasmon contribution supposing an unoxidized planar surface and
    neglecting coupling between the surfaces. This method does not account
    for retardation effects, instrumental broading and surface plasmon
    excitation in particles.

    Note that either refractive index or thickness are required.
    If both are None or if both are provided an exception is raised.

    Parameters
    ----------
    zlp: {None, number, Signal1D}
        ZLP intensity. It is optional (can be None) if `t` is None and `n`
        is not None and the thickness estimation is not required. If `t`
        is not None, the ZLP is required to perform the normalization and
        if `t` is not None, the ZLP is required to calculate the thickness.
        If the ZLP is the same for all spectra, the integral of the ZLP
        can be provided as a number. Otherwise, if the ZLP intensity is not
        the same for all spectra, it can be provided as i) a Signal1D
        of the same dimensions as the current signal containing the ZLP
        spectra for each location ii) a BaseSignal of signal dimension 0
        and navigation_dimension equal to the current signal containing the
        integrated ZLP intensity.
    iterations: int
        Number of the iterations for the internal loop to remove the
        surface plasmon contribution. If 1 the surface plasmon contribution
        is not estimated and subtracted (the default is 1).
    n: {None, float}
        The medium refractive index. Used for normalization of the
        SSD to obtain the energy loss function. If given the thickness
        is estimated and returned. It is only required when `t` is None.
    t: {None, number, Signal1D}
        The sample thickness in nm. Used for normalization of the
         to obtain the energy loss function. It is only required when
        `n` is None. If the thickness is the same for all spectra it can be
        given by a number. Otherwise, it can be provided as a BaseSignal
        with signal dimension 0 and navigation_dimension equal to the
        current signal.
    delta : float
        A small number (0.1-0.5 eV) added to the energy axis in
        specific steps of the calculation the surface loss correction to
        improve stability.
    full_output : bool
        If True, return a dictionary that contains the estimated
        thickness if `t` is None and the estimated surface plasmon
        excitation and the spectrum corrected from surface plasmon
        excitations if `iterations` > 1.

    Returns
    -------
    eps: DielectricFunction instance
        The complex dielectric function results,

            .. math::
                \epsilon = \epsilon_1 + i*\epsilon_2,

        contained in an DielectricFunction instance.
    output: Dictionary (optional)
        A dictionary of optional outputs with the following keys:

        ``thickness``
            The estimated  thickness in nm calculated by normalization of
            the SSD (only when `t` is None)

        ``surface plasmon estimation``
           The estimated surface plasmon excitation (only if
           `iterations` > 1.)

    Raises
    ------
    ValuerError
        If both `n` and `t` are undefined (None).
    AttribureError
        If the beam_energy or the collection semi-angle are not defined in
        metadata.

    Notes
    -----
    This method is based in Egerton's Matlab code [1]_ with some
    minor differences:

    * The wrap-around problem when computing the ffts is workarounded by
      padding the signal instead of substracting the reflected tail.

    .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
       Microscope", Springer-Verlag, 2011.

    """
    output = {}
    # Constants and units
    me = 511.06

    e0 = 200 #  keV
    beta =30 #mrad

    eaxis = deltaE[deltaE>0] #axis.axis.copy()
    ddeltaE = (np.max(deltaE) - np.min(deltaE))/(len(deltaE - 1))
    S_E = I_EELS[deltaE>0]
    y = I_EELS[deltaE>0]
    l = len(eaxis)
    i0 = N_ZLP
    
    # Kinetic definitions
    ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
    tgt = e0 * (2 * me + e0) / (me + e0)
    rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)

    for io in range(iterations):
        # Calculation of the ELF by normalization of the SSD
        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / ddeltaE#axis.scale
        if n is None and t is None:
            raise ValueError("The thickness and the refractive index are "
                             "not defined. Please provide one of them.")
        elif n is not None and t is not None:
            raise ValueError("Please provide the refractive index OR the "
                             "thickness information, not both")
        elif n is not None:
            # normalize using the refractive index.
            K = np.sum(Im/eaxis)*ddeltaE 
            K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
            te = (332.5 * K * ke / i0)
            if full_output is True:
                output['thickness'] = te
        elif t is not None:
            if N_ZLP is None:
                raise ValueError("The ZLP must be provided when the  "
                                 "thickness is used for normalization.")
            # normalize using the thickness
            K = t * i0 / (332.5 * ke)
            te = t
        Im = Im / K

        # Kramers Kronig Transform:
        # We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
        # Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
        # Use an optimal FFT size to speed up the calculation, and
        # make it double the closest upper value to workaround the
        # wrap-around problem.
        esize = next_fast_len(2*l) #2**math.floor(math.log2(l)+1)*4
        q = -2 * np.fft.fft(Im, esize).imag / esize

        q[:l] *= -1
        q = np.fft.fft(q)
        # Final touch, we have Re(1/eps)
        Re = q[:l].real + 1
        # Egerton does this to correct the wrap-around problem, but in our
        # case this is not necessary because we compute the fft on an
        # extended and padded spectrum to avoid this problem.
        # Re=real(q)
        # Tail correction
        # vm=Re[axis.size-1]
        # Re[:(axis.size-1)]=Re[:(axis.size-1)]+1-(0.5*vm*((axis.size-1) /
        #  (axis.size*2-arange(0,axis.size-1)))**2)
        # Re[axis.size:]=1+(0.5*vm*((axis.size-1) /
        #  (axis.size+arange(0,axis.size)))**2)

        # Epsilon appears:
        #  We calculate the real and imaginary parts of the CDF
        e1 = Re / (Re ** 2 + Im ** 2)
        e2 = Im / (Re ** 2 + Im ** 2)

        if iterations > 0 and N_ZLP is not None:
            # Surface losses correction:
            #  Calculates the surface ELF from a vaccumm border effect
            #  A simulated surface plasmon is subtracted from the ELF
            Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
            adep = (tgt / (eaxis + delta) *
                    np.arctan(beta * tgt / eaxis) -
                    beta / 1000. /
                    (beta ** 2 + eaxis ** 2. / tgt ** 2))
            Srfint = 2000 * K * adep * Srfelf / rk0 / te * ddeltaE #axis.scale
            if correct_S_s == True:
                print("correcting S_s")
                Srfint[Srfint<0] = 0
                Srfint[Srfint>S_E] = S_E[Srfint>S_E]
            y = S_E - Srfint
            _logger.debug('Iteration number: %d / %d', io + 1, iterations)
            if iterations == io + 1 and full_output is True:
                output['S_s'] = Srfint
            del Srfint

    eps = (e1 + e2 * 1j)
    del y
    del I_EELS
    if 'thickness' in output:
        # As above,prevent errors if the signal is a single spectrum
        output['thickness'] = te
    if full_output is False:
        return eps
    else:
        return eps, output


def calc_ZLPs_gen( energies, specimen = 4):
    #from Functions import *
    #if specimen == 3:
    #    from load_data_sp3 import *
    #else:
    #    from Load_data import *
    
    tf.reset_default_graph()
    #cols=['y14', 'x14', 'y15', 'x15', 'y16', 'x16', 'y17', 'x17', 'y19', 'x19', 'y20', 'x20', 'y21', 'x21', 'y22', 'x22', 'y23', 'x23']
    
    #ZLP_data = pd.concat((file14, file15, file16, file17, file19, file20, file21, file22, file23), axis=1)
    #ZLP_data = ZLP_data.drop(['x', 'y_norm'],axis=1).rename(columns={'x_shifted': 'x'})
    #ZLP_data.columns = cols
    
    #print(ZLP_data)
    
    ## Window the data file to the desired energy range
    #E_min = -.3
    #CHANGE
    #if specimen ==3:
    #    E_min = -0.93
    #    E_max = 9.07
    #else:
    #    E_min= -4   
    #    E_max = 20
    #original = ZLP_data[(ZLP_data['x14'] >= E_min) & (ZLP_data['x14'] <= E_max)]
    
    if specimen == 3:
        d_string = '06.12.2020'
        path_to_data = 'Data_oud/Results/sp3/%(date)s/'% {"date": d_string} 
    else:
        d_string = '07.09.2020'
        path_to_data = 'Data_oud/Results/%(date)s/'% {"date": d_string} 
    
    path_predict = r'Predictions_*.csv'
    path_cost = r'Cost_*.csv' 
    
    all_files = glob.glob(path_to_data + path_predict)
    
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, delimiter=",",  header=0, usecols=[0,1,2], names=['x', 'y', 'pred'])
        li.append(df)
        
    
    training_data = pd.concat(li, axis=0, ignore_index=True)
    
    
    all_files_cost = glob.glob(path_to_data + path_cost)
    
    
    import natsort
    
    all_files_cost_sorted = natsort.natsorted(all_files_cost)
    
    chi2_array = []
    chi2_index = []
    
    for filename in all_files_cost_sorted:
        df = pd.read_csv(filename, delimiter=",", header=0, usecols=[0,1], names=['train', 'test'])
        best_try = np.argmin(df['test'])
        chi2_array.append(df.iloc[best_try,0])
        chi2_index.append(best_try)
    
    chi_data  = pd.DataFrame()
    chi_data['Best chi2 value'] = chi2_array
    chi_data['Epoch'] = chi2_index
        
    
    
    good_files = []
    count = 0
    threshold = 3
    
    for i,j in enumerate(chi2_array):
        if j < threshold:
            good_files.append(1) 
            count +=1 
        else:
            good_files.append(0)
    
    
    
    
    tf.get_default_graph
    tf.disable_eager_execution()
    
    def make_model(inputs, n_outputs):
        hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
        output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
        return output
    
    x = tf.placeholder("float", [None, 1], name="x")
    predictions = make_model(x, 1)
    
    
    prediction_file = pd.DataFrame()
    len_data = len(energies)
    predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
    predict_x = energies.reshape(len_data,1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(0,len(good_files)):
            if good_files[i] == 1:
                if specimen ==3:
                    best_model = 'Models_oud/Best_models/sp3/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                else:
                    best_model = 'Models_oud/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                saver = tf.train.Saver(max_to_keep=1000)
                saver.restore(sess, best_model)
    
                extrapolation = sess.run(predictions,
                                        feed_dict={
                                        x: predict_x
                                        })
                #prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(1000,)
                prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
    
    
    dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
    dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
    dE0 = np.round(dE1 - .5, 2) 
    
    
    nbins = len_data
    li = []
    diff = []
    
    for i in range(0, len(prediction_file.columns)):
        df = pd.DataFrame()
        #df['x'] = predict_x.reshape(1000,)
        df['x'] = predict_x.reshape(len_data,)
        df['prediction'] = prediction_file.iloc[:,i]
        df['k'] = i
        li.append(df)
    
    extrapolation = pd.concat(li, axis=0, ignore_index = True)
    
    ZLPs_gen = np.zeros((count, len_data))

    for k in range(count): 
        exp_k = extrapolation[extrapolation['k'] == k ]
        #mean_k, var_k, count = binned_statistics(exp_k['x'], exp_k['prediction'], nbins)[0:3]
        
        mean_k = extrapolation[extrapolation['k'] == k ]['prediction']
        
        ZLPs_gen[k,:] =  np.exp(mean_k) #matching(energies,, data)
        
    return ZLPs_gen, dE0, dE1, dE2

def calc_ZLPs_gen2( energies, specimen = 4):
    tf.reset_default_graph()
    
    if specimen == 3:
        d_string = '06.12.2020'
        path_to_data = 'Data_oud/Results/sp3/%(date)s/'% {"date": d_string} 
    else:
        d_string = '07.09.2020'
        path_to_data = 'Data_oud/Results/%(date)s/'% {"date": d_string} 
    
    path_predict = r'Predictions_*.csv'
    path_cost = r'Cost_*.csv' 
    
    all_files = glob.glob(path_to_data + path_predict)
    
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, delimiter=",",  header=0, usecols=[0,1,2], names=['x', 'y', 'pred'])
        li.append(df)
        
    
    training_data = pd.concat(li, axis=0, ignore_index=True)
    
    
    all_files_cost = glob.glob(path_to_data + path_cost)
    
    
    import natsort
    
    all_files_cost_sorted = natsort.natsorted(all_files_cost)
    
    chi2_array = []
    chi2_index = []
    
    for filename in all_files_cost_sorted:
        df = pd.read_csv(filename, delimiter=",", header=0, usecols=[0,1], names=['train', 'test'])
        best_try = np.argmin(df['test'])
        chi2_array.append(df.iloc[best_try,0])
        chi2_index.append(best_try)
    
    chi_data  = pd.DataFrame()
    chi_data['Best chi2 value'] = chi2_array
    chi_data['Epoch'] = chi2_index
        
    
    
    good_files = []
    count = 0
    threshold = 3
    
    for i,j in enumerate(chi2_array):
        if j < threshold:
            good_files.append(1) 
            count +=1 
        else:
            good_files.append(0)
    
    
    
    
    tf.get_default_graph
    tf.disable_eager_execution()
    
    def make_model(inputs, n_outputs):
        hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
        output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
        return output
    
    x = tf.placeholder("float", [None, 1], name="x")
    predictions = make_model(x, 1)
    
    
    prediction_file = pd.DataFrame()
    len_data = len(energies)
    predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
    predict_x = energies.reshape(len_data,1)
    
    ZLPs_gen = np.zeros((count, len_data))
    j=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(0,len(good_files)):
            if good_files[i] == 1:
                if specimen ==3:
                    best_model = 'Models_oud/Best_models/sp3/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                else:
                    best_model = 'Models_oud/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                saver = tf.train.Saver(max_to_keep=1000)
                saver.restore(sess, best_model)
    
                extrapolation = sess.run(predictions,
                                        feed_dict={
                                        x: predict_x
                                        })
                #prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(1000,)
                ZLPs_gen[j,:] = np.exp(extrapolation.reshape(len_data,))
                prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
                j += 1
    
    dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
    dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
    dE0 = np.round(dE1 - .5, 2) 
    
    return ZLPs_gen, dE0, dE1, dE2
    
    nbins = len_data
    li = []
    diff = []
    
    for i in range(0, len(prediction_file.columns)):
        df = pd.DataFrame()
        #df['x'] = predict_x.reshape(1000,)
        df['x'] = predict_x.reshape(len_data,)
        df['prediction'] = prediction_file.iloc[:,i]
        df['k'] = i
        li.append(df)
    
    extrapolation = pd.concat(li, axis=0, ignore_index = True)
    
    ZLPs_gen = np.zeros((count, len_data))

    for k in range(count): 
        exp_k = extrapolation[extrapolation['k'] == k ]
        #mean_k, var_k, count = binned_statistics(exp_k['x'], exp_k['prediction'], nbins)[0:3]
        
        mean_k = extrapolation[extrapolation['k'] == k ]['prediction']
        
        ZLPs_gen[k,:] =  np.exp(mean_k) #matching(energies,, data)
        
    return ZLPs_gen, dE0, dE1, dE2

def calc_ZLPs(data, energies, ZLPs_gen, dE0, dE1, dE2, specimen = 4):

    """
    #from Functions import *
    #if specimen == 3:
    #    from load_data_sp3 import *
    #else:
    #    from Load_data import *
    
    tf.reset_default_graph()
    #cols=['y14', 'x14', 'y15', 'x15', 'y16', 'x16', 'y17', 'x17', 'y19', 'x19', 'y20', 'x20', 'y21', 'x21', 'y22', 'x22', 'y23', 'x23']
    
    #ZLP_data = pd.concat((file14, file15, file16, file17, file19, file20, file21, file22, file23), axis=1)
    #ZLP_data = ZLP_data.drop(['x', 'y_norm'],axis=1).rename(columns={'x_shifted': 'x'})
    #ZLP_data.columns = cols
    
    #print(ZLP_data)
    
    ## Window the data file to the desired energy range
    #E_min = -.3
    #CHANGE
    #if specimen ==3:
    #    E_min = -0.93
    #    E_max = 9.07
    #else:
    #    E_min= -4   
    #    E_max = 20
    #original = ZLP_data[(ZLP_data['x14'] >= E_min) & (ZLP_data['x14'] <= E_max)]
    
    if specimen == 3:
        d_string = '06.12.2020'
        path_to_data = 'Data_oud/Results/sp3/%(date)s/'% {"date": d_string} 
    else:
        d_string = '07.09.2020'
        path_to_data = 'Data_oud/Results/%(date)s/'% {"date": d_string} 
    
    path_predict = r'Predictions_*.csv'
    path_cost = r'Cost_*.csv' 
    
    all_files = glob.glob(path_to_data + path_predict)
    
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, delimiter=",",  header=0, usecols=[0,1,2], names=['x', 'y', 'pred'])
        li.append(df)
        
    
    training_data = pd.concat(li, axis=0, ignore_index=True)
    
    
    all_files_cost = glob.glob(path_to_data + path_cost)
    
    
    import natsort
    
    all_files_cost_sorted = natsort.natsorted(all_files_cost)
    
    chi2_array = []
    chi2_index = []
    
    for filename in all_files_cost_sorted:
        df = pd.read_csv(filename, delimiter=",", header=0, usecols=[0,1], names=['train', 'test'])
        best_try = np.argmin(df['test'])
        chi2_array.append(df.iloc[best_try,0])
        chi2_index.append(best_try)
    
    chi_data  = pd.DataFrame()
    chi_data['Best chi2 value'] = chi2_array
    chi_data['Epoch'] = chi2_index
        
    
    
    good_files = []
    count = 0
    threshold = 3
    
    for i,j in enumerate(chi2_array):
        if j < threshold:
            good_files.append(1) 
            count +=1 
        else:
            good_files.append(0)
    
    
    
    
    
    tf.get_default_graph
    tf.disable_eager_execution()
    
    def make_model(inputs, n_outputs):
        hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
        output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
        return output
    
    x = tf.placeholder("float", [None, 1], name="x")
    predictions = make_model(x, 1)
    
    
    prediction_file = pd.DataFrame()
    len_data = len(energies)
    predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
    predict_x = energies.reshape(len_data,1)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(0,len(good_files)):
            if good_files[i] == 1:
                if specimen ==3:
                    best_model = 'Models_oud/Best_models/sp3/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                else:
                    best_model = 'Models_oud/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                saver = tf.train.Saver(max_to_keep=1000)
                saver.restore(sess, best_model)
    
                extrapolation = sess.run(predictions,
                                        feed_dict={
                                        x: predict_x
                                        })
                #prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(1000,)
                prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
    
    
    dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
    dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
    dE0 = np.round(dE1 - .5, 2) 
    """
    
    
    ### Definition for the matching procedure
    
    def matching(x, y_NN, y_ZLP):
        
        total = pd.DataFrame({"x": x, "prediction y": y_NN, "data ZLP": y_ZLP})
        
        delta = np.divide((dE1 - dE0), 3)
    
        factor_NN = np.exp(- np.divide((x[(x<dE1) & (x >= dE0)] - dE1)**2, delta**2))
        factor_ZLP = 1 - factor_NN
        
        range_0 = total[total['x'] < dE0]['data ZLP'] * 1
        range_1 = total[(total['x'] < dE1) & (total['x'] >= dE0)]['prediction y'] * factor_NN + total[(total['x'] < dE1) & (total['x'] >= dE0)]['data ZLP'] * factor_ZLP
        range_2 = total[(total['x'] >= dE1) & (total['x'] < 3 * dE2)]['prediction y'] * 1 
        range_3 = total[(total['x'] >= 3 * dE2)]['prediction y'] * 0
        totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)
        
        return totalfile
    
    
    
    """
    nbins = len_data
    li = []
    diff = []
    
    for i in range(0, len(prediction_file.columns)):
        df = pd.DataFrame()
        #df['x'] = predict_x.reshape(1000,)
        df['x'] = predict_x.reshape(len_data,)
        df['prediction'] = prediction_file.iloc[:,i]
        df['k'] = i
        li.append(df)
    
    extrapolation = pd.concat(li, axis=0, ignore_index = True)
    """
    ### Window the prediction data to the same energy range as the original spectra
        
    #extrapolation = extrapolation[(extrapolation['x'] >= E_min) & (extrapolation['x'] <= E_max)]
    
    
    
    
    lo = []
    count = ZLPs_gen.shape[0]
    ZLPs = np.zeros(ZLPs_gen.shape) #np.zeros((count, len_data))
    
    
    for k in range(count): 
        #exp_k = extrapolation[extrapolation['k'] == k ]
        #mean_k, var_k, count = binned_statistics(exp_k['x'], exp_k['prediction'], nbins)[0:3]
        
        #mean_k = extrapolation[extrapolation['k'] == k ]['prediction']
        
        ZLPs[k,:] = matching(energies, ZLPs_gen[k,:], data)#matching(energies, np.exp(mean_k), data)
        
    return ZLPs
        
    #total_replicas = pd.concat(lo, axis=0, ignore_index = True, sort=False)
    
    ### Subtracted spectra:  difference = original spectrum - matched spectrum
    
    #for i in ([14, 15, 16, 19, 20, 21]):
    #    total_replicas['dif%(i)s'%{"i": i}] = total_replicas['data y%(i)s'%{"i": i}] - total_replicas['match%(i)s'%{"i": i}]
    
    #total_replicas.to_csv('Data/Results/Replica_files/final_%(s)s' % {"s": dE1})


def crossings_im(die_fun_im, deltaE, delta = 50):
    crossings_E = np.zeros((die_fun_im.shape[0], die_fun_im.shape[1],1))
    crossings_n = np.zeros((die_fun_im.shape[0], die_fun_im.shape[1]))
    n_max = 1
    for i in range(die_fun_im.shape[0]):
        print("cross", i)
        for j in range(die_fun_im.shape[1]): 
            #print("cross", i, j)
            crossings_E_ij, n = crossings(die_fun_im[i,j,:], deltaE, delta)
            if n > n_max:
                #print("cross", i, j, n, n_max, crossings_E.shape)
                crossings_E_new = np.zeros((die_fun_im.shape[0], die_fun_im.shape[1],n))
                print("cross", i, j, n, n_max, crossings_E.shape, crossings_E_new[:,:,:n_max].shape)
                crossings_E_new[:,:,:n_max] = crossings_E
                crossings_E = crossings_E_new
                n_max = n
                del crossings_E_new
            crossings_E[i,j,:n] = crossings_E_ij
            crossings_n[i,j] = n
    return crossings_E, crossings_n

def crossings(die_fun, deltaE, delta = 50):
    l = len(die_fun)
    die_fun_avg = np.zeros(l-delta)
    #die_fun_f = np.zeros(l-2*delta)
    for i in range(l-delta):
        die_fun_avg[i] = np.average(die_fun[i:i+delta])
    
    crossing = (die_fun_avg[:-delta]<0) * (die_fun_avg[delta:] >=0)
    deltaE = deltaE[deltaE>0]
    deltaE = deltaE[50:-50]
    crossing_E = deltaE[crossing]
    n = len(crossing_E)
    return crossing_E, n
    
#%%


#data = np.load("area03-eels-SI-aligned.npy")
#energies = np.load("area03-eels-SI-aligned_energy.npy")
    

#dielectric_function_im_avg, dielectric_function_im_std = im_dielectric_function(data, energies)


#%%
crossings_E, crossings_n =  crossings_im(dielectric_function_im_avg, energies)


#%%

plt.figure()
#plt.imshow(crossings_n, cmap='hot', interpolation='nearest')
#plt.


ax = sns.heatmap(crossings_n)
plt.show()

#%%
dmfile = dm.fileDM('/path/to/area03-eels-SI-aligned.dm4')
data2 = dmfile.getDataset(0)
