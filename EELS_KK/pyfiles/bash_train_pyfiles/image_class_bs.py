# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import seaborn as sns
import natsort
import numpy as np
import math
from scipy.fftpack import next_fast_len
from scipy.optimize import curve_fit
import logging
from ncempy.io import dm
import os
import copy
import pickle
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import bz2
import pickle
import _pickle as cPickle

from k_means_clustering import k_means
from train_nn_torch_bs import train_nn_scaled

_logger = logging.getLogger(__name__)


class Spectral_image():

    """
            This is the spectral image class that provides several tools to analyse spectral images with the zero-loss peak
            subtracted.

            Parameters
            ----------
            data: ndarray
                Spectral image data as 3D numpy array (x-axis, y-axis, energy loss)
            deltadeltaE: float
                Binwidth in energy loss spectrum
            pixelsize: float, optional
                Width of pixels (the default is None, which implies ...)
            beam_energy: float, optional
                Energy of electron beam [eV] (default is None, which implies ...)
            collection_angle: float, optional
                Collection angle of STEM [rad] (default is None, which implies ...)
            name: str, optional
                Title of the plots (default is None, in which case no title is passed)
            dielectric_function_im_avg
                average dielectric function for each pixel
            dielectric_function_im_std
                standard deviation of the dielectric function at each energy for each pixel
            S_s_avg
                average surface scattering distribution for each pixel
            S_s_std
                standard deviation of the surface scattering distribution at each energy for each pixel
            thickness_avg
                average thickness for each pixel
            IEELS_avg
                average bulk scattering distribution for each pixel
            IEELS_std
                standard deviation of the bulk scattering distribution at each energy for each pixel
    """


    #SIGNAL NAMES

    DIELECTRIC_FUNCTION_NAMES = ['dielectric_function', 'dielectricfunction', 'dielec_func', 'die_fun', 'df', 'epsilon']
    EELS_NAMES = ['electron_energy_loss_spectrum', 'electron_energy_loss', 'EELS', 'EEL', 'energy_loss', 'data']
    IEELS_NAMES = ['inelastic_scattering_energy_loss_spectrum', 'inelastic_scattering_energy_loss',
                   'inelastic_scattering', 'IEELS', 'IES']
    ZLP_NAMES = ['zeros_loss_peak', 'zero_loss', 'ZLP', 'ZLPs', 'zlp', 'zlps']
    THICKNESS_NAMES = ['t', 'thickness', 'thick', 'thin']
    POOLED_ADDITION = ['pooled', 'pool', 'p', '_pooled', '_pool', '_p']

    # META DATA NAMES
    COLLECTION_ANGLE_NAMES = ["collection_angle", "col_angle", "beta"]
    BEAM_ENERGY_NAMES = ["beam_energy", "beam_E", "E_beam", "E0", "E_0"]

    m_0 = 5.1106E5  # eV, electron rest mass
    a_0 = 5.29E-11  # m, Bohr radius
    h_bar = 6.582119569E-16  # eV/s
    c = 2.99792458E8  # m/s

    def __init__(self, data, deltadeltaE, pixelsize=None, beam_energy=None, collection_angle=None, name=None,
                 dielectric_function_im_avg=None, dielectric_function_im_std=None,S_s_avg=None, S_s_std=None,
                 thickness_avg=None,thickness_std=None, IEELS_avg=None, IEELS_std=None, **kwargs):
        """Constructor method"""

        self.data = data
        self.ddeltaE = deltadeltaE
        self.determine_deltaE()
        if pixelsize is not None:
            self.pixelsize = pixelsize * 1E6
        self.calc_axes()
        if beam_energy is not None:
            self.beam_energy = beam_energy
        if collection_angle is not None:
            self.collection_angle = collection_angle
        if name is not None:
            self.name = name

        self.dielectric_function_im_avg = dielectric_function_im_avg
        self.dielectric_function_im_std = dielectric_function_im_std
        self.S_s_avg = S_s_avg
        self.S_s_std = S_s_std
        self.thickness_avg = thickness_avg
        self.thickness_std = thickness_std
        self.IEELS_avg = IEELS_avg
        self.IEELS_std = IEELS_std

    def save_image(self, filename):
        """
        Function to save image, including all attributes, in pickle (.pkl) format. Image will be saved \
            at indicated location and name in filename input.

        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pkl", ".pkl" will be added.

        Returns
        -------
        None.

        """
        if filename[-4:] != '.pkl':
            filename + '.pkl'
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_compressed_image(self, filename):
        """
        Function to save image, including all attributes, in compressed pickle (.pbz2) format. Image will \
            be saved at indicated location and name in filename input. Advantage over save_image is that \
            the saved file is some orders smaller, disadvantage is that saving and reloading the image \
            takes significantly longer. 


        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pbz2", ".pbz2" will be added.

        Returns
        -------
        None.

        """
        if filename[-5:] != '.pbz2':
            filename + '.pbz2'
        self.compressed_pickle(filename, self)

    @staticmethod
    # Pickle a file and then compress it into a file with extension 
    def compressed_pickle(title, data):
        """
        Saves data at location title as compressed pickle.
        """
        with bz2.BZ2File(title, 'w') as f:
            cPickle.dump(data, f)

    @staticmethod
    def decompress_pickle(file):
        """
        Opens, decompresses and returns the pickle file at location file.
        """
        data = bz2.BZ2File(file, 'rb')
        data = cPickle.load(data)
        return data

    # %%GENERAL FUNCTIONS

    # %%PROPERTIES
    @property
    def l(self):
        """returns length of spectra, i.e. num energy loss bins"""
        return self.data.shape[2]

    @property
    def image_shape(self):
        """return 2D-shape of spectral image"""
        return self.data.shape[:2]

    @property
    def shape(self):
        """returns 3D-shape of spectral image"""
        return self.data.shape

    @property
    def n_clusters(self):
        """return number of clusters image is clustered into"""
        return len(self.clusters)

    @property
    def n_spectra(self):
        """returns number of spectra in specral image"""
        return np.product(self.image_shape)

    @classmethod
    def load_data(cls, path_to_dmfile, load_additional_data=False):
        """
        INPUT: 
            path_to_dmfile: str, path to spectral image file (.dm3 or .dm4 extension)
        OUTPUT:
            image -- Spectral_image, object of Spectral_image class containing the data of the dm-file
        """
        dmfile_tot = dm.fileDM(path_to_dmfile)
        additional_data = []
        for i in range(dmfile_tot.numObjects - dmfile_tot.thumbnail * 1):
            dmfile = dmfile_tot.getDataset(i)
            if dmfile['data'].ndim == 3:
                dmfile = dmfile_tot.getDataset(i)
                data = np.swapaxes(np.swapaxes(dmfile['data'], 0, 1), 1, 2)
                if not load_additional_data:
                    break
            elif load_additional_data:
                additional_data.append(dmfile_tot.getDataset(i))
            if i == dmfile_tot.numObjects - dmfile_tot.thumbnail * 1 - 1:
                if (len(additional_data) == i + 1) or not load_additional_data:
                    print("No spectral image detected")
                    dmfile = dmfile_tot.getDataset(0)
                    data = dmfile['data']

        # .getDataset(0)
        ddeltaE = dmfile['pixelSize'][0]
        pixelsize = np.array(dmfile['pixelSize'][1:])
        energyUnit = dmfile['pixelUnit'][0]
        ddeltaE *= cls.get_prefix(energyUnit, 'eV')
        pixelUnit = dmfile['pixelUnit'][1]
        pixelsize *= cls.get_prefix(pixelUnit, 'm')
        image = cls(data, ddeltaE, pixelsize=pixelsize, name=path_to_dmfile[:-4])
        if load_additional_data:
            image.additional_data = additional_data
        return image

    @classmethod
    def load_Spectral_image(cls, path_to_pickle):
        """
        Loads spectral image from a pickled file. 

        Parameters
        ----------
        path_to_pickle : str
            path to the pickeled image file.

        Raises
        ------
        ValueError
            If path_to_pickle does not end on the desired format .pkl.
        FileNotFoundError
            If path_to_pickle does not exists.

        Returns
        -------
        image : Specral_image
            Image (i.e. including all atributes) loaded from pickle file.

        """
        if path_to_pickle[-4:] != '.pkl':
            raise ValueError("please provide a path to a pickle file containing a Spectrall_image class object.")
            return
        if not os.path.exists(path_to_pickle):
            raise FileNotFoundError('pickled file: ' + path_to_pickle + ' not found')
        with open(path_to_pickle, 'rb') as pickle_im:
            image = pickle.load(pickle_im)
        return image

    @classmethod
    def load_compressed_Spectral_image(cls, path_to_compressed_pickle):
        """
        Loads spectral image from a compressed pickled file. Will take longer than loading from not compressed pickle.

        Parameters
        ----------
        path_to_compressed_pickle : str
            path to the compressed pickle image file.

        Raises
        ------
        ValueError
            If path_to_compressed_pickle does not end on the desired format .pbz2.
        FileNotFoundError
            If path_to_compressed_pickle does not exists.

        Returns
        -------
        image : Specral_image
            Image (i.e. including all atributes) loaded from compressed pickle file.


        """
        if path_to_compressed_pickle[-5:] != '.pbz2':
            raise ValueError("please provide a path to a compressed .pbz2 pickle file containing a Spectrall_image class object.")
            return
        if not os.path.exists(path_to_compressed_pickle):
            raise FileNotFoundError('pickled file: ' + path_to_compressed_pickle + ' not found')

        image = cls.decompress_pickle(path_to_compressed_pickle)
        return image

    def set_n(self, n, n_background=None):
        """
        Sets value of refractive index for the image as attribute self.n. If unclusered, n will be an \
            array of length one, otherwise it is an array of len n_clusters. If n_background is defined, \
            the cluster with the lowest thickness (cluster 0) will be assumed to be the vacuum/background, \
            and gets the value of the background refractive index.
            
        If there are more specimen present in the image, it is wise to check by hand what cluster belongs \
            to what specimen, and set the values by running image.n[cluster_i] = n_i.

        Parameters
        ----------
        n : float
            refractive index of sample.
        n_background : float, optional
            if defined: the refractive index of the background/vacuum. This value will automatically be \
            assigned to pixels belonging to the thinnest cluster. 

        Returns
        -------
        None.

        """
        if type(n) == float or type(n) == int:
            self.n = np.ones(self.n_clusters) * n
            if n_background is not None:
                # assume thinnest cluster (=cluster 0) is background
                self.n[0] = n_background 
        elif len(n) == self.n_clusters:
            self.n = n

    def determine_deltaE(self):
        """
        INPUT: 
            self
        
        Determines the delta energies of the spectral image, based on the delta delta energie,
        and the index on which the spectral image has on average the highest intesity, this 
        is taken as the zero point for the delta energy.
        """
        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        self.deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l - ind_max - 1) * self.ddeltaE, self.l)
        # return deltaE

    def calc_axes(self):
        """
        Calculates the attribustes x_axis and y_axis of the image. These are the spatial axes, and \
            can be used to find the spatial location of a pixel and are used in the plotting functions.
            
        If one wants to alter these axis, one can do this manually by running image.x_axis = ..., \
            and image.y_axis = ....

        Returns
        -------
        None.

        """
        self.y_axis = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])
        self.x_axis = np.linspace(0, self.image_shape[1] - 1, self.image_shape[1])
        if hasattr(self, 'pixelsize'):
            self.y_axis *= self.pixelsize[0]
            self.x_axis *= self.pixelsize[1]

            # %%RETRIEVING FUNCTIONS

    def get_data(self):  # TODO: add smooth possibility
        """returns spectra image data in 3D-numpy array (x-axis x y-axis x energy loss-axis)"""
        return self.data

    def get_deltaE(self):
        """returns energy loss axis in numpy array"""
        return self.deltaE

    def get_metadata(self):
        """returns list with values for beam_energy and collection_angle, if defined"""
        meta_data = {}
        if self.beam_energy is not None:
            meta_data['beam_energy'] = self.beam_energy
        if self.collection_angle is not None:
            meta_data['collection_angle'] = self.collection_angle
        return meta_data

    def get_pixel_signal(self, i, j, signal='EELS'):
        """
        INPUT:
            i: int, x-coordinate for the pixel
            j: int, y-coordinate for the pixel
        Keyword argument:
            signal: str (default = 'EELS'), what signal is requested, should comply with defined names
        OUTPUT:
            signal: 1D numpy array, array with the requested signal from the requested pixel
        """
        # TODO: add alternative signals + names
        if signal in self.EELS_NAMES:
            return np.copy(self.data[i, j, :])
        elif signal == "pooled":
            return np.copy(self.pooled[i, j, :])
        elif signal in self.DIELECTRIC_FUNCTION_NAMES:
            return np.copy(self.dielectric_function_im_avg[i, j, :])
        else:
            print("no such signal", signal, ", returned general EELS signal.")
            return np.copy(self.data[i, j, :])

    def get_image_signals(self, signal='EELS'):
        # TODO: add alternative signals + names
        if signal in self.EELS_NAMES:
            return np.copy(self.data)
        elif signal == "pooled":
            return np.copy(self.pooled)
        elif signal in self.DIELECTRIC_FUNCTION_NAMES:
            return np.copy(self.dielectric_function_im_avg)
        else:
            print("no such signal", signal, ", returned general EELS data.")
            return np.copy(self.data)

    def get_cluster_spectra(self, conf_interval=1, clusters=None, save_as_attribute=False, based_upon="log",
                            signal="EELS"):
        """
        Parameters
        ----------
        conf_interval : float, optional
            The ratio of spectra returned. The spectra are selected based on the 
            based_upon value. The default is 0.68.
        clusters : list of ints, optional #TODO: finish
            DESCRIPTION. The default is None.
        save_as_attribute : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        cluster_data : np.array of type object, filled with 2D numpy arrays
            Each cell of the super numpy array is filled with the data of all spectra 
            with in one of the requested clusters.
            
        Atributes
        ---------
        self.cluster_data: np.array of type object, filled with 2D numpy arrays
            If save_as_attribute set to True, the cluster data is also saved as attribute

        """
        # TODO: check clustering before everything
        if clusters is None:
            clusters = range(self.n_clusters)

        integrated_I = np.sum(self.data, axis=2)
        cluster_data = np.zeros(len(clusters), dtype=object)

        j = 0
        for i in clusters:
            data_cluster = self.get_image_signals(signal)[self.clustered == i]
            if conf_interval < 1:
                intensities_cluster = integrated_I[self.clustered == i]
                arg_sort_I = np.argsort(intensities_cluster)
                ci_lim = round((1 - conf_interval) / 2 * intensities_cluster.size)  # TODO: ask juan: round up or down?
                data_cluster = data_cluster[arg_sort_I][ci_lim:-ci_lim]
            # intensities_cluster = np.ones(len(intensities_cluster)-2*ci_lim)*self.clusters[i]
            cluster_data[j] = data_cluster
            j += 1

        if save_as_attribute:
            self.cluster_data = cluster_data
        else:
            return cluster_data
        
    def deltaE_to_arg(self, E):
        if type(E) in [int, float]:
            return np.argmin(np.absolute(self.deltaE-E))
        
        for i in len(E):
            E[i] = np.argmin(np.absolute(self.deltaE-E[i]))
        return E
        #TODO: check if works
            

    # %%METHODS ON SIGNAL

    def cut(self, E1=None, E2=None, in_ex="in"):
        """
        Parameters
        ----------
        E1 : TYPE, optional
            DESCRIPTION. The default is None.
        E2 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """
        if (E1 is None) and (E2 is None):
            raise ValueError("To cut energy specra, please specify minimum energy E1 and/or maximum energy E2.")
        if E1 is None:
            E1 = self.deltaE.min() - 1
        if E2 is None:
            E2 = self.deltaE.max() + 1
        if in_ex == "in":
            select = ((self.deltaE >= E1) & (self.deltaE <= E2))
        else:
            select = ((self.deltaE > E1) & (self.deltaE < E2))
        self.data = self.data[:, :, select]
        self.deltaE = self.deltaE[select]
        # TODO add selecting of all attributes
        pass

    def cut_image(self, range_width, range_height):
        # TODO: add floats for cutting to meter sizes?
        self.data = self.data[range_height[0]:range_height[1], range_width[0]:range_width[1]]
        self.y_axis = self.y_axis[range_height[0]:range_height[1]]
        self.x_axis = self.x_axis[range_width[0]:range_width[1]]

    # TODO
    def samenvoegen(self):
        pass

    def smooth(self, window_len=10, window='hanning', keep_original=False):
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
        output:
            the smoothed signal
    
        """
        # TODO: add comnparison
        window_len += (window_len + 1) % 2
        s = np.r_['-1', self.data[:, :, window_len - 1:0:-1], self.data, self.data[:, :, -2:-window_len - 1:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        # y=np.convolve(w/w.sum(),s,mode='valid')
        surplus_data = int((window_len - 1) * 0.5)
        if keep_original:
            self.data_smooth = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=2, arr=s)[
                               :, :, surplus_data:-surplus_data]
        else:
            self.data = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=2, arr=s)[:, :,
                        surplus_data:-surplus_data]

        return  # y[(window_len-1):-(window_len)]

    def deconvolute(self, i, j, ZLP, signal='EELS'):

        y = self.get_pixel_signal(i, j, signal)
        r = 3  # Drude model, can also use estimation from exp. data
        A = y[-1]
        n_times_extra = 2
        sem_inf = next_fast_len(n_times_extra * self.l)

        y_extrp = np.zeros(sem_inf)
        y_ZLP_extrp = np.zeros(sem_inf)
        x_extrp = np.linspace(self.deltaE[0] - self.l * self.ddeltaE,
                              sem_inf * self.ddeltaE + self.deltaE[0] - self.l * self.ddeltaE, sem_inf)

        x_extrp = np.linspace(self.deltaE[0], sem_inf * self.ddeltaE + self.deltaE[0], sem_inf)

        y_ZLP_extrp[:self.l] = ZLP
        y_extrp[:self.l] = y
        x_extrp[:self.l] = self.deltaE[-self.l:]

        y_extrp[self.l:] = A * np.power(1 + x_extrp[self.l:] - x_extrp[self.l], -r)

        x = x_extrp
        y = y_extrp
        y_ZLP = y_ZLP_extrp

        z_nu = CFT(x, y_ZLP)
        i_nu = CFT(x, y)
        abs_i_nu = np.absolute(i_nu)
        N_ZLP = 1  # scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)

        s_nu = N_ZLP * np.log(i_nu / z_nu)
        j1_nu = z_nu * s_nu / N_ZLP
        S_E = np.real(iCFT(x, s_nu))
        s_nu_nc = s_nu
        s_nu_nc[500:-500] = 0
        S_E_nc = np.real(iCFT(x, s_nu_nc))
        J1_E = np.real(iCFT(x, j1_nu))

        return J1_E[:self.l]

    def pool(self, n_p):
        # TODO: add gaussian options ed??
        if n_p % 2 == 0:
            print("Unable to pool with even number " + str(n_p) + ", continuing with n_p=" + str(n_p + 1))
            n_p += 1
        pooled = np.zeros(self.shape)
        n_p_border = int(math.floor(n_p / 2))
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                min_x = max(0, i - n_p_border)
                max_x = min(self.image_shape[0], i + 1 + n_p_border)
                min_y = max(0, j - n_p_border)
                max_y = min(self.image_shape[1], j + 1 + n_p_border)
                pooled[i, j] = np.average(np.average(self.data[min_x:max_x, min_y:max_y, :], axis=1), axis=0)
        self.pooled = pooled

    # %%METHODS ON ZLP
    # CALCULATING ZLPs FROM PRETRAINDED MODELS

    
    def calc_ZLPs2(self, i,j, signal = 'EELS', select_ZLPs = True, **kwargs):
        ### Definition for the matching procedure
        signal = self.get_pixel_signal(i, j, signal)

        if not hasattr(self, 'ZLP_models'):
            try:
                self.load_ZLP_models_smefit(**kwargs)
            except:
                self.load_ZLP_models_smefit()
        if not hasattr(self, 'ZLP_models'):
            ans = input("No ZLP models found. Please specify directory or train models. \n" +
                        "Do you want to define path to models [p], train models [t] or quit [q]?\n")
            if ans[0] == "q":
                return
            elif ans[0] == "p":
                path_to_models = input("Please input path to models: \n")
                try:
                    self.load_ZLP_models(**kwargs)
                except:
                    self.load_ZLP_models()
                if not hasattr(self, 'ZLP_models'):
                    print("You had your chance. Please locate your models.")
                    return
            elif ans[0] == "t":
                try:
                    self.train_ZLPs(**kwargs)
                except:
                    self.train_ZLPs()
                if "path_to_models" in kwargs:
                    path_to_models = kwargs["path_to_models"]
                    self.load_ZLP_models(path_to_models)
                else:
                    self.load_ZLP_models()
            else:
                print("unvalid input, not calculating ZLPs")
                return

        cluster = self.clustered[i, j]

        # TODO: aanpassen
        def matching(signal, gen_i_ZLP, dE1):
            dE0 = dE1 - 0.5
            dE2 = dE1*4
            #gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]#*np.max(signal)/np.max(self.ZLPs_gen[ind_ZLP,:]) #TODO!!!!, normalize?
            delta = (dE1 - dE0)/10 #lau: 3
            
            # factor_NN = np.exp(- np.divide((self.deltaE[(self.deltaE<dE1) & (self.deltaE >= dE0)] - dE1)**2, delta**2))
            factor_NN = 1/(1+np.exp(-(self.deltaE[(self.deltaE<dE1) & (self.deltaE >= dE0)] - (dE0+dE1)/2)/delta))
            factor_dm = 1 - factor_NN

            range_0 = signal[self.deltaE < dE0]
            range_1 = gen_i_ZLP[(self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_NN + signal[
                (self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_dm
            range_2 = gen_i_ZLP[(self.deltaE >= dE1) & (self.deltaE < 3 * dE2)]
            range_3 = gen_i_ZLP[(self.deltaE >= 3 * dE2)] * 0
            totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)
            # TODO: now hardcoding no negative values!!!! CHECKKKK
            totalfile = np.minimum(totalfile, signal)
            return totalfile

        count = len(self.ZLP_models)
        ZLPs = np.zeros((count, self.l))  # np.zeros((count, len_data))

        if not hasattr(self, "scale_var_deltaE"):
            self.scale_var_deltaE = find_scale_var(self.deltaE)

        if not hasattr(self, "scale_var_log_sum_I"):
            all_spectra = self.data
            all_spectra[all_spectra < 1] = 1
            int_log_I = np.log(np.sum(all_spectra, axis=2)).flatten()
            self.scale_var_log_sum_I = find_scale_var(int_log_I)
            del all_spectra

        log_sum_I_pixel = np.log(np.sum(signal))
        predict_x_np = np.zeros((self.l, 2))
        predict_x_np[:, 0] = scale(self.deltaE, self.scale_var_deltaE)
        predict_x_np[:, 1] = scale(log_sum_I_pixel, self.scale_var_log_sum_I)

        predict_x = torch.from_numpy(predict_x_np)
        
        dE1 = self.dE1[1,int(cluster)]
        print("cluster:", cluster, ", dE1:", dE1)
        for k in range(count): 
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions = np.exp(model(predict_x.float()).flatten())
            ZLPs[k,:] = matching(signal, predictions, dE1)#matching(energies, np.exp(mean_k), data)
        
        if select_ZLPs:
            ZLPs = ZLPs[self.select_ZLPs(ZLPs, dE1)]
        
        return ZLPs


    def calc_ZLPs(self, i,j, signal = 'EELS', select_ZLPs = True, **kwargs):
        ### Definition for the matching procedure
        
        #TODO: aanpassen
        def matching( signal, gen_i_ZLP, dE1):
            dE0 = dE1 - 0.5
            dE2 = dE1*4
            #gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]#*np.max(signal)/np.max(self.ZLPs_gen[ind_ZLP,:]) #TODO!!!!, normalize?
            delta = (dE1 - dE0)/10 #lau: 3
            
            # factor_NN = np.exp(- np.divide((self.deltaE[(self.deltaE<dE1) & (self.deltaE >= dE0)] - dE1)**2, delta**2))
            factor_NN = 1/(1+np.exp(-(self.deltaE[(self.deltaE<dE1) & (self.deltaE >= dE0)] - (dE0+dE1)/2)/delta))
            factor_dm = 1 - factor_NN
            
            range_0 = signal[self.deltaE < dE0]
            range_1 = gen_i_ZLP[(self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_NN + signal[(self.deltaE < dE1) & (self.deltaE >= dE0)] * factor_dm
            range_2 = gen_i_ZLP[(self.deltaE >= dE1) & (self.deltaE < 3 * dE2)]
            range_3 = gen_i_ZLP[(self.deltaE >= 3 * dE2)] * 0
            totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)
            #TODO: now hardcoding no negative values!!!! CHECKKKK
            totalfile = np.minimum(totalfile, signal)
            return totalfile
        
        
        ZLPs_gen = self.calc_gen_ZLPs(i,j, signal, select_ZLPs, **kwargs)
        
        count = len(ZLPs_gen)
        ZLPs = np.zeros((count, self.l)) #np.zeros((count, len_data))
        
        
        signal = self.get_pixel_signal(i,j, signal)
        cluster = self.clustered[i,j]
        
        dE1 = self.dE1[1,int(cluster)]
        for k in range(count): 
            predictions = ZLPs_gen[k]
            ZLPs[k,:] = matching(signal, predictions, dE1)#matching(energies, np.exp(mean_k), data)
        return ZLPs

    
    def calc_gen_ZLPs(self, i,j, signal = "eels", select_ZLPs = True, **kwargs):
        ### Definition for the matching procedure
        signal = self.get_pixel_signal(i,j, signal)
        
        if not hasattr(self, 'ZLP_models'):
            try:
                self.load_ZLP_models(**kwargs)
            except:
                self.load_ZLP_models()
        if not hasattr(self, 'ZLP_models'):
            ans = input("No ZLP models found. Please specify directory or train models. \n" +
                        "Do you want to define path to models [p], train models [t] or quit [q]?\n")
            if ans[0] == "q":
                return
            elif ans[0] == "p":
                path_to_models = input("Please input path to models: \n")
                try:
                    self.load_ZLP_models(**kwargs)
                except:
                    self.load_ZLP_models()
                if not hasattr(self, 'ZLP_models'):
                    print("You had your chance. Please locate your models.")
                    return
            elif ans[0] == "t":
                try:
                    self.train_ZLPs(**kwargs)
                except:
                    self.train_ZLPs()
                if "path_to_models" in kwargs:
                    path_to_models = kwargs["path_to_models"]
                    self.load_ZLP_models(path_to_models)
                else:
                    self.load_ZLP_models()
            else:
                print("unvalid input, not calculating ZLPs")
                return

        count = len(self.ZLP_models)
        predictions = np.zeros((count, self.l))  # np.zeros((count, len_data))

        if not hasattr(self, "scale_var_deltaE"):
            self.scale_var_deltaE = find_scale_var(self.deltaE)

        if not hasattr(self, "scale_var_log_sum_I"):
            all_spectra = self.data
            all_spectra[all_spectra < 1] = 1
            int_log_I = np.log(np.sum(all_spectra, axis=2)).flatten()
            self.scale_var_log_sum_I = find_scale_var(int_log_I)
            del all_spectra

        log_sum_I_pixel = np.log(np.sum(signal))
        predict_x_np = np.zeros((self.l, 2))
        predict_x_np[:, 0] = scale(self.deltaE, self.scale_var_deltaE)
        predict_x_np[:, 1] = scale(log_sum_I_pixel, self.scale_var_log_sum_I)

        predict_x = torch.from_numpy(predict_x_np)

        for k in range(count):
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions[k,:] = np.exp(model(predict_x.float()).flatten())
        
        if select_ZLPs:
            predictions = predictions[self.select_ZLPs(predictions)]
        
        return predictions
    
    def select_ZLPs(self, ZLPs, dE1 = None):
        if dE1 is None:
            dE1 = min(self.dE1[1,:])
            dE2 = 3*max(self.dE1[1,:])
        else:
            dE2 = 3*dE1
    
        ZLPs_c = ZLPs[:,(self.deltaE>dE1) & (self.deltaE<dE2)]
        low = np.nanpercentile(ZLPs_c, 1, axis=0)
        high = np.nanpercentile(ZLPs_c, 90, axis=0)
    
        threshold = (low[0]+high[0])/20
    
        low[low<threshold] = 0
        high[high<threshold] = threshold
    
        check = (ZLPs_c<low)|(ZLPs_c>=high)
        check_sum = np.sum(check, axis=1)/check.shape[1]
    
        threshold = np.nanpercentile(check_sum, 85)
        
        return (check_sum<=threshold)


    def select_ZLPs_old(self, ZLPs, dE1 = None):
        if dE1 is None:
            dE1 = min(self.dE1[1,:])
            dE2 = 3*max(self.dE1[1,:])
        else:
            dE2 = 3*dE1
        
        ZLPs_c = ZLPs[:,(self.deltaE>dE1) & (self.deltaE<dE2)]
        low = np.nanpercentile(ZLPs_c, 2, axis=0)
        high = np.nanpercentile(ZLPs_c, 95, axis=0)
        
        threshold = (low[0]+high[0])/100
        
        low[low<threshold] = 0
        high[high<threshold] = threshold
        
        check = (ZLPs_c<low)|(ZLPs_c>=high)
        check = np.sum(check, axis=1)/check.shape[1]
        
        threshold = 0.01
        
        return (check<threshold)
    
    def train_ZLPs(self, n_clusters = None, conf_interval = 1, clusters = None, signal = 'EELS', **kwargs):
        if not hasattr(self, "clustered"):
            if n_clusters is not None:
                self.cluster(n_clusters)
            else:
                self.cluster()
        elif n_clusters is not None and self.n_clusters != n_clusters:
            self.cluster(n_clusters)

        training_data = self.get_cluster_spectra(conf_interval=conf_interval, clusters=clusters, signal=signal)
        # self.models =
        # train_nn_scaled(self, training_data, **kwargs)
        train_nn_scaled(self, training_data, **kwargs)

    def load_ZLP_models(self, path_to_models="models", threshold_costs=1, name_in_path=True, plotting=False):
        if hasattr(self, "name") and name_in_path:
            path_to_models = self.name + "_" + path_to_models

        if not os.path.exists(path_to_models):
            print("No path " + path_to_models + " found. Please ensure spelling and that there are models trained.")
            return

        self.ZLP_models = []

        model = MLP(num_inputs=2, num_outputs=1)

        costs = np.loadtxt(path_to_models + "/costs.txt")

        if plotting:
            plt.figure()
            plt.title("chi^2 distribution of models")
            plt.hist(costs[costs < threshold_costs * 3], bins=20)
            plt.xlabel("chi^2")
            plt.ylabel("number of occurence")

        n_working_models = np.sum(costs < threshold_costs)

        k = 0
        for j in range(len(costs)):
            if costs[j] < threshold_costs:
                with torch.no_grad():
                    model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(j)))
                    if self.check_model(model):
                        #TODO: this check is unnesscicary I believe, to be removed
                        self.ZLP_models.append(copy.deepcopy(model))
                    else:
                        print("disregarded model, straight line.")
                k+=1
    
    @staticmethod
    def check_cost_smefit(path_to_models, idx, threshold=1):
        path_to_models += (path_to_models[-1] != '/') * '/'
        cost = np.loadtxt(path_to_models + "costs" + str(idx) + ".txt")
        return cost < threshold

    @staticmethod
    def check_model(model):
        deltaE = np.linspace(0.1, 0.9, 1000)
        predict_x_np = np.zeros((1000, 2))
        predict_x_np[:, 0] = deltaE
        predict_x_np[:, 1] = 0.5

        predict_x = torch.from_numpy(predict_x_np)
        with torch.no_grad():
            predictions = np.exp(model(predict_x.float()).flatten().numpy())
        
        return (np.std(predictions)/np.average(predictions)) > 1E-3 #very small --> straight line
    
    def load_ZLP_models_smefit(self, path_to_models = "models", threshold_costs = 0.3, name_in_path = False, plotting = False, idx = None):
        # if n_rep is None and idx is None:
        #     print("Please spectify either the number of replicas you wish to load (n_rep)"+\
        #           " or the specific replica model you wist to load (idx) in load_ZLP_models_smefit.")
        #     return
        if hasattr(self, "name") and name_in_path:
            path_to_models = self.name + "_" + path_to_models

        if not os.path.exists(path_to_models):
            print(
                "No path " + os.getcwd() + path_to_models + " found. Please ensure spelling and that there are models trained.")
            return

        self.ZLP_models = []

        path_to_models += (path_to_models[-1] != '/') * '/'
        path_dE1 = "dE1.txt"
        model = MLP(num_inputs=2, num_outputs=1)
        self.dE1 = np.loadtxt(path_to_models + path_dE1)
        
        path_scale_var = 'scale_var.txt' #HIER
        self.scale_var_log_sum_I = np.loadtxt(path_to_models + path_scale_var)
        try:
            path_scale_var_deltaE = 'scale_var_deltaE.txt'
            self.scale_var_deltaE = np.loadtxt(path_to_models + path_scale_var_deltaE)
            print("found delta E vars")
        except:
            pass
        
        if hasattr(self, "clustered"):
            if self.n_clusters != self.dE1.shape[1]:
                print("image clustered in ", self.n_clusters, " clusters, but ZLP-models take ", self.dE1.shape[1],
                      " clusters, reclustering based on models.")
                self.cluster_on_cluster_values(self.dE1[0, :])
        else:
            self.cluster_on_cluster_values(self.dE1[0, :])

        if idx is not None:
            with torch.no_grad():
                model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(idx)))
            self.ZLP_models.append(copy.deepcopy(model))
            return

        path_costs = "costs"
        files_costs = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_costs)]
        idx_costs = np.array([int(s.replace(path_costs,"").replace(".txt","")) for s in files_costs])
        path_model_rep = "nn_rep"
        files_model_rep = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_model_rep)]
        idx_models = np.array([int(s.replace(path_model_rep,"").replace(".txt","")) for s in files_model_rep])
        
        overlap_idx = np.intersect1d(idx_costs, idx_models)
        
        n_rep = len(overlap_idx)
        costs = np.zeros(n_rep)
        files_models = np.zeros(n_rep, dtype='U12') #reads max 999,999 models, you really do not need more.
        
        # for i in range(n_rep):
        #     file = files_costs[i]
        #     costs[i] =  np.loadtxt(path_to_models + file)
        
        
        for i in range(n_rep):
            j = overlap_idx[i]
            idx_cost = np.argwhere(idx_costs == j)[0,0]
            idx_model = np.argwhere(idx_models == j)[0,0]
            
            file = files_costs[idx_cost]
            costs[i] = np.loadtxt(path_to_models + file)
            
            files_models[i] = files_model_rep[idx_model]
            
        
        self.costs = costs[costs<threshold_costs]
    
        if plotting:
            plt.figure()
            plt.title("chi^2 distribution of models")
            plt.hist(costs[costs < threshold_costs * 3], bins=20)
            plt.xlabel("chi^2")
            plt.ylabel("number of occurence")   
        
        
        for j in range(n_rep):
            if costs[j] < threshold_costs:
                file = files_models[j]
                if os.path.getsize(path_to_models + file) > 0:
                    with torch.no_grad():
                        model.load_state_dict(torch.load(path_to_models + file))
                    if self.check_model(model):
                        self.ZLP_models.append(copy.deepcopy(model))
                

    # METHODS ON DIELECTRIC FUNCTIONS
    def calc_thickness(self, I_EELS, n, N_ZLP=1):
        """
        Calculates thickness from sample data, using Egreton #TODO: bron.
        Nota bene: does not correct for surface scatterings. If you wish to correct \
        for surface scatterings, please extract t from kramer_kronig_hs()

        Parameters
        ----------
        I_EELS : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.
        N_ZLP: float/int
            default = 1. Default for already normalized I_EELS spectra.

        Returns
        -------
        None.

        """
        me = self.m_0
        e0 = self.e0
        beta = self.beta

        eaxis = self.deltaE[self.deltaE > 0]  # axis.axis.copy()
        y = I_EELS[self.deltaE > 0]
        i0 = N_ZLP

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2
        tgt = e0 * (2 * me + e0) / (me + e0)

        # Calculation of the ELF by normalization of the SSD
        # We start by the "angular corrections"
        Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE  # axis.scale

        K = np.sum(Im / eaxis) * self.ddeltaE
        K = (K / (np.pi / 2) / (1 - 1. / n ** 2))
        te = (332.5 * K * ke / i0)

        return te

    def kramers_kronig_hs(self, I_EELS,
                          N_ZLP=None,
                          iterations=1,
                          n=None,
                          t=None,
                          delta=0.5, correct_S_s=False):
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


        e0 = self.e0
        beta = self.beta

        eaxis = self.deltaE[self.deltaE > 0]  # axis.axis.copy()
        S_E = I_EELS[self.deltaE > 0]
        y = I_EELS[self.deltaE > 0]
        l = len(eaxis)
        i0 = N_ZLP

        # Kinetic definitions
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2 #m0 v**2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me) #me c**2 / (hbar c) gamma sqrt(2Ekin /(me c**2))
 
        for io in range(iterations):
            # Calculation of the ELF by normalization of the SSD
            # We start by the "angular corrections"
            Im = y / (np.log(1 + (beta * tgt / eaxis) ** 2)) / self.ddeltaE  # axis.scale
            if n is None and t is None:
                raise ValueError("The thickness and the refractive index are "
                                 "not defined. Please provide one of them.")
            elif n is not None and t is not None:
                raise ValueError("Please provide the refractive index OR the "
                                 "thickness information, not both")
            elif n is not None:
                # normalize using the refractive index.
                K = np.sum(Im / eaxis) * self.ddeltaE
                K = K / (np.pi / 2) / (1 - 1. / n ** 2)
                te = (332.5 * K * ke / i0)
            
            Im = Im / K

            # Kramers Kronig Transform:
            # We calculate KKT(Im(-1/epsilon))=1+Re(1/epsilon) with FFT
            # Follows: D W Johnson 1975 J. Phys. A: Math. Gen. 8 490
            # Use an optimal FFT size to speed up the calculation, and
            # make it double the closest upper value to workaround the
            # wrap-around problem.
            esize = next_fast_len(2 * l)  # 2**math.floor(math.log2(l)+1)*4
            q = -2 * np.fft.fft(Im, esize).imag / esize #TODO : min twee?????

            q[:l] *= -1
            q = np.fft.fft(q)
            # Final touch, we have Re(1/eps)
            Re = q[:l].real + 1 #TODO: plus 1???
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

            if iterations > 0 and N_ZLP is not None: #TODO: loop weghalen.
                # Surface losses correction:
                #  Calculates the surface ELF from a vaccumm border effect
                #  A simulated surface plasmon is subtracted from the ELF
                Srfelf = 4 * e2 / ((e1 + 1) ** 2 + e2 ** 2) - Im
                adep = (tgt / (eaxis + delta) *
                        np.arctan(beta * tgt / eaxis) -
                        beta / 1000. /
                        (beta ** 2 + eaxis ** 2. / tgt ** 2))
                Srfint = 2000 * K * adep * Srfelf / rk0 / te * self.ddeltaE  # axis.scale
                if correct_S_s == True:
                    print("correcting S_s")
                    Srfint[Srfint < 0] = 0
                    Srfint[Srfint > S_E] = S_E[Srfint > S_E]
                y = S_E - Srfint
                _logger.debug('Iteration number: %d / %d', io + 1, iterations)

        eps = (e1 + e2 * 1j)
        del y
        del I_EELS
        if 'thickness' in output:
            # As above,prevent errors if the signal is a single spectrum
            output['thickness'] = te

        return eps, te, Srfint
    
    def KK_pixel(self, i, j, signal = 'EELS', select_ZLPs = True, **kwargs):
        """
        
        Option to include pooling, not for thickness, as this is an integral and therefor \
        more noise robust by default.

        Parameters
        ----------
        i : TYPE
            DESCRIPTION.
        j : TYPE
            DESCRIPTION.
        pooled : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dielectric_functions : TYPE
            DESCRIPTION.
        ts : TYPE
            DESCRIPTION.
        S_ss : TYPE
            DESCRIPTION.
        IEELSs : TYPE
            DESCRIPTION.

        """
        #data_ij = self.get_pixel_signal(i,j)#[self.deltaE>0]
        ZLPs = self.calc_ZLPs(i,j, select_ZLPs = select_ZLPs)#[:,self.deltaE>0]
        
        dielectric_functions = (1+1j)* np.zeros(ZLPs[:,self.deltaE>0].shape)
        S_ss = np.zeros(ZLPs[:,self.deltaE>0].shape)
        ts = np.zeros(ZLPs.shape[0])            
        IEELSs = np.zeros(ZLPs.shape)
        max_ieels = np.zeros(ZLPs.shape[0]) 
        n = self.n[self.clustered[i,j]]
        for k in range(ZLPs.shape[0]):
            ZLP_k = ZLPs[k,:]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k)
            IEELSs[k,:] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            if signal in self.EELS_NAMES:
                dielectric_functions[k,:], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP = N_ZLP, n = n, **kwargs)
            else:
                ts[k] = self.calc_thickness(IEELS, n, N_ZLP)
        if signal  in self.EELS_NAMES:
            return dielectric_functions, ts, S_ss, IEELSs, max_ieels
        
        IEELSs_OG = IEELSs
        ts_OG = ts
        max_OG = max_ieels
        
        ZLPs_signal = self.calc_ZLPs(i,j, signal = signal, select_ZLPs=select_ZLPs)
        dielectric_functions = (1+1j)* np.zeros(ZLPs_signal[:,self.deltaE>0].shape)
        S_ss = np.zeros(ZLPs_signal[:,self.deltaE>0].shape)
        ts = np.zeros(ZLPs_signal.shape[0])            
        IEELSs = np.zeros(ZLPs_signal.shape)
        max_ieels = np.zeros(ZLPs_signal.shape[0]) 
        
        for k in range(ZLPs_signal.shape[0]):
            ZLP_k = ZLPs_signal[k,:]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k, signal = signal)
            IEELSs[k] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            dielectric_functions[k,:], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP = N_ZLP, n = n, **kwargs)
        
        return [ts_OG, IEELSs_OG, max_OG], [dielectric_functions, ts, S_ss, IEELSs, max_ieels]
    
    
    def optical_absorption_coeff(self, dielectric_function):
        
        #TODO: now assuming one input for dielectric function. We could check for dimentions, and do everything at once??
        
        eps1 = np.real(dielectric_function)
        E = self.deltaE[self.deltaE>0]
        
        mu = E/(self.h_bar*self.c) * np.power(2 * np.absolute(dielectric_function) - 2*eps1, 0.5)
        
        return mu
        
        pass
    
    
    def im_dielectric_function_bs(self, track_process=False, plot=False, save_index=None, save_path="KK_analysis",
                                  smooth=False):
        """

        Parameters
        ----------
        track_process: bool
            default = False, if True: prints for each pixel that program is busy with that pixel.
        plot: bool
            default = False, if True, plots all calculated dielectric functions
        save_index: opt
        save_path: opt
        smooth: opt

        """
        # TODO
        # data = self.data[self.deltaE>0, :,:]
        # energies = self.deltaE[self.deltaE>0]
        # TODO: make check for models
        # if not hasattr(self, 'ZLPs_gen'):
        #     self.calc_ZLPs_gen2("iets")
        self.dielectric_function_im_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.dielectric_function_im_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.thickness_avg = np.zeros(self.image_shape)
        self.thickness_std = np.zeros(self.image_shape)
        self.IEELS_avg = np.zeros(self.data.shape)
        self.IEELS_std = np.zeros(self.data.shape)
        N_ZLPs_calculated = hasattr(self, 'N_ZLPs')
        # TODO: add N_ZLP saving
        # if not N_ZLPs_calculated:
        #    self.N_ZLPs = np.zeros(self.image_shape)
        if plot:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel ", i, j)
                """
                data_ij = self.get_pixel_signal(i,j)#[self.deltaE>0]
                ZLPs = self.calc_ZLPs(i,j)#[:,self.deltaE>0]
                dielectric_functions = (1+1j)* np.zeros(ZLPs[:,self.deltaE>0].shape)
                S_ss = np.zeros(ZLPs[:,self.deltaE>0].shape)
                ts = np.zeros(ZLPs.shape[0])            
                IEELSs = np.zeros(ZLPs.shape)
                for k in range(ZLPs.shape[0]):
                    ZLP_k = ZLPs[k,:]
                    N_ZLP = np.sum(ZLP_k)
                    IEELS = data_ij-ZLP_k
                    if smooth:
                        IEELS = smooth_1D(IEELS)
                    IEELS = self.deconvolute(i, j, ZLP_k)
                    IEELSs[k,:] = IEELS
                    if plot: 
                        #ax1.plot(self.deltaE, IEELS)
                        plt.figure()
                        plt.plot(self.deltaE, IEELS)
                    #TODO: FIX ZLP: now becomes very negative!!!!!!!
                    #TODO: VERY IMPORTANT
                    dielectric_functions[k,:], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP = N_ZLP, n =3)
                    if plot: 
                        #plt.figure()
                        plt.plot(self.deltaE[self.deltaE>0], dielectric_functions[k,:]*2)
                        plt.xlim(0,10)
                        plt.ylim(-100, 400)
                    """
                dielectric_functions, ts, S_ss, IEELSs = self.KK_pixel(i, j)
                # print(ts)
                self.dielectric_function_im_avg[i, j, :] = np.average(dielectric_functions, axis=0)
                self.dielectric_function_im_std[i, j, :] = np.std(dielectric_functions, axis=0)
                self.S_s_avg[i, j, :] = np.average(S_ss, axis=0)
                self.S_s_std[i, j, :] = np.std(S_ss, axis=0)
                self.thickness_avg[i, j] = np.average(ts)
                self.thickness_std[i, j] = np.std(ts)
                self.IEELS_avg[i, j, :] = np.average(IEELSs, axis=0)
                self.IEELS_std[i, j, :] = np.std(IEELSs, axis=0)
        if save_index is not None:
            save_path += (not save_path[0] == '/') * '/'
            with open(save_path + "diel_fun_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.dielectric_function_im_avg)
            with open(save_path + "S_s_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.S_s_avg)
            with open(save_path + "thickness_" + str(save_index) + ".npy", 'wb') as f:
                np.save(f, self.thickness_avg)
        # return dielectric_function_im_avg, dielectric_function_im_std

    def im_dielectric_function(self, track_process=False, plot=False, save_index=None):
        """
        INPUT:
            self -- the image of which the dielectic functions are calculated
            track_process -- boolean, default = False, if True: prints for each pixel that program is busy with that pixel.
            plot -- boolean, default = False, if True, plots all calculated dielectric functions
        OUTPUT ATRIBUTES:
            self.dielectric_function_im_avg = average dielectric function for each pixel
            self.dielectric_function_im_std = standard deviation of the dielectric function at each energy for each pixel
            self.S_s_avg = average surface scattering distribution for each pixel
            self.S_s_std = standard deviation of the surface scattering distribution at each energy for each pixel
            self.thickness_avg = average thickness for each pixel
            self.thickness_std = standard deviation thickness for each pixel
            self.IEELS_avg = average bulk scattering distribution for each pixel
            self.IEELS_std = standard deviation of the bulk scattering distribution at each energy for each pixel
        """
        # TODO
        # data = self.data[self.deltaE>0, :,:]
        # energies = self.deltaE[self.deltaE>0]
        if not hasattr(self, 'ZLPs_gen'):
            self.calc_ZLPs_gen2("iets")
        self.dielectric_function_im_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.dielectric_function_im_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.thickness_avg = np.zeros(self.image_shape)
        self.thickness_std = np.zeros(self.image_shape)
        self.IEELS_avg = np.zeros(self.data.shape)
        self.IEELS_std = np.zeros(self.data.shape)
        N_ZLPs_calculated = hasattr(self, 'N_ZLPs')
        # TODO: add N_ZLP saving
        # if not N_ZLPs_calculated:
        #    self.N_ZLPs = np.zeros(self.image_shape)
        if plot:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel ", i, j)
                data_ij = self.get_pixel_signal(i, j)  # [self.deltaE>0]
                ZLPs = self.calc_ZLPs(i, j)  # [:,self.deltaE>0]
                dielectric_functions = (1 + 1j) * np.zeros(ZLPs[:, self.deltaE > 0].shape)
                S_ss = np.zeros(ZLPs[:, self.deltaE > 0].shape)
                ts = np.zeros(ZLPs.shape[0])
                IEELSs = np.zeros(ZLPs.shape)
                for k in range(23, 28):  # ZLPs.shape[0]):
                    ZLP_k = ZLPs[k, :]
                    N_ZLP = np.sum(ZLP_k)
                    IEELS = data_ij - ZLP_k
                    IEELS = self.deconvolute(i, j, ZLP_k)
                    IEELSs[k, :] = IEELS
                    if plot:
                        # ax1.plot(self.deltaE, IEELS)
                        plt.figure()
                        plt.plot(self.deltaE, IEELS)
                    # TODO: FIX ZLP: now becomes very negative!!!!!!!
                    # TODO: VERY IMPORTANT
                    dielectric_functions[k, :], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP=N_ZLP, n=3)
                    if plot:
                        # plt.figure()
                        plt.plot(self.deltaE[self.deltaE > 0], dielectric_functions[k, :] * 2)
                        plt.xlim(0, 10)
                        plt.ylim(-100, 400)

                # print(ts)
                self.dielectric_function_im_avg[i, j, :] = np.average(dielectric_functions, axis=0)
                self.dielectric_function_im_std[i, j, :] = np.std(dielectric_functions, axis=0)
                self.S_s_avg[i, j, :] = np.average(S_ss, axis=0)
                self.S_s_std[i, j, :] = np.std(S_ss, axis=0)
                self.thickness_avg[i, j] = np.average(ts)
                self.thickness_std[i, j] = np.std(ts)
                self.IEELS_avg[i, j, :] = np.average(IEELSs, axis=0)
                self.IEELS_std[i, j, :] = np.std(IEELSs, axis=0)
        # return dielectric_function_im_avg, dielectric_function_im_std
    
    def optical_absorption_coeff_im(self):
        #TODO!!
        pass
    
    def crossings_im(self):  # ,  delta = 50):
        """
        INPUT: 
            self
        OUTPUT:
            self.crossings_E = numpy array (image-shape, N_c), where N_c the maximimun number of crossings of any pixel, 0 indicates no crossing
            self.crossings_n = numpy array (image-shape), number of crossings per pixel
        Calculates for each pixel the crossings of the real part of the dielectric function \
            from negative to positive.
        """
        self.crossings_E = np.zeros((self.image_shape[0], self.image_shape[1], 1))
        self.crossings_n = np.zeros(self.image_shape)
        n_max = 1
        for i in range(self.image_shape[0]):
            # print("cross", i)
            for j in range(self.image_shape[1]):
                # print("cross", i, j)
                crossings_E_ij, n = self.crossings(i, j)  # , delta)
                if n > n_max:
                    # print("cross", i, j, n, n_max, crossings_E.shape)
                    crossings_E_new = np.zeros((self.image_shape[0], self.image_shape[1], n))
                    # print("cross", i, j, n, n_max, crossings_E.shape, crossings_E_new[:,:,:n_max].shape)
                    crossings_E_new[:, :, :n_max] = self.crossings_E
                    self.crossings_E = crossings_E_new
                    n_max = n
                    del crossings_E_new
                self.crossings_E[i, j, :n] = crossings_E_ij
                self.crossings_n[i, j] = n

    def crossings(self, i, j):  # , delta = 50):
        # l = len(die_fun)
        die_fun_avg = np.real(self.dielectric_function_im_avg[i, j, :])
        # die_fun_f = np.zeros(l-2*delta)
        # TODO: use smooth?
        """
        for i in range(self.l-delta):
            die_fun_avg[i] = np.average(self.dielectric_function_im_avg[i:i+delta])
        """
        crossing = np.concatenate((np.array([0]), (die_fun_avg[:-1] < 0) * (die_fun_avg[1:] >= 0)))
        deltaE_n = self.deltaE[self.deltaE > 0]
        # deltaE_n = deltaE_n[50:-50]
        crossing_E = deltaE_n[crossing.astype('bool')]
        n = len(crossing_E)
        return crossing_E, n


    # %%
    # TODO: add bandgap finding

    def cluster(self, n_clusters=5, based_upon="log", **kwargs):
        """
        Clusters image into n_clusters clusters, based upon (log) integrated intensity of each \
            pixel or thicknes. Saves cluster centra in self.clusters, and the index of to which \
            each cluster belongs in self.clustered.

        Parameters
        ----------
        n_clusters : TYPE, optional
            DESCRIPTION. The default is 5.
        based_upon : TYPE, optional
            DESCRIPTION. The default is "log".
        **kwargs : keyword arguments for k_means function
            options: n_iterations (int, default 30), n_times (int, default 5)

        Returns
        -------
        None.

        """
        # TODO: add other based_upons
        if based_upon == "sum":
            values = np.sum(self.data, axis=2).flatten()
        elif based_upon == "log":
            values = np.log(np.sum(np.maximum(self.data,1e-14), axis=2).flatten())
        elif based_upon == "thickness":
            values = self.t[:,:,0].flatten()
        elif type(based_upon) == np.ndarray:
            values = based_upon.flatten()
            if values.size != (self.image_shape[0]*self.image_shape[1]):
                raise IndexError("The size of values on which to cluster does not match the image size.")
        else:
            values = np.sum(self.data, axis=2).flatten()
        clusters_unsorted, r = k_means(values, n_clusters=n_clusters, **kwargs)
        self.clusters = np.sort(clusters_unsorted)[::-1]
        arg_sort_clusters = np.argsort(clusters_unsorted)[::-1]
        self.clustered = np.zeros(self.image_shape)
        for i in range(n_clusters):
            in_cluster_i = r[arg_sort_clusters[i]]
            self.clustered += ((np.reshape(in_cluster_i, self.image_shape)) * i)
        self.clustered = self.clustered.astype(int)

    def cluster_on_cluster_values(self, cluster_values):
        """ If the image has been clustered before, and the values of the cluster centra are known, \
            one can use this function to reconstruct the original clustering of the image. At this \
            time works for images clustered on (log) integrated intensity."""
        self.clusters = cluster_values

        values = np.sum(self.data, axis=2)
        check_log = (np.nanpercentile(values, 5) > cluster_values.max())
        if check_log:
            values = np.log(values)
        valar = (values.transpose() * np.ones(np.append(self.image_shape, self.n_clusters)).transpose()).transpose()
        self.clustered = np.argmin(np.absolute(valar - cluster_values), axis=2)
        if len(np.unique(self.clustered)) < self.n_clusters:
            warnings.warn(
                "it seems like the clustered values of dE1 are not clustered on this image/on log or sum. Please check clustering.")

    # PLOTTING FUNCTIONS
    def plot_sum(self, title=None, xlab=None, ylab=None):
        """
        INPUT:
            self -- spectral image 
            title -- str, delfault = None, title of plot
            xlab -- str, default = None, x-label
            ylab -- str, default = None, y-label
        OUTPUT:
        Plots the summation over the intensity for each pixel in a heatmap.
        """
        # TODO: invert colours
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = ''
        plt.figure()
        if title is None:
            plt.title("intgrated intensity spectrum " + name)
        else:
            plt.title(title)
        if hasattr(self, 'pixelsize'):
            #    plt.xlabel(self.pixelsize)
            #    plt.ylabel(self.pixelsize)
            plt.xlabel("[m]")
            plt.ylabel("[m]")
            xticks, yticks = self.get_ticks()
            ax = sns.heatmap(np.sum(self.data, axis=2), xticklabels=xticks, yticklabels=yticks)
        else:
            ax = sns.heatmap(np.sum(self.data, axis=2))
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel(ylab)
        plt.show()

    def plot_heatmap(self, data, title=None, xlab=None, ylab=None, cmap='coolwarm', discrete_colormap=False, sig=2, n_xticks=10, n_yticks=10,
                     save_as=False, color_bin_size=None, equal_axis=True, **kwargs):
        """
        INPUT:
            self -- spectral image 
            title -- str, delfault = None, title of plot
            xlab -- str, default = None, x-label
            ylab -- str, default = None, y-label
        OUTPUT:
        Plots the summation over the intensity for each pixel in a heatmap.
        """
        # TODO: invert colours
        
        plt.figure(dpi=200)
        if title is None:
            if hasattr(self, 'name'):
                plt.title(self.name)
        else:
            plt.title(title)
        if 'mask' in kwargs:
            mask = kwargs['mask']
            if mask.all():
                #raise ValueError("Mask all True: no values to plot.")
                warnings.warn("Mask all True: no values to plot.")
                return
        else:
            mask = np.zeros(data.shape).astype('bool')
        dis_max = None
        dis_min = None
        if discrete_colormap:

            unique_data_points = np.unique(data[~mask])

            if 'vmax' in kwargs:
                dis_max = kwargs['vmax']
                if len(unique_data_points[unique_data_points > kwargs['vmax']]) > 0:
                    unique_data_points = unique_data_points[unique_data_points <= kwargs['vmax']]
                    unique_data_points = np.append(unique_data_points, kwargs['vmax'])
            if 'vmin' in kwargs:
                dis_min = kwargs['vmin']
                if len(unique_data_points[unique_data_points < kwargs['vmin']]) > 0:
                    unique_data_points = unique_data_points[unique_data_points >= kwargs['vmin']]
                    unique_data_points = np.append(kwargs['vmin'], unique_data_points)

            if color_bin_size is None:
                if len(unique_data_points) == 1:
                    color_bin_size = 1
                else:
                    color_bin_size = np.nanpercentile(unique_data_points[1:]-unique_data_points[:-1],30)
            n_colors = round((np.max(unique_data_points) - np.min(unique_data_points))/color_bin_size +1)
            cmap = cm.get_cmap(cmap, n_colors)
            spacing = color_bin_size / 2

            if not 'vmax' in kwargs:
                kwargs['vmax'] = np.max(data[~mask]) + spacing
                # print("vmax", kwargs['vmax'])
            if not 'vmin' in kwargs:
                kwargs['vmin'] = np.min(data[~mask]) - spacing

                # print("spacing", spacing, "vmax", kwargs['vmax'], "vmin", kwargs['vmin'])
        if equal_axis:
            plt.axis('scaled')
        if hasattr(self, 'pixelsize'):
            plt.xlabel("[m]")
            plt.ylabel("[m]")
            xticks, yticks = self.get_ticks(sig=2, n_xticks=n_xticks, n_yticks=n_yticks)#sig)
            ax = sns.heatmap(data, cmap=cmap, **kwargs)
            ax.xaxis.set_ticks(np.arange(0, len(data[0,:]), math.floor(len(data[0,:]) / n_xticks)))
            ax.yaxis.set_ticks(np.arange(0, len(data[:,0]), math.floor(len(data[:,0]) / n_yticks)))
            ax.set_xticklabels(xticks, rotation=0)
            ax.set_yticklabels(yticks)
        else:
            ax = sns.heatmap(data, **kwargs)
        if xlab is not None:
            plt.xlabel(xlab)
        else:
            plt.xlabel('[\u03BCm]')
        if ylab is not None:
            plt.ylabel(ylab)
        else:
            plt.ylabel('[\u03BCm]')
        
        colorbar = ax.collections[0].colorbar
        if discrete_colormap:
            # TODO: even space in colorbar
            if data.dtype == int:
                colorbar.set_ticks(np.unique(data[~mask]))
            else:
                colorbar.set_ticks(np.unique(data[~mask]))#cbar_ticks)
                cbar_ticks_labels = []
                for tick in np.unique(data[~mask]):
                    # fmt = '%.' + str(sig) + 'g'
                    # cbar_ticks.append('%s' % float(fmt % tick))
                    cbar_ticks_labels.append(round_scientific(tick, sig))
                colorbar.ax.set_yticklabels(cbar_ticks_labels)
                # colorbar.set_ticks(cbar_ticks)
        if (('vmin' in kwargs) and not discrete_colormap) or (dis_min is not None):
            if np.nanmin(data[~mask]) < kwargs['vmin']:
                cbar_ticks = colorbar.ax.get_yticklabels()
                loc = 0
                if discrete_colormap:
                    #TODO: add hele groter dan plus bin gedoe net als bij vmax
                    loc = np.min(np.argwhere(colorbar.ax.get_yticks() >= dis_min))
                cbar_ticks[loc] = r'$\leq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)
        if (('vmax' in kwargs) and not discrete_colormap) or (dis_max is not None):
            if np.nanmax(data[~mask]) > kwargs['vmax']:
                cbar_ticks = colorbar.ax.get_yticklabels()
                cbar_ticks_values = colorbar.ax.get_yticks()
                loc = -1
                if discrete_colormap:
                    loc = np.max(np.argwhere(cbar_ticks_values <= dis_max + color_bin_size/2))
                    if cbar_ticks_values[loc] <= cbar_ticks_values[-1] - color_bin_size:
                        tick = colorbar.ax.get_yticks()[loc] + color_bin_size
                        cbar_ticks_values[loc+1] = tick
                        colorbar.ax.set_yticks(cbar_ticks)
                        cbar_ticks_labels[loc+1] = round_scientific(tick, sig)
                        colorbar.ax.set_yticklabels(cbar_ticks_labels)
                        loc += 1
                cbar_ticks[loc] = r'$\geq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)
        plt.show()
        if save_as:
            if type(save_as) != str:
                if hasattr(self, 'name'):
                    save_as = self.name
            if 'mask' in kwargs:
                save_as += '_masked'
            save_as += '.pdf'
            plt.savefig(save_as)

    def get_ticks(self, sig=2, n_xticks=10, n_yticks=10):
        """
        Generates ticks of (spatial) x- and y axis for plotting purposes.

        Parameters
        ----------
        sig : int, optional
            Scientific signifance of ticsk. The default is 2.
        n_tick : int, optional
            desired number of ticks. The default is 10.

        Returns
        -------
        xlabels : np.array of type object
        ylabels : np.array of type object

        """
        fmt = '%.' + str(sig) + 'g'
        
        each_n_pixels = math.floor(self.x_axis.shape[0] / n_xticks)
        xlabels = np.zeros(len(np.arange(0, self.x_axis.shape[0], each_n_pixels)), dtype=object)
        xlabels[:] = ""
        n=0
        for i in range(self.x_axis.shape[0]):
            if i % each_n_pixels == 0:
                xlabels[n] = '%s' % float(fmt % self.x_axis[i])
                n+=1
        
        each_n_pixels = math.floor(self.y_axis.shape[0] / n_yticks)
        ylabels = np.zeros(len(np.arange(0, self.y_axis.shape[0], each_n_pixels)), dtype=object)
        ylabels[:] = ""
        n=0
        for i in range(self.y_axis.shape[0]):
            if i % each_n_pixels == 0:
                ylabels[n] = '%s' % float(fmt % self.y_axis[i])
                n+=1
        return xlabels, ylabels

    def plot_all(self, same_image=True, normalize=False, legend=False,
                 range_x=None, range_y=None, range_E=None, signal="EELS", log=False):
        # TODO: add titles and such
        if range_x is None:
            range_x = [0, self.image_shape[1]]
        if range_y is None:
            range_y = [0, self.image_shape[0]]
        if same_image:
            plt.figure()
            plt.title("Spectrum image " + signal + " spectra")
            plt.xlabel("[eV]")
            if range_E is not None:
                plt.xlim(range_E)
        for i in range(range_y[0], range_y[1]):
            for j in range(range_x[0], range_x[1]):
                if not same_image:
                    plt.figure()
                    plt.title("Spectrum pixel: [" + str(j) + "," + str(i) + "]")
                    plt.xlabel("[eV]")
                    if range_E is not None:
                        plt.xlim(range_E)
                    if legend:
                        plt.legend()
                signal_pixel = self.get_pixel_signal(i, j, signal)
                if normalize:
                    signal_pixel /= np.max(np.absolute(signal_pixel))
                if log:
                    signal_pixel = np.log(signal_pixel)
                    plt.ylabel("log intensity")
                plt.plot(self.deltaE, signal_pixel, label="[" + str(j) + "," + str(i) + "]")
            if legend:
                plt.legend()

    # GENERAL FUNCTIONS
    def get_key(self, key):
        if key.lower() in (string.lower() for string in self.EELS_NAMES):
            return 'data'
        elif key.lower() in (string.lower() for string in self.IEELS_NAMES):
            return 'ieels'
        elif key.lower() in (string.lower() for string in self.ZLP_NAMES):
            return 'zlp'
        elif key.lower() in (string.lower() for string in self.DIELECTRIC_FUNCTION_NAMES):
            return 'eps'
        elif key.lower() in (string.lower() for string in self.THICKNESS_NAMES):
            return 'thickness'
        else:
            return key

    # STATIC METHODS
    @staticmethod
    def get_prefix(unit, SIunit=None, numeric=True):
        """
        INPUT:
            unit -- str, unit of which the prefix is wanted
            SIunit -- str, default = None, the SI unit of the unit of which the prefix is wanted \
                        (eg 'eV' for 'keV'), if None, first character of unit is evaluated as prefix
            numeric -- bool, default = True, if numeric the prefix is translated to the numeric value \
                        (e.g. 1E3 for 'k')
        OUTPUT:
            prefix -- str or int, the character of the prefix or the numeric value of the prefix
        """
        if SIunit is not None:
            lenSI = len(SIunit)
            if unit[-lenSI:] == SIunit:
                prefix = unit[:-lenSI]
                if len(prefix) == 0:
                    if numeric:
                        return 1
                    else:
                        return prefix
            else:
                print("provided unit not same as target unit: " + unit + ", and " + SIunit)
                if numeric:
                    return 1
                else:
                    #TODO:
                    return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix

        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix in ['μ', 'µ' ,'u', 'micron']:
            return 1E-6
        if prefix == 'm':
            return 1E-3
        if prefix == 'k':
            return 1E3
        if prefix == 'M':
            return 1E6
        if prefix == 'G':
            return 1E9
        if prefix == 'T':
            return 1E12
        else:
            print("either no or unknown prefix in unit: " + unit + ", found prefix " + prefix + ", asuming no.")
        return 1

    @staticmethod
    def calc_avg_ci(np_array, axis=0, ci=16, return_low_high=True):
        avg = np.average(np_array, axis=axis)
        ci_low = np.nanpercentile(np_array, ci, axis=axis)
        ci_high = np.nanpercentile(np_array, 100 - ci, axis=axis)
        if return_low_high:
            return avg, ci_low, ci_high
        return avg, ci_high - ci_low

    # CLASS THINGIES
    def __getitem__(self, key):
        """ Determines behavior of `self[key]` """
        return self.data[key]
        # pass

    def __getattr__(self, key):
        key = self.get_key(key)
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        key = self.get_key(key)
        self.__dict__[key] = value

    def __str__(self):
        if hasattr(self, 'name'):
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return 'Spectral image: ' + name_str + ", image size:" + str(self.data.shape[0]) + 'x' + \
               str(self.data.shape[1]) + ', deltaE range: [' + str(round(self.deltaE[0], 3)) + ',' + \
               str(round(self.deltaE[-1], 3)) + '], deltadeltaE: ' + str(round(self.ddeltaE, 3))

    def __repr__(self):
        data_str = "data * np.ones(" + str(self.shape) + ")"
        if hasattr(self, 'name'):
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return "Spectral_image(" + data_str + ", deltadeltaE=" + str(round(self.ddeltaE, 3)) + name_str + ")"

    def __len__(self):
        return self.l


# GENERAL DATA MODIFICATION FUNCTIONS  

def CFT(x, y):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    N = len(x)
    x_max = np.max(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(2j * np.pi * N_0 * k / N) * delta_x  # np.exp(-1j*(x_0)*k*delta_omg)*delta_x
    F_k = cont_factor * np.fft.fft(y)
    return F_k


def iCFT(x, Y_k):
    x_0 = np.min(x)
    N_0 = np.argmin(np.absolute(x))
    x_max = np.max(x)
    N = len(x)
    delta_x = (x_max - x_0) / N
    k = np.linspace(0, N - 1, N)
    cont_factor = np.exp(-2j * np.pi * N_0 * k / N)
    f_n = np.fft.ifft(cont_factor * Y_k) / delta_x  
    return f_n


def smooth_1D(data, window_len=50, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    """
    # TODO: add comnparison
    window_len += (window_len + 1) % 2
    s = np.r_['-1', data[window_len - 1:0:-1], data, data[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    # y=np.convolve(w/w.sum(),s,mode='valid')
    # return y[(window_len-1):-(window_len)]
    surplus_data = int((window_len - 1) * 0.5)
    data = np.apply_along_axis(lambda m: np.convolve(m, w / w.sum(), mode='valid'), axis=0, arr=s)[
           surplus_data:-surplus_data]
    return data


# MODELING CLASSES AND FUNCTIONS
def bandgap(x, amp, BG, b):
    return amp * (x - BG) ** (b)


class MLP(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        # Initialize the modules we need to build the network
        self.linear1 = nn.Linear(num_inputs, 10)
        self.linear2 = nn.Linear(10, 15)
        self.linear3 = nn.Linear(15, 5)
        self.output = nn.Linear(5, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.output(x)
        return x


def scale(inp, ab):
    """
    min_inp = inp.min()
    max_inp = inp.max()
    
    outp = inp/(max_inp-min_inp) * (max_out-min_out)
    outp -= outp.min()
    outp += min_out
    
    return outp
    """

    return inp * ab[0] + ab[1]
    # pass


def find_scale_var(inp, min_out=0.1, max_out=0.9):
    a = (max_out - min_out) / (inp.max() - inp.min())
    b = min_out - a * inp.min()
    return [a, b]


def round_scientific(value, n_sig):
    if value == 0:
        return 0
    scale = int(math.floor(math.log10(abs(value))))
    num = round(value, n_sig - scale - 1)
    return num
