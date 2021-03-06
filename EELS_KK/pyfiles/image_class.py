#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 17:38:57 2020

@author: isabel
"""
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
import seaborn as sns
import natsort
import numpy as np
import math
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm
import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


from k_means_clustering import k_means
from train_nn_torch import train_nn_scaled


_logger = logging.getLogger(__name__)


class Spectral_image():
    DIELECTRIC_FUNCTION_NAMES = ['dielectric_function', 'dielectricfunction', 'dielec_func', 'die_fun', 'df', 'epsilon']
    EELS_NAMES = ['electron_energy_loss_spectrum','electron_energy_loss','EELS', 'EEL', 'energy_loss', 'data']
    IEELS_NAMES = ['inelastic_scattering_energy_loss_spectrum', 'inelastic_scattering_energy_loss', 'inelastic_scattering', 'IEELS', 'IES']
    ZLP_NAMES = ['zeros_loss_peak', 'zero_loss', 'ZLP', 'ZLPs']
    
    m_0 = 511.06 #eV, electron rest mass
    a_0 = 5.29E-11 #m, Bohr radius
    h_bar = 6.582119569E-16 #eV/s
    c = 2.99792458E8 #m/s
    
    
    def __init__(self, data, deltadeltaE, pixelsize = None, beam_energy = None, collection_angle = None, name = None):
        """
        INPUT:
            data = 3D-numpy array (x-axis x y-axis x energy loss-axis), spectral image data
            deltadeltaE = float, width of energy loss bins
        Keyword-arguments:
            pixelsize = float (default: None), width of pixels
            beam_energy = float (default: None), energy of electron beam [eV]
            collection_angle = float (default: None), collection angle of STEM [rad]
            name = str (default: None), name if given along is used in title of plots
        """
        
        self.data = data
        self.ddeltaE = deltadeltaE
        self.determine_deltaE()
        if pixelsize is not None:
            self.pixelsize = pixelsize
        self.calc_axes()
        if beam_energy is not None:
            self.beam_energy = beam_energy
        if collection_angle is not None:
            self.collection_angle = collection_angle
        if name is not None:
            self.name = name
    
    
    #%%GENERAL FUNCTIONS
    #TODO!!!
    @property
    def x(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self._x

    @x.setter
    def x(self, value):
        print("setter of x called")
        self._x = value

    @x.deleter
    def x(self):
        print("deleter of x called")
        del self._x
    
    #%%PROPERTIES
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
    def load_data(cls, path_to_dmfile):
        """
        INPUT: 
            path_to_dmfile: str, path to spectral image file (.dm3 or .dm4 extension)
        OUTPUT:
            image -- Spectral_image, object of Spectral_image class containing the data of the dm-file
        """
        dmfile_tot = dm.fileDM(path_to_dmfile)
        for i in range(dmfile_tot.numObjects - dmfile_tot.thumbnail*1):
            dmfile = dmfile_tot.getDataset(i)
            if dmfile['data'].ndim == 3:
                dmfile = dmfile_tot.getDataset(i)
                data = np.swapaxes(np.swapaxes(dmfile['data'], 0,1), 1,2)
                break
            elif i == dmfile_tot.numObjects - dmfile_tot.thumbnail*1 - 1:
                print("No spectral image detected")
                dmfile = dmfile_tot.getDataset(0)
                data = dmfile['data']
        
        #.getDataset(0)
        ddeltaE = dmfile['pixelSize'][0]
        pixelsize = np.array(dmfile['pixelSize'][1:])
        energyUnit = dmfile['pixelUnit'][0]
        ddeltaE *= cls.get_prefix(energyUnit, 'eV')
        pixelUnit = dmfile['pixelUnit'][1]
        pixelsize *= cls.get_prefix(pixelUnit, 'm')
        image = cls(data, ddeltaE, pixelsize = pixelsize)
        return image
    
    
    
    
    def determine_deltaE(self):
        """
        INPUT: 
            self
        
        Determines the delta energies of the spectral image, based on the delta delta energie,
        and the index on which the spectral image has on average the highest intesity, this 
        is taken as the zero point for the delta energy.
        """
        data_avg = np.average(self.data, axis = (0,1))
        ind_max = np.argmax(data_avg)
        self.deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l-ind_max-1)*self.ddeltaE, self.l)
        #return deltaE
    
    
    def calc_axes(self):
        self.y_axis = np.linspace(0, self.image_shape[0]-1, self.image_shape[0])
        self.x_axis = np.linspace(0, self.image_shape[1]-1, self.image_shape[1])
        if hasattr(self, 'pixelsize'):
            self.y_axis *= self.pixelsize[0]
            self.x_axis *= self.pixelsize[1] 
    
    #%%RETRIEVING FUNCTIONS
    def get_data(self): #TODO: add smooth possibility
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
    
    def get_pixel_signal(self, i,j, signal = 'EELS'):
        """
        INPUT:
            i: int, x-coordinate for the pixel
            j: int, y-coordinate for the pixel
        Keyword argument:
            signal: str (default = 'EELS'), what signal is requested, should comply with defined names
        OUTPUT:
            signal: 1D numpy array, array with the requested signal from the requested pixel
        """
        #TODO: add alternative signals + names
        if signal in self.EELS_NAMES:
            return np.copy(self.data[ i, j, :])
        elif signal in self.DIELECTRIC_FUNCTION_NAMES:
            return np.copy(self.dielectric_function_im_avg[ i, j, :])
        else:
            return np.copy(self.data[ i, j, :])
    
    
    def get_cluster_spectra(self, conf_interval = 0.68, clusters = None, save_as_attribute = False, based_upon = "sum"):
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
        
        if clusters is None:
            clusters = range(self.n_clusters)
        if conf_interval >= 1:
            ci_lim = 0
        
        integrated_I = np.sum(self.data, axis = 2)
        cluster_data = np.zeros(len(clusters), dtype = object)
        
        j=0
        for i in clusters:
            data_cluster = self.data[self.clustered == i]
            intensities_cluster = integrated_I[self.clustered == i]
            arg_sort_I = np.argsort(intensities_cluster)
            if conf_interval < 1:
                ci_lim = round((1-conf_interval)/2 *intensities_cluster.size) #TODO: ask juan: round up or down?
            data_cluster = data_cluster[arg_sort_I][ci_lim:-ci_lim]
            intensities_cluster = np.ones(len(intensities_cluster)-2*ci_lim)*self.clusters[i]
            cluster_data[j] = data_cluster
            j += 1
        
        if save_as_attribute:
            self.cluster_data = cluster_data
        else:
            return cluster_data
    
    
    #%%METHODS ON SIGNAL
    
    def cut(self, E1 = None, E2 = None, in_ex = "in"):
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
        if E1 is None:
            E1 = self.deltaE.min() -1
        if E2 is None:
            E2 = self.deltaE.max() +1
        if in_ex == "in":
            select = ((self.deltaE >= E1) & (self.deltaE <= E2))
        else: 
            select = ((self.deltaE > E1) & (self.deltaE < E2))
        self.data = self.data[:,:,select]
        self.deltaE = self.deltaE[select]
        #TODO add selecting of all attributes
        pass
    
    def cut_image(self, range_width, range_height):
        #TODO: add floats for cutting to meter sizes?
        self.data = self.data[range_height[0]:range_height[1], range_width[0]:range_width[1]]
        self.y_axis = self.y_axis[range_height[0]:range_height[1]]
        self.x_axis = self.x_axis[range_width[0]:range_width[1]]
    
    #TODO
    def samenvoegen(self):
        pass
    
    def smooth(self, window_len=10,window='hanning', keep_original = False):
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
        #TODO: add comnparison
        window_len += (window_len+1)%2
        s=np.r_['-1', self.data[:,:,window_len-1:0:-1],self.data,self.data[:,:,-2:-window_len-1:-1]]
    
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
        
        #y=np.convolve(w/w.sum(),s,mode='valid')
        surplus_data = int((window_len-1)*0.5)
        if keep_original:
            self.data_smooth = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=2, arr=s)[:,:,surplus_data:-surplus_data]
        else:
            self.data = np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=2, arr=s)[:,:,surplus_data:-surplus_data]
        
        
        return #y[(window_len-1):-(window_len)]
    
    
    def deconvolute(self, i,j, ZLP):
        
        y = self.get_pixel_signal(i,j)
        r = 3 #Drude model, can also use estimation from exp. data
        A = y[-1]
        n_times_extra = 2
        sem_inf = next_fast_len(n_times_extra*self.l)
        
        
        
        y_extrp = np.zeros(sem_inf)
        y_ZLP_extrp = np.zeros(sem_inf)
        x_extrp = np.linspace(self.deltaE[0]- self.l*self.ddeltaE, sem_inf*self.ddeltaE+self.deltaE[0]- self.l*self.ddeltaE, sem_inf)
        
        x_extrp = np.linspace(self.deltaE[0], sem_inf*self.ddeltaE+self.deltaE[0], sem_inf)

        y_ZLP_extrp[:self.l] = ZLP        
        y_extrp[:self.l] = y
        x_extrp[:self.l] = self.deltaE[-self.l:]
        
        y_extrp[self.l:] = A*np.power(1+x_extrp[self.l:]-x_extrp[self.l],-r)
        
        x = x_extrp
        y = y_extrp
        y_ZLP = y_ZLP_extrp
        
        z_nu = CFT(x,y_ZLP)
        i_nu = CFT(x,y)
        abs_i_nu = np.absolute(i_nu)
        N_ZLP = 1#scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)
        
        s_nu = N_ZLP*np.log(i_nu/z_nu)
        j1_nu = z_nu*s_nu/N_ZLP
        S_E = np.real(iCFT(x,s_nu))
        s_nu_nc = s_nu
        s_nu_nc[500:-500] = 0
        S_E_nc = np.real(iCFT( x,s_nu_nc))
        J1_E = np.real(iCFT(x,j1_nu))
        
        return J1_E[:self.l]
    
    #%%METHODS ON ZLP
    #CALCULATING ZLPs FROM PRETRAINDED MODELS
  
  
       

    
    def calc_ZLPs(self, i,j, **kwargs):
        ### Definition for the matching procedure
        signal = self.get_pixel_signal(i,j)
        
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
        
        self.dE0 = self.dE1-0.5
        
        #TODO: aanpassen
        def matching( signal, gen_i_ZLP):
            #gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]#*np.max(signal)/np.max(self.ZLPs_gen[ind_ZLP,:]) #TODO!!!!, normalize?
            delta = np.divide((self.dE1 - self.dE0), 3)
            
            factor_NN = np.exp(- np.divide((self.deltaE[(self.deltaE<self.dE1) & (self.deltaE >= self.dE0)] - self.dE1)**2, delta**2))
            factor_dm = 1 - factor_NN
            
            range_0 = signal[self.deltaE < self.dE0]
            range_1 = gen_i_ZLP[(self.deltaE < self.dE1) & (self.deltaE >= self.dE0)] * factor_NN + signal[(self.deltaE < self.dE1) & (self.deltaE >= self.dE0)] * factor_dm
            range_2 = gen_i_ZLP[(self.deltaE >= self.dE1) & (self.deltaE < 3 * self.dE2)]
            range_3 = gen_i_ZLP[(self.deltaE >= 3 * self.dE2)] * 0
            totalfile = np.concatenate((range_0, range_1, range_2, range_3), axis=0)
            #TODO: now hardcoding no negative values!!!! CHECKKKK
            totalfile = np.minimum(totalfile, signal)
            return totalfile
        
        count = len(self.ZLP_models)
        ZLPs = np.zeros((count, self.l)) #np.zeros((count, len_data))
        
        
        if not hasattr(self, "scale_var_deltaE"):
            self.scale_var_deltaE = find_scale_var(self.deltaE)
        
        if not hasattr(self, "scale_var_log_sum_I"):
            all_spectra = self.data
            all_spectra[all_spectra<1] = 1
            int_log_I = np.log(np.sum(all_spectra, axis=2)).flatten()
            self.scale_var_log_sum_I = find_scale_var(int_log_I)
            del all_spectra
        
        log_sum_I_pixel = np.log(np.sum(signal))
        predict_x_np = np.zeros((self.l,2))
        predict_x_np[:,0] = scale(self.deltaE, self.scale_var_deltaE)
        predict_x_np[:,1] = scale(log_sum_I_pixel, self.scale_var_log_sum_I)

        predict_x = torch.from_numpy(predict_x_np)
        
        for k in range(count): 
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions = np.exp(model(predict_x.float()).flatten())
            ZLPs[k,:] = matching(signal, predictions)#matching(energies, np.exp(mean_k), data)
            
        return ZLPs
        
    
    def train_ZLPs(self, n_clusters = None, conf_interval = 0.68, clusters = None, **kwargs):
        if not hasattr(self, "clustered"):
            if n_clusters is not None:
                self.cluster(n_clusters)
            else:
                self.cluster()
        elif n_clusters is not None and self.n_clusters != n_clusters:
            self.cluster(n_clusters)
        
        training_data = self.get_cluster_spectra( conf_interval = conf_interval, clusters = clusters)
        #self.models = 
        train_nn_scaled(self, training_data, **kwargs)
        self.dE1, self.dE2 = train_nn_scaled(self, training_data, **kwargs)

    def load_ZLP_models(self, path_to_models = "models", threshold_costs = 1, name_in_path = True, plotting = False):
        if hasattr(self, "name") and name_in_path:
            path_to_models = self.name + "_" + path_to_models
        
        if not os.path.exists(path_to_models):
            print("No path " + path_to_models + " found. Please ensure spelling and that there are models trained.")
            return
        
        self.ZLP_models = []
        
        model = MLP(num_inputs=2, num_outputs=1)

        files = np.loadtxt(path_to_models + "/costs.txt")
        
        if plotting:
            plt.figure()
            plt.title("chi^2 distribution of models")
            plt.hist(files[files < threshold_costs*3], bins = 20)
            plt.xlabel("chi^2")
            plt.ylabel("number of occurence")
        
        n_working_models = np.sum(files<threshold_costs)
        
        k=0
        for j in range(len(files)):
            if files[j] < threshold_costs:
                with torch.no_grad():
                    model.load_state_dict(torch.load(path_to_models + "/nn_rep" + str(j)))
                    self.ZLP_models.append(copy.deepcopy(model))
                k+=1

    #METHODS ON DIELECTRIC FUNCTIONS
    
    def kramers_kronig_hs(self, I_EELS,
                            N_ZLP=None,
                            iterations=1,
                            n=None,
                            t=None,
                            delta=0.5, correct_S_s = False):
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
    
        eaxis = self.deltaE[self.deltaE>0] #axis.axis.copy()
        S_E = I_EELS[self.deltaE>0]
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
                Srfint = 2000 * K * adep * Srfelf / rk0 / te * self.ddeltaE #axis.scale
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


    
    def im_dielectric_function(self, track_process = False, plot = False):
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
        #TODO
        #data = self.data[self.deltaE>0, :,:]
        #energies = self.deltaE[self.deltaE>0]
        if not hasattr(self, 'ZLPs_gen'):
            self.calc_ZLPs_gen2("iets")
        self.dielectric_function_im_avg = (1+1j)*np.zeros(self.data[ :,:,self.deltaE>0].shape)
        self.dielectric_function_im_std = (1+1j)*np.zeros(self.data[ :,:,self.deltaE>0].shape)
        self.S_s_avg = (1+1j)*np.zeros(self.data[ :,:,self.deltaE>0].shape)
        self.S_s_std = (1+1j)*np.zeros(self.data[ :,:,self.deltaE>0].shape)
        self.thickness_avg = np.zeros(self.image_shape)
        self.thickness_std = np.zeros(self.image_shape)
        self.IEELS_avg = np.zeros(self.data.shape)
        self.IEELS_std = np.zeros(self.data.shape)
        N_ZLPs_calculated = hasattr(self, 'N_ZLPs')
        #TODO: add N_ZLP saving
        #if not N_ZLPs_calculated:
        #    self.N_ZLPs = np.zeros(self.image_shape)
        if plot:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel " , i,j)
                data_ij = self.get_pixel_signal(i,j)#[self.deltaE>0]
                ZLPs = self.calc_ZLPs(i,j)#[:,self.deltaE>0]
                dielectric_functions = (1+1j)* np.zeros(ZLPs[:,self.deltaE>0].shape)
                S_ss = np.zeros(ZLPs[:,self.deltaE>0].shape)
                ts = np.zeros(ZLPs.shape[0])            
                IEELSs = np.zeros(ZLPs.shape)
                for k in range(23,28):#ZLPs.shape[0]):
                    ZLP_k = ZLPs[k,:]
                    N_ZLP = np.sum(ZLP_k)
                    IEELS = data_ij-ZLP_k
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
                
                #print(ts)
                self.dielectric_function_im_avg[i,j,:] = np.average(dielectric_functions, axis = 0)
                self.dielectric_function_im_std[i,j,:] = np.std(dielectric_functions, axis = 0)
                self.S_s_avg[i,j,:] = np.average(S_ss, axis = 0)
                self.S_s_std[i,j,:] = np.std(S_ss, axis = 0)
                self.thickness_avg[i,j] = np.average(ts)
                self.thickness_std[i,j] = np.std(ts)
                self.IEELS_avg[i,j,:] = np.average(IEELSs, axis = 0)
                self.IEELS_std[i,j,:] = np.std(IEELSs, axis = 0)
        #return dielectric_function_im_avg, dielectric_function_im_std
    
    def crossings_im(self):#,  delta = 50):
        """
        INPUT: 
            self
        OUTPUT:
            self.crossings_E = numpy array (image-shape, N_c), where N_c the maximimun number of crossings of any pixel, 0 indicates no crossing
            self.crossings_n = numpy array (image-shape), number of crossings per pixel
        Calculates for each pixel the crossings of the real part of the dielectric function \
            from negative to positive.
        """
        self.crossings_E = np.zeros((self.image_shape[0], self.image_shape[1],1))
        self.crossings_n = np.zeros(self.image_shape)
        n_max = 1
        for i in range(self.image_shape[0]):
            #print("cross", i)
            for j in range(self.image_shape[1]): 
                #print("cross", i, j)
                crossings_E_ij, n = self.crossings(i,j)#, delta)
                if n > n_max:
                    #print("cross", i, j, n, n_max, crossings_E.shape)
                    crossings_E_new = np.zeros((self.image_shape[0], self.image_shape[1],n))
                    #print("cross", i, j, n, n_max, crossings_E.shape, crossings_E_new[:,:,:n_max].shape)
                    crossings_E_new[:,:,:n_max] = self.crossings_E
                    self.crossings_E = crossings_E_new
                    n_max = n
                    del crossings_E_new
                self.crossings_E[i,j,:n] = crossings_E_ij
                self.crossings_n[i,j] = n
    
    def crossings(self, i, j):#, delta = 50):
        #l = len(die_fun)
        die_fun_avg = np.real(self.dielectric_function_im_avg[ i, j, :])
        #die_fun_f = np.zeros(l-2*delta)
        #TODO: use smooth?
        """
        for i in range(self.l-delta):
            die_fun_avg[i] = np.average(self.dielectric_function_im_avg[i:i+delta])
        """
        crossing = np.concatenate((np.array([0]),(die_fun_avg[:-1]<0) * (die_fun_avg[1:] >=0)))
        deltaE_n = self.deltaE[self.deltaE>0]
        #deltaE_n = deltaE_n[50:-50]
        crossing_E = deltaE_n[crossing.astype('bool')]
        n = len(crossing_E)
        return crossing_E, n
    
    def cluster(self, n_clusters = 5, n_iterations = 30, based_upon = "sum"):
        #TODO: add other based_upons
        if based_upon == "sum":
            values = np.sum(self.data, axis = 2).flatten()
        if based_upon == "log":
            values = np.log(np.sum(self.data, axis = 2).flatten())
        else:
            values = np.sum(self.data, axis = 2).flatten()
        clusters_unsorted, r = k_means(values, n_clusters = n_clusters, n_iterations =n_iterations)
        self.clusters = np.sort(clusters_unsorted)[::-1]
        arg_sort_clusters = np.argsort(clusters_unsorted)[::-1]
        self.clustered = np.zeros(self.image_shape)
        for i in range(n_clusters):
            in_cluster_i = r[arg_sort_clusters[i]]
            self.clustered += (np.reshape(in_cluster_i, self.image_shape))*i
    
    
    #PLOTTING FUNCTIONS
    def plot_sum(self, title = None, xlab = None, ylab = None):
        """
        INPUT:
            self -- spectral image 
            title -- str, delfault = None, title of plot
            xlab -- str, default = None, x-label
            ylab -- str, default = None, y-label
        OUTPUT:
        Plots the summation over the intensity for each pixel in a heatmap.
        """
        #TODO: invert colours
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
            ax = sns.heatmap(np.sum(self.data, axis = 2), xticklabels=xticks, yticklabels=yticks)
        else:
            ax = sns.heatmap(np.sum(self.data, axis = 2))
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel(ylab)
        plt.show()
    
    def get_ticks(self, sig = 3, n_tick = 10):
        xlabels = np.zeros(self.x_axis.shape,dtype = object)
        xlabels[:] = ""
        each_n_pixels = math.floor(len(xlabels)/n_tick)
        for i in range(len(xlabels)):
            if i%each_n_pixels == 0:
                xlabels[i] = '%s' % float('%.3g' % self.x_axis[i])
        ylabels = np.zeros(self.y_axis.shape,dtype = object)
        ylabels[:] = ""
        each_n_pixels = math.floor(len(ylabels)/n_tick)
        for i in range(len(ylabels)):
            if i%each_n_pixels == 0:
                ylabels[i] = '%s' % float('%.3g' % self.y_axis[i])
        return xlabels, ylabels
                
    
    def plot_all(self, same_image = True, normalize = False, legend = False, 
                 range_x = None, range_y = None, range_E = None, signal = "EELS", log = False):
        #TODO: add titles and such
        if range_x is None:
            range_x = [0,self.image_shape[1]]
        if range_y is None:
            range_y = [0,self.image_shape[0]]
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
                    plt.title("Spectrum pixel: [" + str(j) +","+ str(i) + "]")
                    plt.xlabel("[eV]")
                    if range_E is not None:
                        plt.xlim(range_E)
                    if legend: 
                        plt.legend()
                signal_pixel = self.get_pixel_signal(i,j,signal)
                if normalize:
                    signal_pixel /= np.max(np.absolute(signal_pixel))
                if log:
                    signal_pixel = np.log(signal_pixel)
                    plt.ylabel("log intensity")
                plt.plot(self.deltaE, signal_pixel, label = "[" + str(j) +","+ str(i) + "]")
            if legend: 
                plt.legend()
                
    
    #STATIC METHODS
    @staticmethod
    def get_prefix(unit, SIunit = None, numeric = True):
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
                    if numeric: return 1
                    else: return prefix
            else:
                print("provided unit not same as target unit: " + unit + ", and " + SIunit)
                if numeric: return 1
                else: return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix
        
        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix == 'μ' or prefix == 'µ' or prefix == 'u':
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
    def calc_avg_ci(np_array, axis=0, ci = 16, return_low_high = True):
        avg = np.average(np_array, axis=axis)
        ci_low = np.nanpercentile(np_array,  ci, axis=axis)
        ci_high = np.nanpercentile(np_array,  100-ci, axis=axis)
        if return_low_high:
            return avg, ci_low, ci_high
        return avg, ci_high-ci_low
    
    #CLASS THINGIES
    def __getitem__(self, key):
        """ Determines behavior of `self[key]` """
        return self.data[key]
        #pass
    
    
    
    def __str__(self):
        if hasattr(self, 'name'):
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return 'Spectral image: ' + name_str + ", image size:"+ str(self.data.shape[0]) + 'x' + \
                    str(self.data.shape[1]) + ', deltaE range: [' + str(round(self.deltaE[0],3)) + ',' + \
                        str(round(self.deltaE[-1],3)) + '], deltadeltaE: ' + str(round(self.ddeltaE,3))
        
    def __repr__(self):
        data_str = "data * np.ones(" + str(self.shape) + ")"
        if hasattr(self, 'name'):
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return "Spectral_image(" + data_str +  ", deltadeltaE=" + str(round(self.ddeltaE, 3)) + name_str + ")"
        
    def __len__(self):
        return self.l
            

    

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
    
    return inp*ab[0] + ab[1]
    #pass

def find_scale_var(inp, min_out = 0.1, max_out=0.9):
    a = (max_out - min_out)/(inp.max()- inp.min())
    b = min_out - a*inp.min()
    return [a, b]
