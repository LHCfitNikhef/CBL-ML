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
import tensorflow.compat.v1 as tf
import seaborn as sns
import numpy as np
#from lmfit import Model
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm;

tf.get_logger().setLevel('ERROR')

_logger = logging.getLogger(__name__)


class Spectral_image():
    def __init__(self, data, deltadeltaE, pixelsize = None, beam_energy = None, collection_angle = None, name = None):
        #TODO: rewrite everything to work with rotated data??? Or rotate data itself??? Lazy or nette oplossing?
        self.data = np.swapaxes(np.swapaxes(data, 0,1), 1,2)
        self.l = self.data.shape[2]
        self.image_shape = self.data.shape[:2]
        self.ddeltaE = deltadeltaE
        self.deltaE = self.determine_deltaE()
        if pixelsize is not None:
            self.pixelsize = pixelsize
        if beam_energy is not None:
            self.beam_energy = beam_energy
        if collection_angle is not None:
            self.collection_angle = collection_angle
        if name is not None:
            self.name = name
    
    def determine_deltaE(self):
        data_avg = np.average(self.data, axis = (0,1))
        ind_max = np.argmax(data_avg)
        #l = self.data.shape[0]
        deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l-ind_max-1)*self.ddeltaE, self.l)
        return deltaE
    
    
    
    #RETRIEVING FUNCTIONS
    def get_data(self):
        return self.data
    
    def get_deltaE(self):
        return self.deltaE
    
    def get_metadata(self):
        meta_data = {}
        if self.beam_energy is not None:
            meta_data['beam_energy'] = self.beam_energy
        if self.collection_angle is not None:
            meta_data['collection_angl'] = self.collection_angle
        return meta_data
    
    def get_pixel_signal(self, i,j, signal = 'EELS'):
        if signal == 'EELS':
            return self.data[ i, j, :]
        elif signal == 'df_avg':
            return self.dielectric_function_im_avg[ i, j, :]
        else:
            return self.data[ i, j, :]
    
    
    #METHODS ON SIGNAL
    
    def cut(self, E1, E2):
        #TODO
        pass
    
    def cut_image(self, range1, range2):
        self.data = self.data[range1[0]:range1[1],range2[0]:range2[1]]
        self.image_shape = self.data.shape[:2]
    
    
    
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
        #x = df_sample.iloc[2].x_shifted
        #y = df_sample.iloc[2].y_norm
        
        #x_lim = 1.3
        
        #x_exp = x[x<2][ x[x<2]> x_lim]
        #y_exp = y[x<2][ x[x<2]> x_lim]
        
        #y_offset = y_exp[0]
        
        #y_fit_exp, cov = scipy.optimize.curve_fit(exp_decay, x_exp, y_exp)
        
        """
        plt.figure()
        plt.plot(x,y)
        plt.plot(x_exp, exp_decay(x_exp, y_fit_exp[0]))
        plt.xlim(0,5)
        plt.ylim(0,0.02)
        """
        
        #y_ZLP = np.concatenate((y[x<= x_lim], exp_decay(x[x>x_lim], y_fit_exp[0])))
        
        y = self.get_pixel_signal(i,j)
        r = 3 #Drude model, can also use estimation from exp. data
        A = y[-1]
        n_times_extra = 2
        sem_inf = next_fast_len(n_times_extra*self.l)
        
        
        
        y_extrp = np.zeros(sem_inf)
        y_ZLP_extrp = np.zeros(sem_inf)
        x_extrp = np.linspace(self.deltaE[0]- self.l*self.ddeltaE, sem_inf*self.ddeltaE+self.deltaE[0]- self.l*self.ddeltaE, sem_inf)
        
        x_extrp = np.linspace(self.deltaE[0], sem_inf*self.ddeltaE+self.deltaE[0], sem_inf)

        
        #y_ZLP_extrp[self.l:self.l*2] = ZLP
        
        #y_extrp[self.l:self.l*2] = y
        #x_extrp[self.l:self.l*2] = self.deltaE[-self.l:]
        
        #y_extrp[2*self.l:] = A*np.power(1+x_extrp[2*self.l:]-x_extrp[2*self.l],-r)
        
        y_ZLP_extrp[:self.l] = ZLP        
        y_extrp[:self.l] = y
        x_extrp[:self.l] = self.deltaE[-self.l:]
        
        y_extrp[self.l:] = A*np.power(1+x_extrp[self.l:]-x_extrp[self.l],-r)
        
        x = x_extrp
        y = y_extrp
        y_ZLP = y_ZLP_extrp
        #y_EEL = y - y_ZLP
        
        
        
        z_nu = CFT(x,y_ZLP)
        i_nu = CFT(x,y)
        abs_i_nu = np.absolute(i_nu)
        max_i_nu = np.max(abs_i_nu)
        i_nu_copy = np.copy(i_nu)
        #i_nu[abs_i_nu<max_i_nu*0.00000000000001] = 0
        N_ZLP = 1#scipy.integrate.cumtrapz(y_ZLP, x, initial=0)[-1]#1 #arbitrary units??? np.sum(EELZLP)
        
        s_nu = N_ZLP*np.log(i_nu/z_nu)
        j1_nu = z_nu*s_nu/N_ZLP
        #s_nu_2 = s_nu
        #s_nu_2[np.isnan(s_nu)] = 0#1E10 #disregard NaN values, but setting them to 0 doesnt seem fair, as they should be inf
        
        """
        plt.figure()
        plt.title("s_nu and j1_nu")
        plt.plot(s_nu)
        plt.plot(j1_nu)
        """
        
        #s_nu[150:1850] = 0
        S_E = np.real(iCFT(x,s_nu))
        s_nu_nc = s_nu
        s_nu_nc[500:-500] = 0
        S_E_nc = np.real(iCFT( x,s_nu_nc))
        J1_E = np.real(iCFT(x,j1_nu))
        
        return J1_E[:self.l]
    
    #METHODS ON ZLP
    #CALCULATING ZLPs FROM PRETRAINDED MODELS
    def calculate_general_ZLPs(self, path_to_models):
        tf.reset_default_graph()
        #TODO: redifine paths based upon new fitter saving modes
        #TODO: rewrite to have models as tributes?
        
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
        
        self.dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
        self.dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
        self.dE0 = np.round(self.dE1 - .5, 2) 
        
        all_files_cost = glob.glob(path_to_data + path_cost)
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
        
        tf.get_default_graph()
        tf.disable_eager_execution()
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        
        
        def make_model(inputs, n_outputs):
            hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
            hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
            hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
            output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
            return output
        
        x = tf.placeholder("float", [None, 1], name="x")
        predictions = make_model(x, 1)
        
        
        prediction_file = pd.DataFrame()
        len_data = self.l
        predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
        predict_x = self.deltaE.reshape(self.l,1)
        
        self.ZLPs_gen = np.zeros((count, len_data))
        with tf.Session() as sess: #TODO: gives warning
            sess.run(tf.global_variables_initializer())
            
            for i in range(0,len(good_files)):
                if good_files[i] == 1:
                    best_model = 'Models_oud/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                    saver = tf.train.Saver(max_to_keep=1000)
                    saver.restore(sess, best_model)
        
                    extrapolation = sess.run(predictions, #TODO: RESTARTS KERNEL!!!!!
                                            feed_dict={
                                            x: predict_x
                                            })
                    prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
                    self.ZLPs_gen[i, :] = np.exp(extrapolation)#.reshape(len_data,)
        
        
        
    
    def calc_ZLPs_gen2(self,  specimen = 4):
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
        len_data = self.l
        predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
        predict_x = self.deltaE.reshape(len_data,1)
        
        self.ZLPs_gen = np.zeros((count, len_data))
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
                    self.ZLPs_gen[j,:] = np.exp(extrapolation.reshape(len_data,))
                    prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
                    j += 1
        
        self.dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
        self.dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
        self.dE0 = np.round(self.dE1 - .5, 2) 
        
        #return ZLPs_gen, dE0, dE1, dE2
    
    def calc_ZLPs(self, i,j):
        ### Definition for the matching procedure
        signal = self.get_pixel_signal(i,j)
        
        if not hasattr(self, 'ZLPs_gen'):
            self.calc_ZLPs_gen2("iets")
        
        def matching( signal, ind_ZLP):
            gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]
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
        
        count = self.ZLPs_gen.shape[0]
        ZLPs = np.zeros(self.ZLPs_gen.shape) #np.zeros((count, len_data))
        
        
        for k in range(count): 
            ZLPs[k,:] = matching(signal, k)#matching(energies, np.exp(mean_k), data)
            
        return ZLPs
        
    
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


    
    def im_dielectric_function(self, track_process = False):
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
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel " , i,j)
                data_ij = self.get_pixel_signal(i,j)#[self.deltaE>0]
                ZLPs = self.calc_ZLPs(i,j)#[:,self.deltaE>0]
                dielectric_functions = (1+1j)* np.zeros(ZLPs[:,self.deltaE>0].shape)
                S_ss = np.zeros(ZLPs[:,self.deltaE>0].shape)
                ts = np.zeros(ZLPs.shape[0])            
                IEELSs = np.zeros(ZLPs.shape)
                for k in range(ZLPs.shape[0]):
                    ZLP_k = ZLPs[k,:]
                    N_ZLP = np.sum(ZLPs)
                    IEELS = data_ij-ZLP_k
                    IEELS = self.deconvolute(i, j, ZLP_k)
                    IEELSs[k,:] = IEELS
                    #TODO: FIX ZLP: now becomes very negative!!!!!!!
                    #TODO: VERY IMPORTANT
                    dielectric_functions[k,:], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP = N_ZLP, n =3)
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
        #return crossings_E, crossings_n
    
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
    
    
    def plot_sum(self, title = None, xlab = None, ylab = None):
        if hasattr(self, 'name'):
            name = self.name
        else:
            name = ''
        plt.figure()
        if title is not None:
            plt.title("intgrated intensity spectrum " + name)
        else:
            plt.title(title)
        ax = sns.heatmap(np.sum(self.data, axis = 2))
        if not hasattr(self, 'pixelsize'):
            plt.xlabel(self.pixelsize)
            plt.ylabel(self.pixelsize)
        if xlab is not None:
            plt.xlabel(xlab)
        if ylab is not None:
            plt.ylabel = ylab
        plt.show()
        
    
            
def load_data(path_to_dmfile):
    dmfile = dm.fileDM(path_to_dmfile).getDataset(0)
    data = dmfile['data']
    ddeltaE = dmfile['pixelSize'][0]
    pixelSize = np.array(dmfile['pixelSize'][1:])
    energyUnit = dmfile['pixelUnit'][0]
    ddeltaE *= get_prefix(energyUnit, 'eV')
    pixelUnit = dmfile['pixelUnit'][1]
    pixelSize *= get_prefix(pixelUnit, 'm')
    image = Spectral_image(data, ddeltaE, pixelsize = pixelSize)
    return image
    



def get_prefix(unit, SIunit = None, numeric = True):
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
    
    #print(prefix, 'μ')
    #print(prefix.type)
    #print('μ'.type)
    
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
    
def exp_decay(x, r):
    """ y_offset * np.power(x - x[0], -r) """
    return y_offset * np.power(1 + x - x[0], -r)

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








    
#%%


#data = np.load("area03-eels-SI-aligned.npy")
#energies = np.load("area03-eels-SI-aligned_energy.npy")
    

#dielectric_function_im_avg, dielectric_function_im_std = im_dielectric_function(data, energies)


#%%
#crossings_E, crossings_n =  crossings_im(dielectric_function_im_avg, energies)


#%%

#plt.figure()
#plt.imshow(crossings_n, cmap='hot', interpolation='nearest')
#plt.


#ax = sns.heatmap(crossings_n)
#plt.show()

#%%
#dmfile = dm.fileDM('area03-eels-SI-aligned.dm4')
#data2 = dmfile.getDataset(0)

im = load_data('area03-eels-SI-aligned.dm4')#('pyfiles/area03-eels-SI-aligned.dm4')
im.cut_image([0,70], [95,100])
#im.cut_image([30,32],[5,6])
im.calc_ZLPs_gen2(specimen = 4)
im.smooth(window_len=50)
im.im_dielectric_function()
im.crossings_im()

#%%
"""
plt.figure()
plt.title("number of crossings real part dielectric function")
ax = sns.heatmap(im.crossings_n)
plt.show()

plt.figure()
plt.title("energy of first crossings real part dielectric function")
ax = sns.heatmap(im.crossings_E[:,:,0])
plt.show()


plt.figure()
plt.title("thickness of sample")
ax = sns.heatmap(im.thickness_avg)
plt.show()

"""