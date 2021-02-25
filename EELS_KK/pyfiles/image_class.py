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
#import tensorflow.compat.v1 as tf
#import seaborn as sns
import numpy as np
import math
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm

from k_means_clustering import k_means
#from train_NN import train_NN
#from train_NN_pc import train_NN_pc


#tf.get_logger().setLevel('ERROR')

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
    
    
    def get_cluster_spectra(self, conf_interval = 0.68, clusters = None, save_as_attribute = False):
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
    
    def cut(self, E1, E2):
        #TODO
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
    def calculate_general_ZLPs(self, path_to_models):
        tf.reset_default_graph()
        #TODO: redifine paths based upon new fitter saving modes
        #TODO: rewrite to have models as atributes?
        
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
        threshold = 6
        
        for i,j in enumerate(chi2_array):
            if j < threshold:
                good_files.append(1) 
                count +=1 
            else:
                good_fi
                
                les.append(0)
        
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
  
    
  
    def calc_ZLPs_gen2_I(self, path_model = "Models", d_string = '13.02.2021', count = 50, n_input = 1, specimen = 4):
        #GET GOODFILES KAN HEUL VEEL MAKKELIJKER< NIET NU
        
        path_to_data = path_model + '/Results/%(date)s/'% {"date": d_string} 
        
        path_predict = r'Predictions_*.csv'
        path_cost = r'Cost_*.csv' 
        
        all_files = glob.glob(path_to_data + path_predict)
        
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, delimiter=",",  header=0, usecols=[0,1,2], names=['x', 'y', 'pred'])
            li.append(df)
        
        
        
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
        
        count = len(good_files)
        
        
        tf.get_default_graph()
        tf.disable_eager_execution()
        
        #n_input = 1
        if n_input == 2:
            x = tf.placeholder("float", [None, 2], name="x")
        else:
            x = tf.placeholder("float", [None, 1], name="x")
        predictions = self.make_model(x, 1)
        
        
        prediction_file = pd.DataFrame()
        len_data = self.l * self.n_clusters
        predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
        
        if n_input == 1:
            len_data = self.l
            predict_x = self.deltaE.reshape(len_data,1)
        else:
            predict_x = np.empty((0,2))
            for i in range(self.n_clusters):
                predict_x = np.concatenate((predict_x, np.vstack((self.deltaE,np.ones(self.l)*self.clusters[i])).T))
    
        
        #good_files = np.ones(count)
        
        
        
        self.ZLPs_gen = np.zeros((count, len_data))
        j=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(0,len(good_files)):
                if good_files[i] == 1:
                    #print("ik kom hier")
                    if n_input == 1:#2: #HIER CHANGE #TODO
                        best_model = path_model + '/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                    else:
                        best_model = path_model + '_ni_1/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
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
        
        #self.dE1 = np.round(max(training_data['x'][(training_data['x']< 3)]),2)
        #self.dE2 = np.round(min(training_data['x'][(training_data['x']> 3)]),1)
        #self.dE0 = np.round(self.dE1 - .5, 2) 
        
    def calculate_general_ZLPs_I(self, path_to_models):
        tf.reset_default_graph()
        #TODO: redifine paths based upon new fitter saving modes
        #TODO: rewrite to have models as atributes?
        """
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
        """
        
        
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
        
        x = tf.placeholder("float", [None, 2], name="x")
        predictions = make_model(x, 1)
        
        
        prediction_file = pd.DataFrame()
        len_data = self.l
        predict_x = np.linspace(-0.5, 20, 1000).reshape(1000,1)
        predict_x = self.deltaE.reshape(self.l,1)
        
        predict_x = np.empty((0,2))
        for i in range(self.n_clusters):
            predict_x = np.concatenate((predict_x, np.vstack((self.deltaE,np.ones(self.l)*self.clusters[i])).T))
        len_data = self.l*self.n_clusters
        
        count = 40
        good_files = np.ones(count)
        
        d_string = '11.02.2021'
        
        self.ZLPs_gen = np.zeros((count, len_data))
        with tf.Session() as sess: #TODO: gives warning
            sess.run(tf.global_variables_initializer())
            
            for i in range(0,len(good_files)):
                if good_files[i] == 1:
                    best_model = 'Models_004_2/Best_models/%(s)s/best_model_%(i)s'% {'s': d_string, 'i': i}
                    saver = tf.train.Saver(max_to_keep=1000)
                    saver.restore(sess, best_model)
        
                    extrapolation = sess.run(predictions, #TODO: RESTARTS KERNEL!!!!!
                                            feed_dict={
                                            x: predict_x
                                            })
                    prediction_file['prediction_%(i)s' % {"i": i}] = extrapolation.reshape(len_data,)
                    self.ZLPs_gen[i, :] = np.exp(extrapolation)#.reshape(len_data,)

        
    @staticmethod
    def make_model(inputs, n_outputs):
        hidden_layer_1 = tf.layers.dense(inputs, 10, activation=tf.nn.sigmoid)
        hidden_layer_2 = tf.layers.dense(hidden_layer_1, 15, activation=tf.nn.sigmoid)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 5, activation=tf.nn.relu)
        output = tf.layers.dense(hidden_layer_3, n_outputs, name='outputs', reuse=tf.AUTO_REUSE)
        return output
    
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
        
        
        
        x = tf.placeholder("float", [None, 1], name="x")
        predictions = self.make_model(x, 1)
        
        
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
            gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]*np.max(signal)/np.max(self.ZLPs_gen[ind_ZLP,:]) #TODO!!!!, normalize?
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
        
    
    def train_ZLPs(self, n_clusters = None, conf_interval = 0.68, clusters = None):
        if not hasattr(self, "clustered"):
            if n_clusters is not None:
                self.cluster(n_clusters)
            else:
                self.cluster()
        elif n_clusters is not None and self.n_clusters != n_clusters:
            self.cluster(n_clusters)
        
        training_data = self.get_cluster_spectra( conf_interval = conf_interval, clusters = clusters)
        self.models = train_NN(self, training_data)


    def train_ZLPs_pc(self, n_clusters = None, conf_interval = 0.68, clusters = None):
        if not hasattr(self, "clustered"):
            if n_clusters is not None:
                self.cluster(n_clusters)
            else:
                self.cluster()
        elif n_clusters is not None and self.n_clusters != n_clusters:
            self.cluster(n_clusters)
        
        
        intensities = []
        for i in clusters:
            intensities.append(self.clusters[i])
        
        training_data = self.get_cluster_spectra( conf_interval = conf_interval, clusters = clusters)
        self.models = train_NN_pc(self, training_data, intensities = intensities)


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
        OUTPUT:
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
    def  plot_sum(self, title = None, xlab = None, ylab = None):
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
"""
im2 = Spectral_image.load_data('../dmfiles/h-ws2_eels-SI_003.dm4')#('pyfiles/area03-eels-SI-aligned.dm4')
im2.plot_sum()
for i in [5]:#[3,4,5,10]:
    im2.cluster(n_clusters = i)
    plt.figure()
    plt.title("spectral image, clustered with " + str(i) + " clusters")
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    xticks, yticks = im2.get_ticks()
    ax = sns.heatmap(im2.clustered, xticklabels=xticks, yticklabels=yticks)
    plt.show()
im2.calc_ZLPs_gen2_I()
"""
"""
im.train_ZLPs(conf_interval = 0.7)
"""


im = Spectral_image.load_data('../dmfiles/h-ws2_eels-SI_004.dm4')#('pyfiles/area03-eels-SI-aligned.dm4')
im.plot_sum()

for i in [5]:#[3,4,5,10]:
    im.cluster(n_clusters = i)
    plt.figure()
    plt.title("spectral image, clustered with " + str(i) + " clusters")
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    xticks, yticks = im.get_ticks()
    ax = sns.heatmap(im.clustered, xticklabels=xticks, yticklabels=yticks)
    plt.show()
"""
for i in range(im.n_clusters):
    im.train_ZLPs_pc(conf_interval = 0.7, clusters=[i])


cluster_ZLPs = np.zeros((im.n_clusters, 50, im.l))

plt.figure()
for i in range(im.n_clusters):
    im.calc_ZLPs_gen2_I(path_model= "Models" + str(i) + '_2', d_string="16.02.2021", count = 50, n_input=1)
    cluster_ZLPs[i,:len(im.ZLPs_gen),:] = im.ZLPs_gen
    avg,low,high = im.calc_avg_ci(im.ZLPs_gen)
    plt.fill_between(im.deltaE, low, high, alpha = 0.1)
    plt.plot(im.deltaE, avg, label = "cluser " + str(i))
plt.legend()
"""


"""
im.cut_image([0,70], [95,100])
#im.cut_image([40,41],[4,5])
im.calc_ZLPs_gen2(specimen = 4)
im.smooth(window_len=50)
im.im_dielectric_function()
im.crossings_im()

#%%

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
