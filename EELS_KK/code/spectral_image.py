import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import math
from scipy.fftpack import next_fast_len
import logging
from ncempy.io import dm
import os
import copy
import warnings
import torch
import bz2
import pickle
import _pickle as cPickle
from matplotlib import rc


from k_means_clustering import k_means
import training as train

_logger = logging.getLogger(__name__)


class SpectralImage:
    """
    The spectral image class that provides several tools to analyse spectral images with the zero-loss peak
    subtracted.

    Parameters
    ----------
    data: array_like
        Array containing the 3-D spectral image. The axes correspond to the x-axis, y-axis and energy-loss.
    deltadeltaE: float
        bin width in energy loss spectrum
    pixelsize: array_like, optional
        width of pixels
    beam_energy: float, optional
        Energy of electron beam in eV
    collection_angle: float, optional
        Collection angle of STEM in rad
    name: str, optional
        Title of the plots
    dielectric_function_im_avg
        average dielectric function for each pixel
    dielectric_function_im_std
        standard deviation of the dielectric function at each energy per pixel
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
    cluster_data: array_like
        filled with 2D numpy arrays. If save_as_attribute set to True, the cluster data is also saved as attribute
    deltaE: array_like
        shifted array of energy losses such that the zero point corresponds to the point of highest intensity.
    x_axis: array_like
        x-axis of the spectral image
    y_axis: array_like
        y-axis of the spectral image
    clusters: array_like
        cluster means of each cluster
    clustered: array_like
        A 2D array containing the index of the cluster to which each pixel belongs

    Examples
    --------
    An example how to train and anlyse a spectral image::

        dm4_path = 'path to dm4 file'
        im = SpectralImage.load_data(dm4_path)
        im.train_zlp(n_clusters=n_clusters,
                 n_rep=n_rep,
                 n_epochs=n_epochs,
                 bs_rep_num=bs_rep_num,
                 path_to_models=path_to_models,
                 display_step=display_step)

    """

    #  signal names
    DIELECTRIC_FUNCTION_NAMES = ['dielectric_function', 'dielectricfunction', 'dielec_func', 'die_fun', 'df', 'epsilon']
    EELS_NAMES = ['electron_energy_loss_spectrum', 'electron_energy_loss', 'EELS', 'EEL', 'energy_loss', 'data']
    IEELS_NAMES = ['inelastic_scattering_energy_loss_spectrum', 'inelastic_scattering_energy_loss',
                   'inelastic_scattering', 'IEELS', 'IES']
    ZLP_NAMES = ['zeros_loss_peak', 'zero_loss', 'ZLP', 'ZLPs', 'zlp', 'zlps']
    THICKNESS_NAMES = ['t', 'thickness', 'thick', 'thin']
    POOLED_ADDITION = ['pooled', 'pool', 'p', '_pooled', '_pool', '_p']

    # meta data names
    COLLECTION_ANGLE_NAMES = ["collection_angle", "col_angle", "beta"]
    BEAM_ENERGY_NAMES = ["beam_energy", "beam_E", "E_beam", "E0", "E_0"]

    m_0 = 5.1106E5  # eV, electron rest mass
    a_0 = 5.29E-11  # m, Bohr radius
    h_bar = 6.582119569E-16  # eV/s
    c = 2.99792458E8  # m/s

    def __init__(self, data, deltadeltaE, pixelsize=None, beam_energy=None, collection_angle=None, name=None,
                 dielectric_function_im_avg=None, dielectric_function_im_std=None, S_s_avg=None, S_s_std=None,
                 thickness_avg=None, thickness_std=None, IEELS_avg=None, IEELS_std=None, clusters=None, clustered=None, cluster_data=None, deltaE=None, x_axis=None, y_axis = None,
                 ZLP_models = None, scale_var_deltaE=None, crossings_E=None, crossings_n=None, data_smooth=None, dE1=None, n=None, pooled=None,
                 scale_var_log_sum_I = None, **kwargs):

        self.data = data
        self.ddeltaE = deltadeltaE
        self.deltaE = self.determine_deltaE()

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
        self.clusters = clusters
        self.clustered = clustered
        self.cluster_data = cluster_data
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.ZLP_models = ZLP_models
        self.scale_var_deltaE = scale_var_deltaE
        self.crossings_E = crossings_E
        self.crossings_n = crossings_n
        self.data_smooth = data_smooth
        self.dE1 = dE1
        self.n = n
        self.pooled = pooled
        self.scale_var_log_sum_I = scale_var_log_sum_I
        self.output_path = os.getcwd()

    def save_image(self, filename):
        """
        Function to save image, including all attributes, in pickle (.pkl) format. Image will be saved \
        at indicated location and name in filename input.

        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pkl", ".pkl" will be added.
        """
        if filename[-4:] != '.pkl':
            filename = filename + '.pkl'
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_compressed_image(self, filename):
        """
        Function to save image, including all attributes, in compressed pickle (.pbz2) format. Image will \
            be saved at location ``filename``. Advantage over :py:meth:`save_image() <save_image>` is that \
            the saved file has a reduced file size, disadvantage is that saving and reloading the image \
            takes significantly longer.


        Parameters
        ----------
        filename : str
            path to save location plus filename. If it does not end on ".pbz2", ".pbz2" will be added.

        """
        if filename[-5:] != '.pbz2':
            filename = filename + '.pbz2'
        self.compressed_pickle(filename, self)

    @staticmethod
    # Pickle a file and then compress it into a file with extension 
    def compressed_pickle(title, data):
        """
        Saves ``data`` at location ``title`` as compressed pickle.
        """
        with bz2.BZ2File(title, 'w') as f:
            cPickle.dump(data, f)

    @staticmethod
    def decompress_pickle(file):
        """
        Opens, decompresses and returns the pickle file at location ``file``.

        Parameters
        ----------
        file: str
            location where the pickle file is stored

        Returns
        -------
        data: SpectralImage
        """
        data = bz2.BZ2File(file, 'rb')
        data = cPickle.load(data)
        return data

    # %%GENERAL FUNCTIONS

    # %%PROPERTIES
    @property
    def l(self):
        """Returns length of :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object, i.e. num energy loss bins"""
        return self.data.shape[2]

    @property
    def image_shape(self):
        """Returns 2D-shape of :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object"""
        return self.data.shape[:2]

    @property
    def shape(self):
        """Returns 3D-shape of :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object"""
        return self.data.shape

    @property
    def n_clusters(self):
        """Returns the number of clusters in the :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object."""
        return len(self.clusters)

    @property
    def n_spectra(self):
        """
        Returns the number of spectra present in :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object

        Returns
        -------
        nspectra: int
            number of spectra in spectral image"""
        nspectra = np.product(self.image_shape)
        return nspectra

    @classmethod
    def load_data(cls, path_to_dmfile, load_additional_data=False):
        """
        Load the .dm4 spectral image and return a :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` instance.

        Parameters
        ----------
        path_to_dmfile: str
            location of .dm4 file
        load_additional_data: bool, optional
            Default is `False`. If `True`

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` instance of the dm4 file
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
    def load_spectral_image(cls, path_to_pickle):
        """
        Loads :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` instance from a pickled file.

        Parameters
        ----------
        path_to_pickle : str
            path to the pickled image file.

        Raises
        ------
        ValueError
            If path_to_pickle does not end on the desired format .pkl.
        FileNotFoundError
            If path_to_pickle does not exists.

        Returns
        -------
        SpectralImage
            :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object (i.e. including all attributes) loaded from pickle file.

        """
        if path_to_pickle[-4:] != '.pkl':
            raise ValueError("please provide a path to a pickle file containing a Spectral_image class object.")
        if not os.path.exists(path_to_pickle):
            raise FileNotFoundError('pickled file: ' + path_to_pickle + ' not found')
        with open(path_to_pickle, 'rb') as pickle_im:
            image = pickle.load(pickle_im)
        return image

    @classmethod
    def load_compressed_Spectral_image(cls, path_to_compressed_pickle):
        """
        Loads spectral image from a compressed pickled file. This will take longer than loading from non compressed pickle.

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
        image : SpectralImage
             :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` instance loaded from the compressed pickle file.
        """
        if path_to_compressed_pickle[-5:] != '.pbz2':
            raise ValueError(
                "please provide a path to a compressed .pbz2 pickle file containing a Spectrall_image class object.")
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
            to what specimen, and set the values by running::

             image.n[cluster_i] = n_i

        Parameters
        ----------
        n : float
            refractive index of sample.
        n_background : float, optional
            if defined: the refractive index of the background/vacuum. This value will automatically be \
            assigned to pixels belonging to the thinnest cluster.
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
        Determines the energy losses of the spectral image, based on the bin width of the energy loss.
        It shifts the ``self.deltaE`` attribute such that the zero point corresponds with the point of highest
        intensity.

        Returns
        -------
        deltaE: array_like
            Array of :math:`\Delta E` values
        """
        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l - ind_max - 1) * self.ddeltaE, self.l)
        return deltaE

    def calc_axes(self):
        """
        Determines the  x_axis and y_axis of the spectral image. Stores them in ``self.x_axis`` and ``self.y_axis`` respectively.
        """
        self.y_axis = np.linspace(0, self.image_shape[0] - 1, self.image_shape[0])
        self.x_axis = np.linspace(0, self.image_shape[1] - 1, self.image_shape[1])
        if self.pixelsize is not None:
            self.y_axis *= self.pixelsize[0]
            self.x_axis *= self.pixelsize[1]

    def get_pixel_signal(self, i, j, signal='EELS'):
        """
        Retrieves the spectrum at pixel (``i``, ``j``)`.

        Parameters
        ----------
        i: int
            x-coordinate of the pixel
        j: int
            y-coordinate of the pixel
        signal: str, optional
            The type of signal that is requested, should comply with the defined names. Set to `EELS` by default.

        Returns
        -------
        signal : array_like
            Array with the requested signal from the requested pixel
        """

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

    def get_cluster_spectra(self, conf_interval=1, clusters=None, signal="EELS"):
        """
        Returns a clustered spectral image.

        Parameters
        ----------
        conf_interval : float, optional
            The ratio of spectra returned. The spectra are selected based on the 
            based_on value. The default is 1.
        clusters : list of ints, optional
            list with all the cluster labels.
        signal: str, optional
            Description of signal, ``"EELS"`` by default.

        Returns
        -------
        cluster_data : array_like
            An array with size equal to the number of clusters. Each entry is a 2D array that contains all the spectra within that cluster.
        """
        # TODO: check clustering before everything
        if clusters is None:
            clusters = range(self.n_clusters)

        integrated_int = np.sum(self.data, axis=2)
        cluster_data = np.zeros(len(clusters), dtype=object)

        j = 0
        for i in clusters:
            data_cluster = self.get_image_signals(signal)[self.clustered == i]
            if conf_interval < 1:
                intensities_cluster = integrated_int[self.clustered == i]
                arg_sort_int = np.argsort(intensities_cluster)
                ci_lim = round((1 - conf_interval) / 2 * intensities_cluster.size)  # TODO: ask juan: round up or down?
                data_cluster = data_cluster[arg_sort_int][ci_lim:-ci_lim]
            cluster_data[j] = data_cluster
            j += 1

        return cluster_data

    def deltaE_to_arg(self, E):
        if type(E) in [int, float]:
            return np.argmin(np.absolute(self.deltaE - E))

        for i in len(E):
            E[i] = np.argmin(np.absolute(self.deltaE - E[i]))
        return E
        # TODO: check if works

    # %%METHODS ON SIGNAL

    def cut(self, E1=None, E2=None, in_ex="in"):
        """
        Cuts the spectral image at ``E1`` and ``E2`` and keeps only the part in between.

        Parameters
        ----------
        E1 : float, optional
            lower cut. The default is ``None``, which means no cut is applied.
        E2 : float, optional
            upper cut. The default is ``None``, which means no cut is applied.

        """
        if (E1 is None) and (E2 is None):
            raise ValueError("To cut energy spectra, please specify minimum energy E1 and/or maximum energy E2.")
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

    def cut_image(self, range_width, range_height):
        # TODO: add floats for cutting to meter sizes?
        self.data = self.data[range_height[0]:range_height[1], range_width[0]:range_width[1]]
        self.y_axis = self.y_axis[range_height[0]:range_height[1]]
        self.x_axis = self.x_axis[range_width[0]:range_width[1]]

    # TODO
    def samenvoegen(self):
        pass

    def smooth(self, window_len=10, window='hanning', keep_original=False):
        """
        Smooth the data using a window length ``window_len``.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        
        Parameters
        ----------
        window_len: int, optional
            The dimension of the smoothing window; should be an odd integer.
        window: str, optional
            the type of window from ``"flat"``, ``"hanning"``,  ``"bartlett"``, ``"blackman"``.
            ``"flat"`` will produce a moving average smoothing.
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

    def calc_zlps_matched(self, i, j, signal='EELS', select_ZLPs=False, **kwargs):
        """
        Returns the shape-(M, N) array of matched ZLP model predictions at pixel (``i``, ``j``) after training.
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        i: int
            horizontal pixel.
        j: int
            vertical pixel.
        signal: str, bool
            Description of signal type. Set to ``"EELS"`` by default.
        select_ZLPs: bool, optional
            Filter out ZLP models based on their arclength in the extrapolating region in between :math:`\Delta E_I` and :math:`\Delta E_{II}`.
        kwargs: dict, optional
            Additional keyword arguments.

        Returns
        -------
        predictions: numpy.ndarray, shape=(M, N)
            The matched ZLP predictions at pixel (``i``, ``j``).
        """
        # Definition for the matching procedure
        def matching(signal, gen_i_ZLP, dE1):
            dE0 = dE1 - 0.5
            dE2 = dE1 * 3
            # gen_i_ZLP = self.ZLPs_gen[ind_ZLP, :]#*np.max(signal)/np.max(self.ZLPs_gen[ind_ZLP,:]) #TODO!!!!, normalize?
            delta = (dE1 - dE0) / 10  # lau: 3

            # factor_NN = np.exp(- np.divide((self.deltaE[(self.deltaE<dE1) & (self.deltaE >= dE0)] - dE1)**2, delta**2))
            factor_NN = 1 / (1 + np.exp(
                -(self.deltaE[(self.deltaE < dE1) & (self.deltaE >= dE0)] - (dE0 + dE1) / 2) / delta))
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

        ZLPs_gen = self.calc_zlps(i, j, signal, select_ZLPs, **kwargs)

        count = len(ZLPs_gen)
        ZLPs = np.zeros((count, self.l))  # np.zeros((count, len_data))

        signal = self.get_pixel_signal(i, j, signal)
        cluster = self.clustered[i, j]

        dE1 = self.dE1[1, int(cluster)]
        for k in range(count):
            predictions = ZLPs_gen[k]
            ZLPs[k, :] = matching(signal, predictions, dE1)  # matching(energies, np.exp(mean_k), data)
        return ZLPs

    def calc_zlps(self, i, j, signal='EELS', select_ZLPs=False, **kwargs):
        """
        Returns the shape-(M, N) array of ZLP model predictions at pixel (``i``, ``j``) after training, where
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.


        Parameters
        ----------'
        i: int
            horizontal pixel.
        j: int
            vertical pixel.
        signal: str, bool
            Description of signal type. Set to ``"EELS"`` by default.
        select_ZLPs: bool, optional
            Filter out ZLP models based on their arclength in the extrapolating region in between :math:`\Delta E_I` and :math:`\Delta E_{II}`.
        kwargs: dict, optional
            Additional keyword arguments.


        Returns
        -------
        predictions: numpy.ndarray, shape=(M, N)
            The ZLP predictions at pixel (``i``, ``j``).

        """
        # Definition for the matching procedure
        signal = self.get_pixel_signal(i, j, signal)

        if self.ZLP_models is None:
            try:
                self.load_zlp_models(**kwargs)
            except:
                self.load_zlp_models()

        count = len(self.ZLP_models)

        predictions = np.zeros((count, self.l))  # np.zeros((count, len_data))

        if self.scale_var_deltaE is None:
            self.scale_var_deltaE = find_scale_var(self.deltaE)

        if self.scale_var_log_sum_I is None:
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
                predictions[k, :] = np.exp(model(predict_x.float()).flatten())

        if select_ZLPs:
            predictions = predictions[self.select_ZLPs(predictions)]

        return predictions

    def select_ZLPs(self, ZLPs, dE1=None):
        if dE1 is None:
            dE1 = min(self.dE1[1, :])
            dE2 = 3 * max(self.dE1[1, :])
        else:
            dE2 = 3 * dE1

        ZLPs_c = ZLPs[:, (self.deltaE > dE1) & (self.deltaE < dE2)]
        low = np.nanpercentile(ZLPs_c, 1, axis=0)
        high = np.nanpercentile(ZLPs_c, 90, axis=0)

        threshold = (low[0] + high[0]) / 20

        low[low < threshold] = 0
        high[high < threshold] = threshold

        check = (ZLPs_c < low) | (ZLPs_c >= high)
        check_sum = np.sum(check, axis=1) / check.shape[1]

        threshold = np.nanpercentile(check_sum, 85)

        return (check_sum <= threshold)

    def train_zlp(self, n_clusters=5, conf_interval=1, clusters=None, signal='EELS', **kwargs):
        """
        Train the ZLP on the spectral image.

        The spectral image is clustered in ``n_clusters`` clusters, according to e.g. the integrated intensity or thickness.
        A random spectrum is then taken from each cluster, which together defines one replica. The training is initiated
        by calling :py:meth:`train_zlp_scaled() <training.train_zlp_scaled>`.

        Parameters
        ----------
        n_clusters: int, optional
            number of clusters
        conf_interval: int, optional
            Default is 1
        clusters
        signal: str, optional
            Type of spectrum. Set to EELS by default.
        **kwargs
            Additional keyword arguments that are passed to the method :py:meth:`train_zlp_scaled() <training.train_zlp_scaled>` in the :py:mod:`training` module.
        """
        self.cluster(n_clusters)

        training_data = self.get_cluster_spectra(conf_interval=conf_interval, signal=signal)
        train.train_zlp_scaled(self, training_data, **kwargs)

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

        return (np.std(predictions) / np.average(predictions)) > 1E-3  # very small --> straight line

    def calc_zlp_ntot(self, ntot):
        """
        Returns the shape-(M, N) array of zlp model predictions at the scaled log integrated intensity ``ntot``.
        M and N correspond to the number of model predictions and :math:`\Delta E` s respectively.

        Parameters
        ----------
        ntot: float
            Log integrated intensity (rescaled)
        """
        deltaE = np.linspace(0.1, 0.9, self.l)
        predict_x_np = np.zeros((self.l, 2))
        predict_x_np[:, 0] = deltaE
        predict_x_np[:, 1] = ntot

        predict_x = torch.from_numpy(predict_x_np)
        count = len(self.ZLP_models)
        ZLPs = np.zeros((count, self.l))

        for k in range(count):
            model = self.ZLP_models[k]
            with torch.no_grad():
                predictions = np.exp(model(predict_x.float()).flatten())
            ZLPs[k, :] = predictions

        return ZLPs

    def plot_zlp_ntot(self):
        """
        Plot the trained ZLP including uncertainties for the cluster means.
        """

        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 10})
        rc('text', usetex=True)
        print("plotting zlp for clusters...")
        fig, ax = plt.subplots(dpi=200)
        ax.set_title(r"$\rm{Predicted\;ZLPs\;for\;cluster\;means}$")
        ax.set_xlabel(r"$\rm{Energy\;loss\;[eV]}$")
        ax.set_ylabel(r"$I_{\rm{EELS}}\;\rm{[a.u.]}$")

        cluster_means = self.clusters
        scaled_int = [self.scale_var_log_sum_I[0] * i + self.scale_var_log_sum_I[1] for i in cluster_means]

        for i, cluster_mean_scaled in enumerate(scaled_int):
            zlps = self.calc_zlp_ntot(cluster_mean_scaled)

            low = np.nanpercentile(zlps, 16, axis=0)
            high = np.nanpercentile(zlps, 84, axis=0)
            median = np.nanpercentile(zlps, 50, axis=0)

            ax.fill_between(self.deltaE, low, high, alpha=0.3)
            label = r"$\rm{Vacuum}$" if i == 0 else r"$\rm{Cluster\;%d}$" % i
            ax.plot(self.deltaE, median, label=label)

        ax.set_ylim(1, 1e6)
        ax.set_xlim(0, 6)
        ax.legend()
        plt.yscale('log')
        fig.savefig(os.path.join(self.output_path, 'scaled_int.pdf'))


    def load_zlp_models(self, path_to_models, plot_chi2=False, idx=None):
        """
        Loads the trained ZLP models and stores them in ``self.ZLP_models``. Models that have a :math:`\chi^2 > \chi^2_{\mathrm{mean}} + 5\sigma` are
        discarded, where :math:`\sigma` denotes the 68% CI.

        Parameters
        ----------
        path_to_models: str
            Location where the model predictions have been stored after training.
        plot_chi2: bool, optional
            When set to `True`, plot and save the :math:`\chi^2` distribution.
        idx: int, optional
            When specified, only the zlp labelled by ``idx`` is loaded, instead of all model predictions.

        """

        if not os.path.exists(path_to_models):
            print(
                "No path " + path_to_models + " found. Please ensure spelling and that there are models trained.")
            return

        self.ZLP_models = []

        path_to_models += (path_to_models[-1] != '/') * '/'
        path_dE1 = "dE1.txt"
        model = train.MLP(num_inputs=2, num_outputs=1)
        self.dE1 = np.loadtxt(os.path.join(path_to_models, path_dE1))

        path_scale_var = 'scale_var.txt'
        self.scale_var_log_sum_I = np.loadtxt(os.path.join(path_to_models, path_scale_var))
        try:
            path_scale_var_deltaE = 'scale_var_deltaE.txt'
            self.scale_var_deltaE = np.loadtxt(os.path.join(path_to_models, path_scale_var_deltaE))
            print("found delta E vars")
        except:
            pass

        if self.clustered is not None:
            if self.n_clusters != self.dE1.shape[1]:
                print("image clustered in ", self.n_clusters, " clusters, but ZLP-models take ", self.dE1.shape[1],
                      " clusters, reclustering based on models.")
                self.cluster_on_cluster_values(self.dE1[0, :])
        else:
            self.cluster_on_cluster_values(self.dE1[0, :])

        if idx is not None:
            with torch.no_grad():
                model.load_state_dict(torch.load(os.path.join(path_to_models, "nn_rep_{}".format(idx))))
            self.ZLP_models.append(copy.deepcopy(model))
            return

        path_costs_test = "costs_test_"
        path_costs_train = "costs_train_"
        files_costs_test = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_costs_test)]
        files_costs_train = [filename for filename in os.listdir(path_to_models) if filename.startswith(path_costs_train)]

        cost_tests = []
        cost_trains = []
        nn_rep_idx = []

        for path_cost_test, path_cost_train in zip(files_costs_test, files_costs_train):
            start = path_cost_test.find("test_") + len("test_")
            end = path_cost_test.find(".txt")
            bs_rep_num = int(path_cost_test[start:end])

            path_tests = os.path.join(path_to_models, path_cost_test)
            path_trains = os.path.join(path_to_models, path_cost_train)

            n_rep = 0
            with open(path_tests) as f:
                for line in f:
                    cost_tests.append(float(line.strip()))
                    n_rep += 1

            with open(path_trains) as f:
                for line in f:
                    cost_trains.append(float(line.strip()))

            save_idx_low = n_rep * bs_rep_num - n_rep + 1
            save_idx_high = n_rep * bs_rep_num

            nn_rep_idx.extend(range(save_idx_low, save_idx_high + 1))

        nn_rep_idx = np.array(nn_rep_idx)
        cost_tests = np.array(cost_tests)
        cost_trains = np.array(cost_trains)

        cost_tests_mean = np.mean(cost_tests)
        cost_tests_std = np.percentile(cost_tests, 68)
        threshold_costs_tests = cost_tests_mean + 5 * cost_tests_std
        cost_tests = cost_tests[cost_tests < threshold_costs_tests]

        cost_trains_mean = np.mean(cost_trains)
        cost_trains_std = np.percentile(cost_trains, 68)
        threshold_costs_trains = cost_trains_mean + 5 * cost_trains_std

        nn_rep_idx = nn_rep_idx[cost_trains < threshold_costs_trains]
        cost_trains = cost_trains[cost_trains < threshold_costs_trains]

        # plot the chi2 distributions
        if plot_chi2:
            fig, ax = plt.subplots(figsize=(1.1 * 10, 1.1 * 6))
            plt.hist(cost_trains, label=r'$\rm{Training}$', bins=40, range=(0, 5* cost_tests_std), alpha=0.4)
            plt.hist(cost_tests, label=r'$\rm{Validation}$', bins=40, range= (0, 5* cost_tests_std), alpha=0.4)
            plt.title(r'$\chi^2\;\rm{distribution}$')
            plt.xlabel(r'$\chi^2$')
            plt.legend(frameon=False, loc='upper right')
            fig.savefig(os.path.join(self.output_path, 'chi2_dist_selected.pdf'))

        for idx in nn_rep_idx.flatten():
            path = os.path.join(path_to_models, 'nn_rep_{}'.format(idx))
            model.load_state_dict(torch.load(path))
            self.ZLP_models.append(copy.deepcopy(model))


    # METHODS ON DIELECTRIC FUNCTIONS
    def calc_thickness(self, spect, n, N_ZLP=1):
        """
        Calculates thickness from sample data, using Egerton [1]_

        Parameters
        ----------
        spect : array_like
            spectral image
        n : float
            refraction index
        N_ZLP: float or int
            Set to 1 by default, for already normalized EELS spectra.

        Returns
        -------
        te: float
            thickness

        Notes
        -----
        Surface scatterings are not corrected for. If you wish to correct
        for surface scatterings, please extract the thickness ``t`` from :py:meth:`kramers_kronig_hs() <kramers_kronig_hs>`.


        .. [1] Ray Egerton, "Electron Energy-Loss Spectroscopy in the Electron
           Microscope", Springer-Verlag, 2011.

        """
        me = self.m_0
        e0 = self.e0
        beta = self.beta

        eaxis = self.deltaE[self.deltaE > 0]  # axis.axis.copy()
        y = spect[self.deltaE > 0]
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

        References
        ----------
    
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
        ke = e0 * (1 + e0 / 2. / me) / (1 + e0 / me) ** 2  # m0 v**2
        tgt = e0 * (2 * me + e0) / (me + e0)
        rk0 = 2590 * (1 + e0 / me) * np.sqrt(2 * ke / me)  # me c**2 / (hbar c) gamma sqrt(2Ekin /(me c**2))

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
            q = -2 * np.fft.fft(Im, esize).imag / esize  # TODO : min twee?????

            q[:l] *= -1
            q = np.fft.fft(q)
            # Final touch, we have Re(1/eps)
            Re = q[:l].real + 1  # TODO: plus 1???
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

            if iterations > 0 and N_ZLP is not None:  # TODO: loop weghalen.
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

    def KK_pixel(self, i, j, signal='EELS', select_ZLPs=False, **kwargs):
        """
        Perform a Kramer-Krönig analysis on pixel (``i``, ``j``).


        Parameters
        ----------
        i : int
            x-coordinate of the pixel
        j : int
            y-coordinate of the pixel.

        Returns
        -------
        dielectric_functions : array_like
            Collection dielectric-functions replicas at pixel (``i``, ``j``).
        ts : float
            Thickness.
        S_ss : array_like
            Surface scatterings.
        IEELSs : array_like
            Deconvonluted EELS spectrum.

        """
        # data_ij = self.get_pixel_signal(i,j)#[self.deltaE>0]
        ZLPs = self.calc_zlps_matched(i, j, select_ZLPs=select_ZLPs)  # [:,self.deltaE>0]

        dielectric_functions = (1 + 1j) * np.zeros(ZLPs[:, self.deltaE > 0].shape)
        S_ss = np.zeros(ZLPs[:, self.deltaE > 0].shape)
        ts = np.zeros(ZLPs.shape[0])
        IEELSs = np.zeros(ZLPs.shape)
        max_ieels = np.zeros(ZLPs.shape[0])
        n = self.n[self.clustered[i, j]]
        for k in range(ZLPs.shape[0]):
            ZLP_k = ZLPs[k, :]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k)
            IEELSs[k, :] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            if signal in self.EELS_NAMES:
                dielectric_functions[k, :], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP=N_ZLP, n=n, **kwargs)
            else:
                ts[k] = self.calc_thickness(IEELS, n, N_ZLP)
        if signal in self.EELS_NAMES:
            return dielectric_functions, ts, S_ss, IEELSs, max_ieels

        IEELSs_OG = IEELSs
        ts_OG = ts
        max_OG = max_ieels

        ZLPs_signal = self.calc_zlps_matched(i, j, signal=signal, select_ZLPs=select_ZLPs)
        dielectric_functions = (1 + 1j) * np.zeros(ZLPs_signal[:, self.deltaE > 0].shape)
        S_ss = np.zeros(ZLPs_signal[:, self.deltaE > 0].shape)
        ts = np.zeros(ZLPs_signal.shape[0])
        IEELSs = np.zeros(ZLPs_signal.shape)
        max_ieels = np.zeros(ZLPs_signal.shape[0])

        for k in range(ZLPs_signal.shape[0]):
            ZLP_k = ZLPs_signal[k, :]
            N_ZLP = np.sum(ZLP_k)
            IEELS = self.deconvolute(i, j, ZLP_k, signal=signal)
            IEELSs[k] = IEELS
            max_ieels[k] = self.deltaE[np.argmax(IEELS)]
            dielectric_functions[k, :], ts[k], S_ss[k] = self.kramers_kronig_hs(IEELS, N_ZLP=N_ZLP, n=n, **kwargs)

        return [ts_OG, IEELSs_OG, max_OG], [dielectric_functions, ts, S_ss, IEELSs, max_ieels]

    def optical_absorption_coeff(self, dielectric_function):

        # TODO: now assuming one input for dielectric function. We could check for dimentions, and do everything at once??

        eps1 = np.real(dielectric_function)
        E = self.deltaE[self.deltaE > 0]

        mu = E / (self.h_bar * self.c) * np.power(2 * np.absolute(dielectric_function) - 2 * eps1, 0.5)

        return mu

        pass

    def im_dielectric_function(self, track_process=False, plot=False, save_index=None, save_path="KK_analysis"):
        """
        Computes the dielectric function by performing a Kramer-Krönig analysis at each pixel.

        Parameters
        ----------
        track_process: bool, optional
            default is `False`, if `True`,  outputs for each pixel the program that is busy with that pixel.
        plot: bool, optional
            default is `False`, if `True`, plots all calculated dielectric functions
        save_index: int, optional
            optional labelling to include in ``save_path``.
        save_path: str, optional
            location where the dielectric function, SSD and thickness are stored.
        """

        self.dielectric_function_im_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.dielectric_function_im_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_avg = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.S_s_std = (1 + 1j) * np.zeros(self.data[:, :, self.deltaE > 0].shape)
        self.thickness_avg = np.zeros(self.image_shape)
        self.thickness_std = np.zeros(self.image_shape)
        self.IEELS_avg = np.zeros(self.data.shape)
        self.IEELS_std = np.zeros(self.data.shape)
        if plot:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots()
        for i in range(self.image_shape[0]):
            for j in range(self.image_shape[1]):
                if track_process: print("calculating dielectric function for pixel ", i, j)

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

    def optical_absorption_coeff_im(self):
        # TODO!!
        pass

    def crossings_im(self):
        """
        Determines the number of crossings of the real part of dielectric function at all pixels together with the associated
        ``dE`` values.
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

    def cluster(self, n_clusters=5, based_on="log", **kwargs):
        """
        Clusters the spectral image into clusters according to the (log) integrated intensity at each
        pixel. Cluster means are stored in the attribute ``self.clusters`` and the index to which each cluster belongs is
        stored in the attribute ``self.clustered``.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters, 5 by default
        based_on : str, optional
            One can cluster either on the sum of the intensities (pass ````sum````), the log of the sum (pass ````log````) or the thickness (pass ````thickness````).
            The default is ````log````.
        **kwargs : keyword arguments
            additional keyword arguments to pass to :py:meth:`k_means_clustering.k_means() <k_means_clustering.k_means()>`.
        """

        if based_on == "sum":
            values = np.sum(self.data, axis=2).flatten()
        elif based_on == "log":
            values = np.log(np.sum(np.maximum(self.data, 1e-14), axis=2).flatten())
        elif based_on == "thickness":
            values = self.t[:, :, 0].flatten()
        elif type(based_on) == np.ndarray:
            values = based_on.flatten()
            if values.size != (self.image_shape[0] * self.image_shape[1]):
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
        """
        If the image has been clustered before and the the cluster means are already known,
        one can use this function to reconstruct the original clustering of the image.

        Parameters
        ----------
        cluster_values: array_like
            Array with the cluster means

        Notes
        -----
        Works only for images clustered on (log) integrated intensity."""
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
        Plots the summation over the intensity for each pixel in a heatmap.

        Parameters
        ----------
        title: str, optional
            Title of the plot
        xlab: str, optional
            x-label
        ylab: str, optional
            y-label
        """
        # TODO: invert colours
        if self.name is not None:
            name = self.name
        else:
            name = ''
        plt.figure()
        if title is None:
            plt.title("intgrated intensity spectrum " + name)
        else:
            plt.title(title)
        if self.pixelsize is not None:
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

    def plot_heatmap(self, data, title=None, xlab='[\u03BCm]', ylab='[\u03BCm]', cmap='coolwarm', 
                     discrete_colormap=False, sig_cbar=3, color_bin_size=None, equal_axis=True, 
                     sig_ticks=2, npix_xtick=10, npix_ytick=10, scale_ticks=1, tick_int=False, 
                     save_as=False, **kwargs):
        """
        Plots a heatmap for given data input.

        Parameters
        ----------
        data : array
            Input data for heatmap, but be 2 dimensional.
        title : str, optional
            Set the title of the heatmap. The default is None.
        xlab : str, optional
            Set the label of the x-axis. Micron ([\u03BCm]) are assumed as standard scale. The default is '[\u03BCm]'.
        ylab : str, optional
            Set the label of the y-axis. Microns ([\u03BCm]) are assumed as standard scale. The default is '[\u03BCm]'.
        cmap : str, optional
            Set the colormap of the heatmap. The default is 'coolwarm'.
        discrete_colormap : bool, optional
            Enables the heatmap values to be discretised. Best used in conjuction with color_bin_size. The default is False.
        sig_cbar : int, optional
            Set the amount of significant numbers displayed in the colorbar. The default is 3.
        color_bin_size : float, optional
            Set the size of the bins used for discretisation. Best used in conjuction discrete_colormap. The default is None.
        equal_axis : bool, optional
            Enables the pixels to look square or not. The default is True.
        sig_ticks : int, optional
            Set the amount of significant numbers displayed in the ticks. The default is 2.
        npix_xtick : float, optional
            Display a tick per n pixels in the x-axis. Note that this value can be a float. The default is 10.
        npix_ytick : float, optional
            Display a tick per n pixels in the y-axis. Note that this value can be a float. The default is 10.
        scale_ticks : float, optional
            Change the scaling of the numbers displayed in the ticks. Microns ([\u03BCm]) are assumed as standard scale, adjust scaling from there. The default is 1.
        tick_int : bool, optional
            Set whether you only want the ticks to display as integers instead of floats. The default is False.
        save_as : str, optional
            Set the location and name for the heatmap to be saved to. The default is False.
        **kwargs : dictionary
            Additional keyword arguments.

        """
        
        # TODO: invert colours
        
        plt.figure(dpi=200)
        if title is None:
            if self.name is not None:
                plt.title(self.name)
        else:
            plt.title(title)
            
        if 'mask' in kwargs:
            mask = kwargs['mask']
            if mask.all():
                warnings.warn("Mask all True: no values to plot.")
                return
        else:
            mask = np.zeros(data.shape).astype('bool')
            
        if equal_axis:
            plt.axis('scaled')

        if discrete_colormap:

            unique_data_points = np.unique(data[~mask])
            if 'vmax' in kwargs:
                if len(unique_data_points[unique_data_points > kwargs['vmax']]) > 0:
                    unique_data_points = unique_data_points[unique_data_points <= kwargs['vmax']]

            if 'vmin' in kwargs:
                if len(unique_data_points[unique_data_points < kwargs['vmin']]) > 0:
                    unique_data_points = unique_data_points[unique_data_points >= kwargs['vmin']]

            if color_bin_size is None:
                if len(unique_data_points) == 1:
                    color_bin_size = 1
                else:
                    color_bin_size = np.nanpercentile(unique_data_points[1:]-unique_data_points[:-1],30)
                    
            n_colors = round((np.max(unique_data_points) - np.min(unique_data_points))/color_bin_size +1)
            cmap = cm.get_cmap(cmap, n_colors)
            spacing = color_bin_size / 2
            kwargs['vmax'] = np.max(unique_data_points) + spacing
            kwargs['vmin'] = np.min(unique_data_points) - spacing

        if self.pixelsize is not None:
            ax = sns.heatmap(data, cmap=cmap, **kwargs)
            xticks, yticks, xticks_labels, yticks_labels = self.get_ticks(sig_ticks, npix_xtick, npix_ytick, scale_ticks, tick_int)
            ax.xaxis.set_ticks(xticks)
            ax.yaxis.set_ticks(yticks)
            ax.set_xticklabels(xticks_labels, rotation=0)
            ax.set_yticklabels(yticks_labels)
        else:
            ax = sns.heatmap(data, **kwargs)
            
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        
        colorbar = ax.collections[0].colorbar
        if discrete_colormap:    
            if data.dtype == int:
                colorbar.set_ticks(np.unique(data[~mask]))
            else:
                colorbar.set_ticks(np.unique(data[~mask]))
                cbar_ticks_labels = []
                for tick in np.unique(data[~mask]):
                    if tick >= 1:
                        cbar_ticks_labels.append(round_scientific(tick, sig_cbar+len(str(abs(int(math.floor(tick)))))))
                    else:
                        cbar_ticks_labels.append(round_scientific(tick, sig_cbar))
                colorbar.ax.set_yticklabels(cbar_ticks_labels)
    
        if 'vmin' in kwargs:
            if np.nanmin(data[~mask]) < kwargs['vmin']:
                cbar_ticks = colorbar.ax.get_yticklabels()
                loc = 0
                if discrete_colormap:
                    loc = np.min(np.argwhere(colorbar.ax.get_yticks() >= kwargs['vmin'] + spacing))
                cbar_ticks[loc] = r'$\leq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)
                
        if 'vmax' in kwargs:
            if np.nanmax(data[~mask]) > kwargs['vmax']:
                cbar_ticks = colorbar.ax.get_yticklabels()
                cbar_ticks_values = colorbar.ax.get_yticks()
                loc = -1
                if discrete_colormap:
                    loc = np.max(np.argwhere(cbar_ticks_values <= kwargs['vmax'] - spacing))
                cbar_ticks[loc] = r'$\geq$' + cbar_ticks[loc].get_text()
                colorbar.ax.set_yticklabels(cbar_ticks)                

        if save_as:
            if type(save_as) != str:
                if self.name is not None:
                    save_as = self.name
            if 'mask' in kwargs:
                save_as += '_masked'
            save_as += '.pdf'
            plt.savefig(save_as, bbox_inches='tight')
        plt.show()
 
    
    def get_ticks(self, sig_ticks=2, npix_xtick=10, npix_ytick=10, scale_ticks=1, tick_int=False):
        """
        Sets the proper tick labels and tick positions for the heatmap plots.
        
        Parameters
        ----------
        sig_ticks : int, optional
            Set the amount of significant numbers displayed in the ticks. The default is 2.
        npix_xtick : float, optional
            Display a tick per n pixels in the x-axis. Note that this value can be a float. The default is 10.
        npix_ytick : float, optional
            Display a tick per n pixels in the y-axis. Note that this value can be a float. The default is 10.
        scale_ticks : float, optional
            Change the scaling of the numbers displayed in the ticks. Microns ([\u03BCm]) are assumed as standard scale, adjust scaling from there. The default is 1.
        tick_int : bool, optional
            Set whether you only want the ticks to display as integers instead of floats. The default is False.
            
        Returns
        -------
        xticks : array_like
            Array of the xticks positions.
        yticks : array_like
            Array of the yticks positions.
        xticks_labels : array_like
            Array with strings of the xtick labels.
        yticks_labels : array_like
            Array with strings of the ytick labels.
        """
        
        xticks = np.arange(0, self.x_axis.shape[0], npix_xtick)
        yticks = np.arange(0, self.y_axis.shape[0], npix_ytick) 
        if tick_int == True:
            xticks_labels = (xticks * round_scientific(self.pixelsize[1] * scale_ticks, sig_ticks)).astype(int)
            yticks_labels = (yticks * round_scientific(self.pixelsize[0] * scale_ticks, sig_ticks)).astype(int)
        else:
            xticks_labels = trunc(xticks * round_scientific(self.pixelsize[1] * scale_ticks, sig_ticks), sig_ticks)
            yticks_labels = trunc(yticks * round_scientific(self.pixelsize[0] * scale_ticks, sig_ticks), sig_ticks)
            
        return xticks, yticks, xticks_labels, yticks_labels

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
        Method to convert units to their associated SI values.

        Parameters
        ----------

        unit: str,
            unit of which the prefix is requested
        SIunit: str, optional
            The SI unit of the unit
        numeric: bool, optional
            Default is `True`. If `True` the prefix is translated to the numeric value
            (e.g. :math:`10^3` for `k`)


        Returns
        ------
        prefix: str or int
            The character of the prefix or the numeric value of the prefix
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
                    # TODO: is this correct? JTH 01/07
                    prefix = None
                    return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix

        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix in ['u', 'micron']:
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
        if self.name is not None:
            name_str = ", name = " + self.name
        else:
            name_str = ""
        return 'Spectral image: ' + name_str + ", image size:" + str(self.data.shape[0]) + 'x' + \
               str(self.data.shape[1]) + ', deltaE range: [' + str(round(self.deltaE[0], 3)) + ',' + \
               str(round(self.deltaE[-1], 3)) + '], deltadeltaE: ' + str(round(self.ddeltaE, 3))

    def __repr__(self):
        data_str = "data * np.ones(" + str(self.shape) + ")"
        if self.name is not None:
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


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)


def round_to_nearest(value, base=5):
    return base * round(float(value) / base)
