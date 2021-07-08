import numpy as np
import random
import os
import scipy
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 22})
rc('text', usetex=True)

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
    Rescale the training data to lie between 0.1 and 0.9

    Parameters
    ----------
    inp: array_like
        training data to be rescaled, e.g. dE
    ab: array_like
        scaling parameters, which can be find with `find_scale_var`.
    Returns
    -------
    Rescaled training data
    """
    return inp*ab[0] + ab[1]

def find_scale_var(inp, min_out = 0.1, max_out=0.9):
    """
    Computes the scaling parameters needed to rescale the training data to lie between `min_out` and `max_out`.

    Parameters
    ----------
    inp: array_like
        training data to be rescaled
    min_out: float
        lower limit. Set to 0.1 by default.
    max_out: float
        upper limit. Set to 0.9 by default

    Returns
    -------
    list of rescaling parameters `[a, b]`
    """
    a = (max_out - min_out)/(inp.max() - inp.min())
    b = min_out - a*inp.min()
    return [a, b]


def weight_reset(m):
    """
    Reset the weights and biases associated with the model ``m``.

    Parameters
    ----------
    m: MLP
        Model of type :py:meth:`MLP <MLP>`.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def loss_fn(output, target, error):
    loss = torch.mean(torch.square((output - target)/error))
    return loss


def MC_reps(data_avg, data_std, n_rep):
    n_full = len(data_avg)
    full_y_reps = np.zeros(shape=(n_full, n_rep))
    for i in range(n_rep):
        full_rep = np.random.normal(0, data_std)
        full_y_reps[:,i] = (data_avg + full_rep).reshape(n_full)
    return full_y_reps


def binned_statistics(x,y, nbins, stats = None):
    """Find the mean, variance and number of counts within the bins described by ewd"""
    if stats is None:
        stats = []
        edges = None
    
    x = np.tile(x, len(y))
    y = y.flatten()
        
    
    
    #df_train, 
    cuts1, cuts2 = ewd(x,nbins)
    result = []
    if "mean" in stats:
        mean, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='mean', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='mean', bins=cuts2)
        result.append(mean)
    if "var" in stats:
        #var, edges, binnum = scipy.stats.binned_statistic(x,y, statistic='std', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='std', bins=cuts2)
        low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
        high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)            
        var = high-low
        result.append(var)
    if "count" in stats:
        count, edges, binnum = scipy.stats.binned_statistic(x,y,statistic='count', bins=cuts2)#df_train[:,0], df_train[:,1], statistic='count', bins=cuts2)
        result.append(count)
    if "low" in stats:
        low, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_low, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
        result.append(low)
    if "high" in stats:
        high, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=CI_high, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)
        result.append(high)
    if "mean2" in stats:
        mean2, edges, binnum = scipy.stats.binned_statistic(x,y,statistic=get_mean, bins=cuts2)#df_train[:,0], df_train[:,1], statistic=get_mean, bins=cuts2)
        result.append(mean2)
    
    return result, edges

def split_test_train(data, test_size=0.2):
    #TODO: to use if we do not use single complete spectra
    n_test = round(test_size*data.shape[1])
    train, test = torch.utils.data.random_split(data, [data.shape[1]-n_test, n_test])
    return train
    pass


def smooth(data, window_len=10,window='hanning', keep_original = False):
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
    s=np.r_['-1', data[:,window_len-1:0:-1],data,data[:,-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    
    #y=np.convolve(w/w.sum(),s,mode='valid')
    surplus_data = int((window_len-1)*0.5)
    return np.apply_along_axis(lambda m: np.convolve(m, w/w.sum(), mode='valid'), axis=1, arr=s)[:,surplus_data:-surplus_data]


def smooth_clusters(image, clusters, window_len = None):
    smoothed_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
        smoothed_clusters[i] = smooth(clusters[i])
    return smoothed_clusters

def derivative_clusters(image, clusters):
    dx = image.ddeltaE
    der_clusters = np.zeros((len(clusters)), dtype = object)
    for i in range(len(clusters)):
        der_clusters[i] = (clusters[i][:,1:]-clusters[i][:,:-1])/dx
    return der_clusters


def find_min_de1(image, dy_dx, y_smooth):
    """
    Finds the minimum o
    Parameters
    ----------
    image
    dy_dx
    y_smooth

    Returns
    -------

    """
    crossing = (dy_dx > 0)
    if not crossing.any():
        print("shouldn't get here")
        up = np.argmin(np.absolute(dy_dx)[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
    else:
        up = np.argmax(crossing[np.argmax(y_smooth)+1:]) + np.argmax(y_smooth) +1
    pos_der = image.deltaE[up]
    return pos_der


def plot_dE1(image, y_smooth_clusters, dy_dx_clusters, min_clusters, de1_prob, de1_shift):
    """
    Produces two plots of the locations of `dE1`:

    - The slope of the EELS spectrum for each cluster plus uncertainties.
    - The log EELS intensity per cluster plus uncertainties.

    Parameters.
    ----------
    image: SpectralImage
    y_smooth_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the smoothed spectrum at each
        pixel within the cluster.
    dy_dx_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the slope of the spectrum at each
        pixel within the cluster.
    min_clusters: array_like
        Location of first local minimum for each cluster
    dE1_prob: array_like
        Values of dE1 as determined from the 16% replica rule
    dE1_shift: array_like
        Values of dE1 as determined from the shifted first local minimum rule
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    der_deltaE = image.deltaE[:-1]

    # plot with location of dE1 shown on top of the slope of the raw (smoothened) spectrum
    fig, ax = plt.subplots(figsize=(1.1 * 10, 1.1 * 6))
    for i in range(len(y_smooth_clusters)):

        ci_low = np.nanpercentile(dy_dx_clusters[i], 16, axis=0)
        ci_high = np.nanpercentile(dy_dx_clusters[i], 84, axis=0)
        if i == 0:
            lab = r'$\rm{Vacuum}$'
        else:
            lab = r'$\rm{cluster\;%s}$' % i
        plt.fill_between(der_deltaE, ci_low, ci_high, color=colors[i], alpha=0.2, label=lab)
        plt.vlines(de1_shift[i], -3E3, 2E3, ls='dashdot', color=colors[i])
        plt.vlines(de1_prob[i], -3E3, 2E3, ls='dotted', color=colors[i])

    plt.plot([der_deltaE[0], der_deltaE[-1]], [0, 0], color='black')
    plt.title(r"$\rm{Slope\;of\;EELS\;spectrum\;per\;cluster}$")
    plt.xlabel(r"$\rm{Energy\;loss}\;$" + r"$\Delta E\;$" + r"$\rm{[eV]}$")
    plt.ylabel(r"$dI/d\Delta E$")
    plt.legend(loc='lower right', frameon=False, fontsize=15)
    plt.xlim(np.min(min_clusters) / 4, np.max(min_clusters) * 2)
    plt.ylim(-3e3, 2e3)
    plt.show()

    # plot with location of dE1 shown on top of raw (smoothened) spectrum
    fig, ax = plt.subplots(figsize=(1.1 * 10, 1.1 * 6))
    for i in range(len(y_smooth_clusters)):

        # dx_dy_i_std = np.std(dy_dx_clusters[i], axis = 0)
        ci_low = np.nanpercentile(y_smooth_clusters[i], 16, axis=0)
        ci_high = np.nanpercentile(y_smooth_clusters[i], 84, axis=0)
        plt.fill_between(image.deltaE, ci_low, ci_high, color=colors[i], alpha=0.2)
        plt.vlines(de1_shift[i], 0, 3e4, ls='dotted', color=colors[i])
        plt.vlines(de1_prob[i], 0, 3e4, ls='dashdot', color=colors[i])
        if i == 0:
            lab = r'$\rm{Vacuum}$'
        else:
            lab = r'$\rm{cluster\;%s}$' % i
        plt.plot(image.deltaE, np.average(y_smooth_clusters[i], axis=0), color=colors[i], label=lab)
    plt.plot([der_deltaE[0], der_deltaE[-1]], [0, 0], color='black')
    plt.title(r"$\rm{Position\;of\;}$" + r"$\Delta E_I\;$" + r"$\rm{per\;cluster}$")
    plt.xlabel(r"$\rm{Energy\;loss}\;$" + r"$\Delta E\;$" + r"$\rm{[eV]}$")
    plt.ylabel(r"$\log I_{\rm{EELS}}$")
    plt.legend(loc='upper right', frameon=False, fontsize=15)
    plt.xlim(np.min(min_clusters) / 4, np.max(min_clusters) * 2)
    plt.ylim(1e2, 3e4)
    plt.xlim(0.2, 4.0)
    plt.yscale('log')
    plt.show()


def determine_de1(image, dy_dx_clusters, y_smooth_clusters, shift_de1=0.7):
    """
    Computes the hyperparamter :math:`\Delta E_I` for every cluster in two ways:

    - Take a certain fraction of the location of the first local minimum. The fraction is tuned by ``shift_de1``.
    - Take :math:`\Delta E_I` such that only 16% of the replicas have a positive slope.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object.
    dy_dx_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the slope of the spectrum at each
        pixel within the cluster.
    y_smooth_clusters: array_like
        An array that contains an array for each cluster, which subsequently contains the smoothed spectrum at each
        pixel within the cluster.
    shift_de1: float, optional
        Shift the location of :math:`\Delta E_I` by a factor of ``shift_dE1`` w.r.t. to the first local minimum.

    Returns
    -------
    dE1_clusters: array_like
        Array with the value of :math:`\Delta E_I` for each cluster.
    """

    # number of clusters
    n_clusters = len(y_smooth_clusters)

    # median EELS spectrum for each cluster
    y_smooth_clusters_avg = np.array([np.average(y_smooth_clusters[i], axis=0) for i in range(n_clusters)])

    # median slope of the EElS spectrum for each cluster
    dy_dx_avg = np.array([np.median(dy_dx_clusters[i], axis=0) for i in range(n_clusters)])

    # zeros of the first derivative for each cluster
    min_clusters = np.array([find_min_de1(image, dy_dx_avg[i], y_smooth_clusters_avg[i]) for i in range(n_clusters)])

    # find the replica such that 16% of the replicas have a steeper slope
    dy_dx_16_perc = np.array([np.nanpercentile(dy_dx_clusters[i], 84, axis=0) for i in range(n_clusters)])

    # find the index at which 16% of the replicas have a positive slope
    # and evaluate the corresponding value of deltaE.
    idx = np.argwhere(np.diff(np.sign(dy_dx_16_perc)))[:, 1][1::2]
    de1_prob = image.deltaE[idx]

    # find dE1 by taking a certain fraction of the location of the first local minimum.
    # The fraction is controlled by shift_dE1
    de1_shift = np.array([min_clusters[i] * shift_de1 for i in range(n_clusters)])

    # number of replicas per cluster
    n_rep_cluster = np.array([dy_dx_clusters[i].shape[0] for i in range(n_clusters)])

    # display the computed values of dE1 (both methods) together with the raw EELS spectrum
    plot_dE1(image, y_smooth_clusters, dy_dx_clusters, min_clusters, de1_prob, de1_shift)

    # return de1_prob, replace by de1_shift it this method is prefered
    return de1_prob


def train_zlp_scaled(image, spectra, n_rep=500, n_epochs=30000, lr=1e-3, shift_dE1=0.7, shift_dE2 = 3, path_to_models="models",
                     display_step=1000, bs_rep_num=0):
    """
    Trains the ZLP on the (clustered) spectral image.

    One should train on two input features: the (scaled) log intensity :math:`\log I_{\mathrm{EELS}}` and the (scaled) :math:`\Delta E`.

    Parameters
    ----------
    image: SpectralImage
        :py:meth:`spectral_image.SpectralImage <spectral_image.SpectralImage>` object.
    spectra: array_like
        An array of size ``n_clusters``, with each entry being a 2D array that contains all the spectra within the cluster.
    n_rep: int, optional
        Number of replicas to train on. Each replica consists of one spectrum from each cluster. Set to 500 by default.
    n_epochs: int, optional
        Number of epochs. Set to 30000 by default
    lr: float, optional
        Learning rate. Set to :math:`10^{-3}` by default.
    shift_dE1: float, optional
        Set to 0.7 by default. When given, :math:`\Delta E_I` is located at ``shift_dE1`` times the first local minimum in the spectrum.
    path_to_models: str, optional
        Path to where the trained models should be stored.
    display_step: int, optional
        Number of epochs after which to print output.
    bs_rep_num: int, optional
        For cluster users only: labels the parallel run.

    Examples
    --------
    To train the ZLP, one can run the following lines of code:

    >>> dm4_path = 'path to dm4 file'
    >>> im = SpectralImage.load_data(dm4_path)
    >>> training_data = im.get_cluster_spectra(conf_interval=1, signal='EELS')
    >>> train_zlp_scaled(im, training_data, n_clusters=5, n_rep=500, n_epochs=1000, path_to_models='path', display_step=10)
    """

    if display_step is None:
        print_progress = False
        display_step = 1E6
    else:
        print_progress = True

    if not os.path.exists(path_to_models):
        os.mkdir(path_to_models)
    
    num_saving_per_rep = 50
    saving_step = int(n_epochs/num_saving_per_rep)

    # set all intensities smaller than 1 to 1
    for i in range(len(spectra)):
        spectra[i][spectra[i] < 1] = 1
    
    loss_test_reps = np.zeros(n_rep)

    sigma_clusters = np.zeros((image.n_clusters, image.l))  # shape = (n_clusters, n dE)
    for cluster in range(image.n_clusters):
        ci_low = np.nanpercentile(np.log(spectra[cluster]), 16, axis= 0)
        ci_high = np.nanpercentile(np.log(spectra[cluster]), 84, axis= 0)
        sigma_clusters[cluster, :] = np.absolute(ci_high-ci_low)

    wl1 = round(image.l/20)
    wl2 = wl1*2

    spectra_smooth = smooth_clusters(image, spectra, wl1)
    dy_dx = derivative_clusters(image, spectra_smooth)
    smooth_dy_dx = smooth_clusters(image, dy_dx, wl2)  # shape = (n_clusters, n_pix per cluster, n dE)

    dE1 = determine_de1(image, smooth_dy_dx, spectra_smooth, shift_dE1)
    dE2 = shift_dE2 * dE1
    
    if print_progress: print("dE1 & dE2:", np.round(dE1,3), dE2)

    # rescale the dE features
    ab_deltaE = find_scale_var(image.deltaE)
    deltaE_scaled = scale(image.deltaE, ab_deltaE)
    
    all_spectra  = np.empty((0,image.l))
    for i in range(len(spectra)):
        all_spectra = np.append(all_spectra, spectra[i], axis=0)
    
    int_log_I = np.log(np.sum(all_spectra, axis=1)).flatten()
    ab_int_log_I = find_scale_var(int_log_I)
    del all_spectra
    
    if not os.path.exists(path_to_models + "scale_var.txt"):
        np.savetxt(path_to_models + "scale_var.txt", ab_int_log_I)
    
    if not os.path.exists(path_to_models+ "dE1.txt"):
        np.savetxt(path_to_models + "dE1.txt", np.vstack((image.clusters, dE1)))
    
    for i in range(n_rep):
        save_idx = i + n_rep*bs_rep_num
        if print_progress: print("Started training on replica number {}".format(i) + ", at time ", dt.datetime.now())
        data = np.empty((0,1))
        data_x = np.empty((0,2))
        data_sigma = np.empty((0,1))
        
        for cluster in range(image.n_clusters):
            n_cluster = len(spectra[cluster]) #  number of spectra in a cluster
            idx = random.randint(0,n_cluster-1)

            select1 = len(image.deltaE[image.deltaE<dE1[cluster]])
            select2 = len(image.deltaE[image.deltaE>dE2[cluster]])
            data = np.append(data, np.log(spectra[cluster][idx][:select1]))
            data = np.append(data, np.zeros(select2))
            
            pseudo_x = np.ones((select1+select2, 2))
            pseudo_x[:select1,0] = deltaE_scaled[:select1]
            pseudo_x[-select2:,0] = deltaE_scaled[-select2:]
            int_log_I_idx_scaled = scale(np.log(np.sum(spectra[cluster][idx])), ab_int_log_I)
            pseudo_x[:,1] = int_log_I_idx_scaled
            
            data_x = np.concatenate((data_x,pseudo_x))
            data_sigma = np.append(data_sigma, sigma_clusters[cluster][:select1])
            data_sigma = np.append(data_sigma, 0.8 * np.ones(select2))

        model = MLP(num_inputs=2, num_outputs=1)
        model.apply(weight_reset)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_x, test_x, train_y, test_y, train_sigma, test_sigma = train_test_split(data_x, data, data_sigma, test_size=0.4)
        
        N_test = len(test_x)
        N_train = len(train_x)
        
        test_x = test_x.reshape(N_test, 2)
        test_y = test_y.reshape(N_test, 1)
        train_x = train_x.reshape(N_train, 2)
        train_y = train_y.reshape(N_train, 1)
        train_sigma = train_sigma.reshape(N_train, 1)
        test_sigma = test_sigma.reshape(N_test, 1)
        
        train_x = torch.from_numpy(train_x)
        train_y = torch.from_numpy(train_y)
        train_sigma = torch.from_numpy(train_sigma)
        test_x = torch.from_numpy(test_x)
        test_y = torch.from_numpy(test_y)
        test_sigma = torch.from_numpy(test_sigma)

        loss_test = np.zeros(n_epochs)
        loss_train_n = np.zeros(n_epochs)
        min_loss_test = 1e6 #big number
        n_stagnant = 0
        n_stagnant_max = 5
        for epoch in range(1, n_epochs + 1):
            model.train()
            output = model(train_x.float())
            loss_train = loss_fn(output, train_y, train_sigma)
            loss_train_n[epoch-1] = loss_train.item()
            
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            
            model.eval()
            with torch.no_grad():
                output_test = model(test_x.float())
                loss_test[epoch-1] = loss_fn(output_test, test_y, test_sigma).item()
                if epoch % display_step == 0 and print_progress:
                    print('Rep {}, Epoch {}, Training loss {}, Testing loss {}'.format(i, epoch, round(loss_train.item(),3), round(loss_test[epoch-1],3)))
                    if round(loss_test[epoch-1],3) >= round(loss_test[epoch-1-display_step],3):
                        n_stagnant += 1
                    else:
                        n_stagnant = 0
                    if n_stagnant >= n_stagnant_max:
                        if print_progress: print("detected stagnant training, breaking")
                        break
                if loss_test[epoch-1] < min_loss_test:
                    loss_test_reps[i] = loss_test[epoch-1]
                    min_loss_test = loss_test_reps[i]
                    min_model = copy.deepcopy(model)
                    #iets met copy.deepcopy(model)
                if epoch % saving_step == 0:
                    torch.save(min_model.state_dict(), path_to_models + "nn_rep" + str(save_idx))
                    with open(path_to_models+ "costs" + str(bs_rep_num) + ".txt", "w") as text_file:
                        text_file.write(str(min_loss_test))
        torch.save(min_model.state_dict(), path_to_models + "nn_rep" + str(save_idx))
        with open(path_to_models+ "costs" + str(bs_rep_num) + ".txt", "w") as text_file:
            text_file.write(str(min_loss_test))
        # np.savetxt(path_to_models+ "costs" + str(bs_rep_num) + ".txt", min_loss_test) # loss_test_reps[:epoch])


def ewd(x, nbins):
    """
    INPUT:
        x:
        y:
        nbins:

    OUTPUT:
        df_train:
        cuts1:
        cuts2:

    Apply Equal Width Discretization (EWD) to x and y data to determine variances
    """
    # TODO: I think everything that was here isn't needed?? since x is already sorted, and a 1D array
    # df_train = np.array(np.c_[x,y])
    cuts1, cuts2 = pd.cut(x, nbins, retbins=True)

    return cuts1, cuts2


def CI_high(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values

    a = np.array(data)
    n = len(a)
    b = np.sort(data)

    highest = np.int((1 - (1 - confidence) / 2) * n)
    high_a = b[highest]

    return high_a


def CI_low(data, confidence=0.68):
    ## remove the lowest and highest 16% of the values

    a = np.array(data)
    n = len(a)
    b = np.sort(data)
    lowest = np.int(((1 - confidence) / 2) * n)
    low_a = b[lowest]

    return low_a


def get_median(x, y, nbins):
    # df_train,
    cuts1, cuts2 = ewd(x, y, nbins)
    median, edges, binnum = scipy.stats.binned_statistic(x, y, statistic='median',
                                                         bins=cuts2)  # df_train[:,0], df_train[:,1], statistic='median', bins=cuts2)
    return median


def get_mean(data):
    return np.mean(data)

