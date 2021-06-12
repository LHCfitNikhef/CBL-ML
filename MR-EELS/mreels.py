from scipy.optimize import curve_fit as cv
from logging import error
from os import path
from typing import Tuple
from ncempy import io
import numpy as np
import scipy.fftpack as sfft
from scipy.signal import convolve2d as cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


c = 299792458
hbar = 6.626e-34
electron_mass = 9.109e-31
e_charge = 1.602e-19


def develop(frame):
    """Develops or processes an image so that it can be better processed by get_drift(),
    Applies a hamming filter to the frame and gets convolved by two 3x3 Sobel matrices, the resulting frames get normalised by a sigmoid function centered around the mean and adds the result in a square root.

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy

    Returns
    -------
    developed_frame : ndarray
        The processed frame
    """
    edge_x = np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ])

    edge_y = np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]
    ])

    new_image = np.copy(frame)[:,:] #Copies the given frame into new_image to conserve the original

    a = np.average(new_image)
    b = np.std(new_image)

    developed_frame = np.sqrt(cv2(1/(1+np.exp(-(new_image-a)/b))*np.hamming( new_image.shape[0] ),
                        edge_x)**2 + cv2(1/(1+np.exp(-(new_image-a)/b))
                        *np.hamming( new_image.shape[0] ), edge_y)**2)

    return developed_frame


def get_drift(frame0, frame1):
    """Calculates the drift between frame0 and frame1 using fourier phase correlation.
    https://en.wikipedia.org/wiki/Phase_correlation

    Parameters
    ----------
    frame0 : ndarray
        A slice of the stack for a given energy
    frame1 : ndarray
        A slice of the stack for a given energy

    Returns
    -------
    list
        [ydrift, xdrift] pixel drift in floats calculated from frame0 to frame1
    """

    fr0_win = frame0*np.hanning( frame0.shape )
    fr1_win = frame1*np.hanning( frame1.shape )

    fr0_pws = sfft.fft2( fr0_win )
    fr1_pws = sfft.fft2( fr1_win )

    cross_pws = ( fr0_pws * fr1_pws.conj() ) / np.abs(fr0_pws * fr1_pws.conj())
    cross_cor = np.abs( sfft.ifft2( cross_pws ) )
    cross_cor = sfft.fftshift( cross_cor )

    [py, px] = np.argwhere( cross_cor == cross_cor.max())[0]
    [ydrift, xdrift] = [py - frame0.shape[1]/2, px-frame0.shape[0]/2 ]
    return [ydrift, xdrift]


def shift(frame, yshift, xshift):
    """Shift the given frame by yshift in the y-direction axis 0 assumed, shifts by xshift in the axis 1 direction.
    Function pads the original image with zeros as not to add or misplace information.

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy
    yshift : int
        Number of pixels the frame is to be shifted will be casted to int
    xshift : int
        Number of pixels the frame is to be shifted will be casted to int

    Returns
    -------
    ndarray
        The original frame shifted
    """
    new_frame_shape = ( frame.shape[0]+int(abs(yshift)), frame.shape[1]+int(abs(xshift)))
    new_frame = np.zeros( new_frame_shape )
    new_frame[0:frame.shape[0], 0:frame.shape[1]] = frame

    y_corrected_frame = np.roll( new_frame, int(yshift), axis=0)
    corrected_frame = np.roll( y_corrected_frame, int(xshift), axis=1)

    return corrected_frame[0:frame.shape[0], 0:frame.shape[1]]


def get_true_centres(frame, false_centres, leeway=50):
    """Takes a list of estimated centres of bright spots in a frame and with some leeway around this false centre looks for the brightest pixel which will be designated the true centre

    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a given energy
    false_centres : list
        list of false centres in list format: [ [fy0, fx0], [fy.., fx..], [fyN, fxN] ]
    leeway : int, optional
        Amount of pixels around the false centre in which the true centre will be searched for, by default 50

    Returns
    -------
    list
        list of true centres in zipped list [ ty0, tx0, ty.., tx.., tyN, txN ]
    """
    true_centres = np.array([], dtype=object)
    for i in false_centres:
        part_frame = frame[i[0]-leeway:i[0]+leeway, i[1]-leeway:i[1]+leeway]
        [part_true_y, part_true_x] = np.argwhere(part_frame==part_frame.max())[0]
        [true_y, true_x] = [part_true_y+(i[0]-leeway), part_true_x+(i[1]-leeway)]
        true_centres = np.append(true_centres, (true_y, true_x))
    return true_centres


def get_peak_width(frame: np.ndarray, tr_centre: tuple) -> int:
    """Returns an estimate of the width of a peak located at tr_centre, width is estimated as twice the distance from the centre to a value 1/10th the value of centre.

    Parameters
    ----------
    frame : np.ndarray
        A frame in which the peak lies.
    tr_centre : tuple
        Tuple or Iterable containing y and x coordinate of peak i.e.: (y, x) or [y, x]

    Returns
    -------
    int
        Width of the peak
    """
    trace = frame[tr_centre[0],tr_centre[1]:tr_centre[1]+100]
    width = np.argwhere( trace<(frame[tr_centre[0],tr_centre[1]]/15))[0]
    return width


def get_beam_centre(peak, antipeak):
    """Calculates the centre of the ZLP using the centres of a peak and a corresponding anti-peak

    Parameters
    ----------
    peak : iterable
        Coordinates of peak in an iterable i.e.: (y, x) or [y, x]
    antipeak : iterable
        Coordinates of peak in an iterable i.e.: (y, x) or [y, x]

    Returns
    -------
    tuple
        tuple of coordinates of centre of the zero-loss peak (y, x)
    """
    if peak[0] > antipeak[0]:
        zlp_x = antipeak[0]+(peak[0]-antipeak[0])/2
    else:
        zlp_x = peak[0]+(antipeak[0]-peak[0])/2
    if peak[1] > antipeak[1]:
        zlp_y = antipeak[1]+(peak[1]-antipeak[1])/2
    else:
        zlp_y = peak[1]+(antipeak[1]-peak[1])/2
    return (zlp_y, zlp_x)


def generate_angle_map(frame_dimensions, beam_centre, ccd_pixel_size, camera_distance):
    """Generates a map of angles between centre of zero-loss peak and pixel on ccd

    Parameters
    ----------
    frame_dimensions : tuple
        Dimensions of frame: (ysize, xsize)
    beam_centre : tuple
        Coordinates of the centre of the beam (y, x)
    ccd_pixel_size : float
        Size of the physical pixel on the ccd, only square pixels supported
    camera_distance : float
        Distance between beam_centre pixel and sample in the same units as ccd_pixel_size

    Returns
    -------
    ndarray
        Array of angles that correspond to each pixel in the frame
    """
    map_size_y = (np.arange(frame_dimensions[1]) -int(frame_dimensions[1]/2)
                  +beam_centre[1])*ccd_pixel_size
    map_size_x = (np.arange(frame_dimensions[1]) -int(frame_dimensions[0]/2)
                  +beam_centre[0])*ccd_pixel_size

    map_y, map_x = np.meshgrid(map_size_y, map_size_x)
    radius_map = np.sqrt(map_y**2 + map_x**2)
    angle_map = radius_map /2 /camera_distance

    return angle_map


def momentum_trans_map(angle_map, dE, E0, e_k):
    """Generates a map momenta transfers per pixel on the ccd

    Parameters
    ----------
    angle_map : ndarray
        Output of mreels.generate_angle_map() or an array of angles between centre of zero-loss peak and pixel on ccd
    dE : float
        Energy difference of adjacent energy slices
    E0 : float
        Energy of unscattered electron

    Returns
    -------
    ndarray
        Map of size angle_map that contains the momentum transfer per pixel

    See Also
    -------
    mreels.generate_angle_map :
        Generates a map of angles between centre of zero-loss peak and pixel on ccd
    """
    char_elec_angle = dE /2 /E0
    mom_trans_par = char_elec_angle*e_k
    mom_trans_per = angle_map*e_k
    return np.sqrt(mom_trans_par**2 + mom_trans_per**2)


def radial_integration(r1, frame, radii, r0=0):
    """Performs radial integration of the slice from slice_centre outwards.
    sums all values where the distance of those values is greater than r0 and inbetween r1 and r1-ringsize.
    Parameters
    ----------
    frame : ndarray
        A slice of the stack for a specific energy
    slice_centre : tuple
        Coordinates of the centre of the slice
    r1 : int
        Outermost diameter of integration
    ringsize : int
        Thickness of integration ring
    r0 : int, optional
        Inner limit of integration ring, disregarded if r1-ringsize>r0, by default 0
    Returns
    -------
    value
        value of the sum over the integration area
    """

    integration_area = np.where( radii<r1, frame, 0)
    #integration_area = np.where( radii>(r1-ringsize), integration_area1, 0)

    entries = np.where( radii<r1, 1, 0)
    #entries = np.where( radii>(r1-ringsize), entries1, 0)
    integral = np.sum(integration_area) / np.sum(entries)

    return integral


def line_integration_mom(stack: np.ndarray, radii: np.ndarray, r1: int, ringsize: int) -> np.ndarray:
    """Returns the maximum momentum in stack between two radii

    Parameters
    ----------
    stack : np.ndarray
        Map of total momentum per pixel
    radii : np.ndarray
        Map of total radius per pixel
    r1 : int
        Maximum radius
    ringsize : int
        unused

    Returns
    -------
    np.ndarray
        Maximum momentum in masked stack
    """
    selection_area = np.where( (radii<r1), stack, 0)
    #entries = np.where((radii<r1), 1, 0)
    max_mom = np.max(selection_area)
    return max_mom


def line_integration_int(radius: int, stack: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Integrates along a circle segment from where radii is zero to where radii is radius,
    Uses radii as a mask for selecting the values from stack. Averages EELS spectra at a radius and returns this averaged spectrum

    Parameters
    ----------
    radius : int
        Upper bound of integration
    stack : np.ndarray
        EFTEM stack of which the EELS spectra will be averaged
    radii : np.ndarray
        Map of total radius per pixel

    Returns
    -------
    np.ndarray
        The averaged EELS spectrum
    """
    integration_area = np.where(radii<radius, stack, 0)
    entries = np.where((radii<radius), 1, 0)
    integral = np.sum(integration_area)/np.sum(entries)
    return integral



def get_qeels_data(mr_data_stack: object, r1: int, ringsize: int, preferred_frame: int,
                   forward_peak=None, method='radial',
                   threads=2) -> Tuple[np.ndarray,np.ndarray]:
    """Gets averaged/integrated EELS data per momenta and energy and returns this as a 2D array of size (M,E) with a corresponding 1D array of size (M) containing the corresponding momenta.
    E is the same size as the energy axis of the EFTEM stack.

    Parameters
    ----------
    mr_data_stack : object
        MomentumResolvedDataSTack object, pre-assigned and potentially cleaned with methods
    r1 : int
        Upper bound of integration for 'radial' method, unused in 'line' method
    ringsize : int
        Size of integration ring, allows for pooling spectra in range (r, r+ringsize), smaller ringsizes increase computation time significantely.
    preferred_frame : int
        Index of preferred frame, used for get_centre() method.
    forward_peak : Tuple or List, optional
        Coordinates of the forward peak if using 'line' method, by default None
    method : str, optional
        Method of integration, 'radial' or 'line', by default 'radial'
    threads : int, optional
        Number of processor threads the function is allowed to use, if there is enough ram this cuts compute time by 1/threads, by default 2

    Returns
    -------
    Tuple[np.ndarray,np.ndarray]
        Tuple containing the qeels_map of size (M,E) and the corresponding momentum_axis of size (M)
    """
    #stop counting negative energy values
    (esize, ysize, xsize) = mr_data_stack.stack.shape
    momentum_qaxis = np.array([])

    mr_data_stack.build_axes()
    mom_y, mom_x = np.meshgrid(mr_data_stack.axis1, mr_data_stack.axis2)
    momentum_map = np.sqrt(mom_y**2 + mom_x**2)

    stack_centre = mr_data_stack.get_centre(preferred_frame)

    stack_centre = (stack_centre[1], stack_centre[0])

    offset_y = stack_centre[0]-int(ysize/2)
    offset_x = stack_centre[1]-int(xsize/2)
    y = (np.arange(ysize)-ysize/2)+0.5
    x = (np.arange(xsize)-xsize/2)+0.5
    X, Y = np.meshgrid(x, y)
    radii = np.sqrt( (X-offset_x)**2 + (Y-offset_y)**2 )
    #zeros = np.zeros((mr_data_stack.stack.shape), dtype=np.float32)
    #ones = np.ones((mr_data_stack.stack.shape), dtype=np.float32)
    r0 = 0
    qmap = None

    if method == 'radial':
        if forward_peak is not None:
            print('forward_peak not used in method "radial" use r1 for upper limit')

        iterate = range( r0, r1, ringsize)
        qmap = np.zeros((len(iterate), esize))
        for i in tqdm(iterate):
            momentum_frame_total = radial_integration(i, momentum_map, radii, r0)
            momentum_qaxis = np.append(momentum_qaxis, momentum_frame_total)

        rs = [i for i in range(r0,r1,ringsize)]
        energies = [ei for ei in range(0, len(mr_data_stack.axis0))]

        def part_func(ei):
            args = (mr_data_stack.stack[ei], radii)
            tr = np.zeros(len(rs))
            for r in range(len(rs)):
                tr[r] = radial_integration(rs[r], *args)
            return tr

        with ThreadPoolExecutor(threads) as ex:
            results = list(tqdm(ex.map(part_func, energies),
                                desc="Radial integration", total=len(energies)))

        for i in range(0, len(energies)):
            qmap[:,i] = results[i]


    elif method == 'line':
        if forward_peak == None:
            print('forward_peak required for line integration')

        true_fw_peak = get_true_centres(mr_data_stack.stack[preferred_frame],
                                        ((forward_peak[0],forward_peak[1]),(forward_peak[0],forward_peak[1])), leeway=50)
        peak_width = get_peak_width(mr_data_stack.stack[preferred_frame],
                                    (true_fw_peak[0],true_fw_peak[1]))
        angles = np.arctan2((Y-offset_y),(X-offset_x))
        angle_to_centre = np.arctan2((true_fw_peak[0]-stack_centre[0]),(true_fw_peak[1]-stack_centre[1]))

        r1 = int(np.sqrt( (true_fw_peak[0]-stack_centre[0])**2 + (true_fw_peak[1]-stack_centre[1])**2 )+peak_width)
        small_angle = np.abs(np.arctan2(np.sqrt(2)*peak_width,r1))

        stack = np.where( (angles<(angle_to_centre+small_angle))&(angles>(angle_to_centre-small_angle)), mr_data_stack.stack, 0)

        iterate = range( r0, r1, ringsize)
        qmap = np.zeros((len(iterate), esize))
        for i in tqdm(iterate):
            momentum_frame_total = line_integration_mom(momentum_map, radii, i, ringsize)
            momentum_qaxis = np.append(momentum_qaxis, momentum_frame_total)


        #since our peak is in one quadrant we can cut it from the array and only use that cut.
        #This will in the optimal case reduce memory usage by 75%.

        if true_fw_peak[0] < stack_centre[0]:
            #peak upper-half:
            if true_fw_peak[1] < stack_centre[1]:
                #peak lhs
                radii = radii[0:stack_centre[0], 0:stack_centre[1]]
                stack = stack[:, 0:stack_centre[0], 0:stack_centre[1]]
            else:
                #peak rhs
                radii = radii[0:stack_centre[0], stack_centre[1]:]
                stack = stack[:, 0:stack_centre[0], stack_centre[1]:]
        else:
            #peak lower-half
            if true_fw_peak[1] < stack_centre[1]:
                #peak lhs
                radii = radii[stack_centre[0]:, 0:stack_centre[1]]
                stack = stack[:, stack_centre[0]:, 0:stack_centre[1]]
            else:
                #peak rhs
                radii = radii[stack_centre[0]:, stack_centre[1]:]
                stack = stack[:, stack_centre[0]:, stack_centre[1]:]

        def part_func(e_index):
            args = (stack[e_index], radii)
            to_return = np.zeros(len(rs))
            for r in range(len(rs)):
                to_return[r] = line_integration_int(rs[r], *args)
            return to_return

        rs = [i for i in range(r0,r1,ringsize)]
        energies = [e_index for e_index in range(len(mr_data_stack.axis0))]

        with ThreadPoolExecutor(threads) as ex:
            results = list(tqdm(ex.map(part_func, energies), total=len(energies)))

        for i in range(0, len(energies)):
            qmap[:,i] = results[i]

    return qmap, momentum_qaxis


def get_qeels_slice(data_stack: object, point: tuple,
                    use_k_axis=False, starting_point=None) -> np.ndarray:
    """Get QEELS data by slicing EFTEM stack between two points. Return QEELS data per momentum and energy in 2d array of shape (M,E),
    returns array of corresponding momenta, shape (M)

    Parameters
    ----------
    data_stack : object
        MomentumResolvedDataStack object
    point : tuple
        (y, x) coordinate of point to slice to
    use_k_axis : bool, optional
        Toggle return the k-axis instead of q-axis as momentum data/axis, by default False
    starting_point : Tuple, optional
        (ys, xs) coordinate of starting point to slice from, by default None

    Returns
    -------
    Tuple of ndarray
        Containing QEELS data of shape (M,E) as entry 0 and q/k-axis ndarray of shape (M) as entry 1

    Raises
    ------
    ValueError
        Raises ValueError if using k-axis is toggled on but MomentumResolvedDataStack has no k-axis attribute, use transform_axis() before using k-axis.
    """
    if starting_point == None:
        centre = data_stack.get_centre(data_stack.pref_frame)
    else:
        centre = starting_point


    yp, xp = point
    path_length = int(np.hypot(xp-centre[1], yp-centre[0]))
    xsamp = np.linspace(centre[1], xp, path_length)
    ysamp = np.linspace(centre[0], yp, path_length)
    qmap = data_stack.stack[:,ysamp.astype(int),xsamp.astype(int)].T

    qaxis = np.zeros(int(path_length))
    data_stack.build_axes()


    if use_k_axis == False:
        mom_y, mom_x = np.meshgrid(data_stack.axis1, data_stack.axis2)
        mom_map = np.sqrt(mom_y**2 + mom_x**2)
        qaxis = mom_map[xsamp.astype(int), ysamp.astype(int)]
    else:
        if data_stack.naxis0 == None:
            raise ValueError('The transformed axes are not build, use transform_axis()')
        k_y, k_x = np.meshgrid(data_stack.naxis1, data_stack.naxis2)
        kmap = np.sqrt(k_x**2 + k_y**2)
        qaxis = kmap[xsamp.astype(int), ysamp.astype(int)]


    double_entries = np.asarray([])
    for i in range(0,len(qaxis)-1):
        if qaxis[i] == qaxis[i+1]:
            double_entries = np.append(double_entries, i)

    qaxis_sc = np.asarray([])
    qmap_sc = np.asarray([])
    for i in range(len(qaxis)):
        if i not in double_entries:
            qaxis_sc = np.append(qaxis_sc, qaxis[i])
            qmap_sc = np.append(qmap_sc, qmap[i])
    """ else:
            qm_avg = (qmap[i]+qmap[i+1])/2
            qaxis_sc = np.append(qaxis_sc, qaxis[i])
            qmap_sc = np.append(qmap_sc, qmap[i])
    """
    qmap_sc = qmap_sc.reshape((len(qaxis_sc), qmap.shape[1]))
    return qmap_sc, qaxis_sc


def sigmoid(x: np.ndarray, borders=None) -> np.ndarray:
    """Maps values of x to values between 0..1 by using a scaled logistic function, optimises logistic function for region if borders is set.

    Parameters
    ----------
    x : np.ndarray
        Input image
    borders : Tuple of int, optional
        (lower, upper) index values of region in which to optimise logistic function, by default None

    Returns
    -------
    np.ndarray
        Altered image with values between 0 and 1
    """
    if borders == None:
        avg = np.average(x)
        std = np.std(x)
        im = 1/(1+np.exp(-(x-avg)/std))
    else:
        avg = np.average(x[borders[0]:borders[1], :])
        std = np.std(x[borders[0]:borders[1], :])
        im = 1/(1+np.exp(-(x-avg)/std))
    return im

def plot_qeels_data(mr_data: object, intensity_qmap: np.ndarray,
                    momentum_qaxis: np.ndarray, prefix: str, save=False) -> None:
    """Function to quickly plot qeels data

    Parameters
    ----------
    mr_data : object
        MomentumResolvedDataStack object
    intensity_qmap : np.ndarray
        QEELS map of shape (M,E)
    momentum_qaxis : np.ndarray
        Q/K-axis of shape (M)
    prefix : str
        prefix used for file saving and plot title
    save : bool, optional
        Toggle saving the file or not, by default False
    """
    plt.close()
    mask = np.where( np.isnan(momentum_qaxis) | (momentum_qaxis == 0.0) , False, True)
    qmap = intensity_qmap[mask]
    qaxis = momentum_qaxis[mask]
    min_q = qaxis.min()
    max_q = qaxis.max()

    mr_data.build_axes()
    e = mr_data.axis0

    #Q, E = np.mgrid[min_q:max_q:step_q, min_e:max_e:step_e]
    fig, ax = plt.subplots(1,1)
    c = ax.pcolormesh(e, qaxis, qmap, shading='nearest')
    plt.gcf().set_size_inches((8,8))
    plt.colorbar(c)
    plt.title(prefix)
    ax.set_xlabel(r"Energy [$eV$]")
    ax.set_ylabel(r"$q$ [$nm^{-1}$]")
    if save == True:
        fig.savefig(prefix+'_plot_q[{min_q:.2f}_{max_q:.2f}].pdf'.format(min_q=min_q, max_q=max_q), format='pdf')
    plt.show()


def transform_axis(DataStack: object, Setup: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform axes of MomentumResovledDataStack to \omega, ky and kx and return these.

    Parameters
    ----------
    DataStack : object
        MomentumResolvedDataStack
    Setup : object
        ImagingSetup object

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        \omega axis, ky-axis, kx-axis.
    """
    DataStack.build_axes()
    Setup.angle_on_ccd_axis()
    momentum_y = DataStack.axis1
    momentum_x = DataStack.axis2
    k_y_axis = Setup.y_angles / momentum_y
    k_x_axis = Setup.x_angles / momentum_x
    omega_axis = DataStack.axis0 / 6.626e-34
    DataStack.naxis0 = omega_axis
    DataStack.naxis1 = k_y_axis
    DataStack.naxis2 = k_x_axis
    return omega_axis, k_y_axis, k_x_axis


def angle_map(setup: object) -> np.ndarray:
    """Generates a map of angles corresponding to the pixels on the CCD device.

    Parameters
    ----------
    setup : object
        ImagingSetup object

    Returns
    -------
    np.ndarray
        Map of angles corresponding to pixels on CCD
    """
    pixels_x = np.arange(setup.resolution[1])-setup.resolution[1]/2+0.5
    pixels_y = np.arange(setup.resolution[0])-setup.resolution[0]/2+0.5
    PX, PY = np.meshgrid(pixels_x, pixels_y)
    angles = np.arctan2(PY,PX)
    return angles


def batson_correct(eels_obj: object, energy_window: int, qmap: np.ndarray):
    """Performs Batson correction on the QEELS data and returns the corrected QEELS data.

    Parameters
    ----------
    eels_obj : object
        MomentumResolvedDataStack object
    energy_window : int
        Size of energy windows in units the same as MRDS.axis0
    qmap : np.ndarray
        QEELS data

    Returns
    -------
    np.ndarray
        The batson corrected QEELS data
    """
    eels_obj.build_axes()

    area = eels_obj.axis_1_steps*eels_obj.axis_2_steps
    image_spectrum = np.sum(eels_obj.stack, axis=(2,1))/area
    image_spectrum[image_spectrum <= 0] = 0

    def integrate_window_fp(slice, energy_window):
        slice_max = np.argwhere(slice == slice.max())[0][0]
        half_window_len = int(energy_window /2 /eels_obj.axis_0_scale)
        lower_bound = int(slice_max - half_window_len)
        upper_bound = int(slice_max + half_window_len)
        if lower_bound < eels_obj.axis0.min():
            lower_bound = 0
        if upper_bound > eels_obj.axis0.max():
            upper_bound = len(eels_obj.axis0)
        integral = np.sum(slice[lower_bound:upper_bound])
        return integral

    im_spec_int = integrate_window_fp(image_spectrum, energy_window)
    im_spec_max_index = np.argwhere(image_spectrum == image_spectrum.max())[0][0]
    batson_map = np.copy(qmap)

    for i in range(1, qmap.shape[0]):
        slice = qmap[i]
        slice_int = integrate_window_fp(slice, energy_window)
        norm_im_spec = np.copy(image_spectrum)
        norm_im_spec *= slice_int/im_spec_int
        peakshift = im_spec_max_index - np.argwhere(slice == slice.max())[0][0]
        if peakshift > 0:
            slice[0:qmap.shape[1]-peakshift] -= norm_im_spec[0:qmap.shape[1]-peakshift]
        elif peakshift < 0:
            slice[-peakshift:qmap.shape[1]] -= norm_im_spec[-peakshift:qmap.shape[1]]
        else:
            slice -= norm_im_spec
        batson_map[i] = slice

    return batson_map


def pool_qmap(qmap: np.ndarray, qaxis: np.ndarray, poolsize: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pools QEELS data by addition in the range of (m-1/2*poolsize, m+1/2*poolsize) and pools qaxis by averaging in same range.

    Parameters
    ----------
    qmap : np.ndarray
        QEELS data
    qaxis : np.ndarray
        Q/K-axis
    poolsize : int
        Size of the pooling in bins

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing pooled QEELS data and Q/K-axis
    """
    half_size = poolsize // 2
    bounds = np.asarray([ i for i in range(0, qmap.shape[0], 2*half_size)])
    pooled_qmap = np.zeros((len(bounds)-1, qmap.shape[1]))
    pooled_qaxis = np.zeros(len(bounds)-1)

    for i in range(len(bounds)-1):
        upp = bounds[i+1]
        low = bounds[i]
        pooled_qmap[i] = np.sum(qmap[low:upp,:], axis = 0)
        pooled_qaxis[i] = np.sum(qaxis[low:upp])/2 /half_size

    return pooled_qmap, pooled_qaxis


def find_peak_in_range(qmap: np.ndarray, centre: int, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Tries to find peak in given range determined by windows_size/2 centered around centre, returns index of peak and error estimate.
    error is estimated as energy resolution multiplied by the standard deviation of the multiple peaks.

    Parameters
    ----------
    qmap : np.ndarray
        QEELS data
    centre : int
        Index of the centre of the windows in which to find the peak
    window_size : int
        Size of the window in bins

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        peak position and error in peak position
    """
    half_size = window_size // 2
    search_field = qmap*0
    search_field[:, centre-half_size:centre+half_size] = qmap[:, centre-half_size:centre+half_size]
    search_field[np.isnan(search_field)] = 0

    ppos = np.array([], dtype='int')
    perr = np.array([])
    for i in range(0,len(search_field[:,0])):
        search_slice = search_field[i]
        tmp = np.argwhere(search_slice==search_slice.max())
        perr = np.append(perr, np.std(tmp))
        ppos = np.append(ppos, int(np.average(tmp)))

    return ppos, perr


class MomentumResolvedDataStack:

    def __init__(self, filename: str, pref_frame=0) -> None:

        dm4_file = io.dm.fileDM(filename)

        KEY = '.ImageList.2.ImageData.Calibrations.Dimension.'
        KEY2 = '.ImageList.2.ImageTags.EFTEMSI.Acquisition.'

        self.axis_0_origin = dm4_file.allTags[KEY2+'Start Energy (eV)']
        self.axis_1_origin = dm4_file.allTags[KEY+'2.Origin']
        self.axis_2_origin = dm4_file.allTags[KEY+'1.Origin']

        self.axis_0_scale = dm4_file.allTags[KEY2+'Step Size (eV)']
        self.axis_1_scale = dm4_file.allTags[KEY+'2.Scale']
        self.axis_2_scale = dm4_file.allTags[KEY+'1.Scale']

        self.axis_0_units = dm4_file.allTags[KEY+'3.Units']
        self.axis_1_units = dm4_file.allTags[KEY+'2.Units']
        self.axis_2_units = dm4_file.allTags[KEY+'1.Units']

        self.axis_0_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.3']
        self.axis_1_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.2']
        self.axis_2_steps = dm4_file.allTags['.ImageList.2.ImageData.Dimensions.1']

        self.axis_0_end = dm4_file.allTags[KEY2+'End Energy (eV)']

        self.naxis0 = None
        self.naxis1 = None
        self.naxis2 = None

        self.stack = dm4_file.getDataset(0)['data']
        self.stack_corrected = None

        self.pref_frame = pref_frame

    def rem_neg_el(self):
        if self.pref_frame == None:
            raise ValueError("preferred frame must be set for the building of the axis")
        self.build_axes()
        mask = np.where(self.axis0 < 0, False, True)
        self.axis0 = self.axis0[mask]
        self.axis_0_steps = np.sum(mask)
        self.axis_0_origin = self.axis0.min()
        self.axis_0_end = self.axis0.max()
        self.stack = self.stack[mask,:,:]

    def remove_neg_val(self):
        if self.stack.min() < 0:
            self.stack += self.stack.min()

    def get_centre(self, index: int) -> tuple:
        if index == None:
            index = self.pref_frame

        slice = self.stack[index]
        (y_centre, x_centre) = np.argwhere(slice==slice.max())[0]
        self.centre = (y_centre, x_centre)
        return (y_centre, x_centre)

    def build_axes(self):
        c = self.get_centre(self.pref_frame)
        off_y, off_x = c[0]-self.axis_1_steps/2, c[1]-self.axis_2_steps/2
        self.axis0 = np.linspace(self.axis_0_origin,
                                 self.axis_0_end,
                                 self.axis_0_steps)
        self.axis1 = (np.arange(self.axis_1_steps)-self.axis_1_steps/2-off_y+0.5)*self.axis_1_scale
        self.axis2 = (np.arange(self.axis_2_steps)-self.axis_2_steps/2-off_x+0.5)*self.axis_2_scale

    def correct_drift(self) -> None:
        preferred_frame = self.pref_frame
        stack_copy = np.copy(self.stack)
        stack_avg = np.average(stack_copy)
        stack_std = np.std(stack_copy)
        stack_develop = 1/(1+np.exp(-(stack_copy-stack_avg)/stack_std))

        pref_develop = stack_develop[preferred_frame]
        pref_fft = sfft.fft2(pref_develop)

        stack_shape = pref_develop.shape
        self.stack_corrected = np.zeros(self.stack.shape)

        for i in tqdm(range(0, self.axis_0_steps), desc='Aligning EFTEM stack'):
            current_slice = stack_develop[i]
            cur_slice_fft = sfft.fft2(current_slice)

            cross_pws = (pref_fft*cur_slice_fft.conj())/np.abs(pref_fft*cur_slice_fft.conj())
            cross_cor = np.abs(sfft.ifft2(cross_pws))
            cross_cor = sfft.fftshift(cross_cor)

            if len( np.argwhere( cross_cor == cross_cor.max())) == 0:
                [sy, sx] = [0, 0]
            else:
                [sy, sx] = np.argwhere( cross_cor == cross_cor.max())[0]
                [sy, sx] = [sy - stack_shape[0]/2, sx - stack_shape[1]/2]

            new_frame_shape = ( stack_shape[0]+int(abs(sy)), stack_shape[1]+int(abs(sx)))
            new_frame = np.zeros(new_frame_shape)
            new_frame[0:stack_shape[0], 0:stack_shape[1]] = stack_copy[i]

            y_corrected = np.roll(new_frame, int(sy), axis=0)
            corrected = np.roll(y_corrected, int(sx), axis=1)
            self.stack_corrected[i,:,:] = corrected[0:stack_shape[0], 0:stack_shape[1]]

        self.stack = self.stack_corrected
        self.stack_corrected = None
        stack_copy = None


    def remove_zlp(self, threads=2):
        def function(e, a, b, c):
            return a*np.exp(-(e-b)**2 / c**2)

        def zlp_x(i):
            for j in tqdm(range(len(self.axis2)), desc="removing zlp: "+str(i)+"/"+str(len(self.axis2)), total=2048):
                tmp = self.stack[:,i,j]
                fit_to = tmp[mask]
                try:
                    opt, *_ = cv(function, self.axis0[mask], fit_to)
                    #plt.plot(self.axis0 ,function(self.axis0, *opt)); plt.show()
                    self.stack[:, i, j] - function(self.axis0, *opt)
                except:
                    self.stack[:, i, j]

        mask = self.axis0 <= 3

        with ThreadPoolExecutor(threads) as ex:
            for i in range(len(self.axis1)):
                ex.submit(zlp_x, i)

class ImagingSetup:
    def __init__(self, filename: str ) -> None:

        dm4_file = io.dm.fileDM(filename)

        KEY0 = '.ImageList.2.ImageTags.Acquisition.Device.'
        KEY1 = '.ImageList.2.ImageTags.Microscope Info.'

        self.resolution = dm4_file.allTags[KEY0+'Active Size (pixels)']
        self.pixelsize = dm4_file.allTags[KEY0+'CCD.Pixel Size (um)']*1e-6
        self.voltage = dm4_file.allTags[KEY1+'Voltage']
        self.distance = dm4_file.allTags[KEY1+'STEM Camera Length']*0.1

        planck_constant = 6.626e-34
        electron_mass = 9.109e-31
        elementary_charge = 1.602e-19
        speed_limit = 299792458

        electron_wave_lambda = (planck_constant * speed_limit
                                / np.sqrt( (elementary_charge*self.voltage)**2
                                +2*elementary_charge*self.voltage*electron_mass*speed_limit**2 ) )
        self.e_wavenumber = 2*np.pi / electron_wave_lambda

    def angle_on_ccd_axis(self):
        pixels_x = np.arange(self.resolution[1])-self.resolution[1]
        pixels_y = np.arange(self.resolution[0])-self.resolution[0]
        dist_x = pixels_x * self.pixelsize[1]
        dist_y = pixels_y * self.pixelsize[0]
        angles_x = np.arctan2(dist_x , self.distance)
        angles_y = np.arctan2(dist_y , self.distance)
        self.x_angles = angles_x
        self.y_angles = angles_y