from ncempy.io import dm
import numpy as np
import matplotlib.pyplot as plt


class Spectral_image():

    def __init__(self, data, deltadeltaE, pixelsize=None, name=None):
        self.data = data
        self.ddeltaE = deltadeltaE
        self.deltaE = self.determine_deltaE()

        if pixelsize is not None:
            self.pixelsize = pixelsize * 1E6

    def determine_deltaE(self):
        data_avg = np.average(self.data, axis=(0, 1))
        ind_max = np.argmax(data_avg)
        self.deltaE = np.linspace(-ind_max * self.ddeltaE, (self.l - ind_max - 1) * self.ddeltaE, self.l)
        return self.deltaE

    def plot_all(self, i, j, same_image=True, normalize=False, legend=False,
                 range_x=None, range_y=None, range_E=None, signal="EELS", log=False):

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

        signal_pixel = self.get_pixel_signal(i, j, signal)
        if normalize:
            signal_pixel /= np.max(np.absolute(signal_pixel))
        if log:
            signal_pixel = np.log(signal_pixel)
            plt.ylabel("log intensity")
        plt.plot(self.deltaE, signal_pixel, label="[" + str(j) + "," + str(i) + "]")
        plt.legend()
        plt.show()

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
        return self.data[i, j, :]

    @property
    def l(self):
        """returns length of spectra, i.e. num energy loss bins"""
        return self.data.shape[2]

    @property
    def image_shape(self):
        """return 2D-shape of spectral image"""
        return self.data.shape[:2]

    @staticmethod
    def get_prefix(unit, SIunit=None, numeric=True):
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
                    return prefix
        else:
            prefix = unit[0]
        if not numeric:
            return prefix

        if prefix == 'p':
            return 1E-12
        if prefix == 'n':
            return 1E-9
        if prefix in ['μ', 'µ', 'u', 'micron']:
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
