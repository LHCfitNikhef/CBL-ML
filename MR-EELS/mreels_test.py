import mreels
import matplotlib.pyplot as plt


if __name__ == '__main__':
    #Creating a eels/eftem data cube object
    eels_stack = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4', 25)

    #Scrubbing the negative energy values
    eels_stack.rem_neg_el()

    #Correcting possible drift from the sample using a fourier phase correlation method
    eels_stack.correct_drift()

    #Calculating a q-eels map by using an integration method, im going to try alternative methods soon™
    qmap, qaxis = mreels.get_qeels_data(eels_stack, 1000, 2, 25, (172,882), 'line', threads=4)

    #plotting the q-EELS map, and saving it as a pdf
    mreels.plot_qeels_data(eels_stack, qmap, qaxis, "1px_step_2brzones_")
