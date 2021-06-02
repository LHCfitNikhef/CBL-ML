import mreels
import numpy as np
import matplotlib.pyplot as plt

data_obj = mreels.MomentumResolvedDataStack('n-inse_C1_EFTEM-SI-004 [-3,36] eV.dm4')
qmap = np.load('slice_1_GKGK_qmap_alt.npy')
qaxis = np.load('slice_1_GKGK_qaxis_alt.npy')
#qmap = mreels.sigmoid(qmap)
#mreels.plot_qeels_data(data_obj, qmap, qaxis, '..')

mask = np.where( np.isnan(qaxis) | (qaxis == 0.0) , False, True)
qmap = qmap[mask]
qaxis = qaxis[mask]
min_q = qaxis.min()
max_q = qaxis.max()
step_q = (max_q-min_q)/len(qaxis)

data_obj.build_axes()
e = data_obj.axis0
min_e = e.min()
max_e = e.max()
step_e = (max_e-min_e)/len(e)

#Q, E = np.mgrid[min_q:max_q:step_q, min_e:max_e:step_e]
fig, ax = plt.subplots(1,2)
c = ax[0].pcolormesh(e, qaxis, mreels.sigmoid(np.copy(qmap)), shading='nearest')
plt.gcf().set_size_inches((16,8))
#cbar = fig.colorbar(c, ax=ax[0])
#cbar.set_label('Arbitrary Scale', rotation=90)
plt.title(r"$\Gamma$ M $\Gamma$ M")
ax[0].set_xlabel(r"Energy [$eV$]")
ax[0].set_ylabel(r"$q$ [$nm^{-1}$]")

centre = data_obj.get_centre(25)
my, mx = (1071-471)/2+471, (1023-533)/2+533
disty, distx = my-centre[0], mx-centre[1]

upper = 500
lower = 80

iterate = range(lower,upper,25)

ax[1].set_facecolor('#739686')


for i in iterate:
    color = i/upper
    ax[1].plot(data_obj.axis0, qmap[i], color=str(color))
    ax[1].axhline(np.min(qmap[i]))

plt.savefig('I slicing line plots', format='pdf')
plt.show()