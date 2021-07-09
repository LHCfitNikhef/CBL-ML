from spectral_image import SpectralImage
import sys

bs_rep_num = 0#int(sys.argv[1])

dm4_path = '../dmfiles/h-ws2_eels-SI_003.dm4'
path_to_models = '/Users/jaco/Documents/CBL-ML/EELS_KK/output/models'

n_clusters = 5  # number of cluster
n_rep = 2  # number of replicas
n_epochs = 1000  # number of epochs
display_step = 10

im = SpectralImage.load_data(dm4_path)
im.train_zlp(n_clusters=n_clusters,
             n_rep=n_rep,
             n_epochs=n_epochs,
             bs_rep_num=bs_rep_num,
             path_to_models=path_to_models,
             display_step=display_step)
