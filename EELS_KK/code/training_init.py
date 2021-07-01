from spectral_image import SpectralImage
import sys

bs_rep_num = int(sys.argv[1])

dm4_path = '../dmfiles/h-ws2_eels-SI_003.dm4'
path_to_models = '/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/bash_train_pyfiles/models/train_003'

n_clusters = 5
n_rep = 1
n_epochs = 300000
display_step = 10000

im = SpectralImage.load_data(dm4_path)
im.train_zlp(n_clusters=n_clusters,
             n_rep=1,
             n_epochs=n_epochs,
             bs_rep_num=bs_rep_num,
             path_to_models=path_to_models,
             display_step=display_step)
