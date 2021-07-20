from spectral_image import SpectralImage
import sys

bs_rep_num = int(sys.argv[1])

dm4_path = '/data/theorie/abelbk/WS2/area03-eels-SI-aligned.dm4'

#dm4_path = '/data/theorie/jthoeve/EELSfitter/dmfiles/h-ws2_eels-SI_003.dm4'
#path_to_models = '/Users/jaco/Documents/CBL-ML/EELS_KK/output/models'

n_clusters = 5  # number of cluster
n_rep = 5  # number of replicas
n_epochs = 10000  # number of epochs
display_step = 10


im = SpectralImage.load_data(dm4_path)
im.output_path = '/data/theorie/jthoeve/EELSfitter/output/'

#im.cluster(5)
#im.pool(5)
#path_to_models = '/data/theorie/abelbk/bash_train_pyfiles/models/dE_nf-ws2_SI-001/E1_p5'
path_to_models = '/data/theorie/abelbk/bash_train_pyfiles/models/dE_nf-ws2_SI-001/E1_new/'
#im.calc_zlps(30, 30, signal="pooled", path_to_models=path_to_models)
im.calc_zlps_matched(30, 30, select_ZLPs=False, path_to_models=path_to_models)
#im.load_zlp_models(path_to_models=path_to_models, plotting=True)

# im.train_zlp(n_clusters=n_clusters,
#              n_rep=n_rep,
#              n_epochs=n_epochs,
#              bs_rep_num=bs_rep_num,
#              path_to_models=path_to_models,
#              display_step=display_step,
#              plot_de1=True,
#              perc_de1=5)


