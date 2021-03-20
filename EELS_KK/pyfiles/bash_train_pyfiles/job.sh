#!/bin/sh

#PBS -q smefit
#PBS -W group_list=smefit

source ~/../../data/theorie/jthoeve/miniconda3/etc/profile.d/conda.sh
conda activate eels_kk

python ~/../../data/theorie/jthoeve/EELS_KK/pyfiles/training_images.py
