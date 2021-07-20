#!/bin/bash
source /data/theorie/jthoeve/miniconda3/etc/profile.d/conda.sh
conda activate eels_kk

pbs_file=/data/theorie/jthoeve/EELSfitter/code/train_zlp.pbs
for mc_run in `seq 1 2`; do
    qsub -q short -W group_list=theorie -v ARG=$mc_run $pbs_file
done