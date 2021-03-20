#!/bin/bash

pbs_file=/data/theorie/jthoeve/cluster_program/EELS_KK/pyfiles/bash_train_pyfiles/pbs_training_images.pbs
for replicas in `seq 1 500`; do
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=1 -v ARG=$replicas $pbs_file
done
