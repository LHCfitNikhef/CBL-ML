#!/bin/bash
pbs_file=/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/bash_train_pyfiles/pbs_training_images.pbs
for replicas in `seq 1 10`; do
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=2 -v ARG=$replicas $pbs_file
done
