#!/bin/bash
pbs_file=pbs_training_images.pbs
for replicas in `seq 1 100`; do
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=2 -v ARG=$replicas $pbs_file
done