#!/bin/bash


# script to submit jobs to the smefit queue

pbs_file=/data/theorie/jthoeve/EELSfitter/code/train_zlp.pbs
for mc_run in `seq 1 10`; do
    qsub -q smefit -W group_list=smefit -v ARG=$mc_run $pbs_file
done