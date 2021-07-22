#!/bin/bash


# script to submit jobs to the smefit queue

pbs_file=/data/theorie/jthoeve/EELSfitter/code/train_zlp.pbs
for mc_run in `seq 1 1000`; do
    qsub -q generic -W group_list=theorie -v ARG=$mc_run $pbs_file
done