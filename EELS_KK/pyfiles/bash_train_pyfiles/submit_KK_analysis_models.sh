#!/bin/bash
pbs_file=/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/bash_train_pyfiles/pbs_KK_analysis_models.pbs
path_models="../../data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/bash_train_pyfiles/dE1/E1_05"
path_save="../../data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/bash_train_pyfiles/dE1/KK_results/"
for replicas in `seq 1 2`; do
    qsub -q smefit -W group_list=smefit -l nodes=1:ppn=2 -v ARG=$path_models,ARG2=$path_save,ARG3=$replicas $pbs_file
done
replicas=0
qsub -q smefit -W group_list=smefit -l nodes=3:ppn=10 -v ARG=$path_models,ARG2=$path_save,ARG3=$replicas $pbs_file

