#!/bin/bash

for i in {11..5009}
do
  j=$((i-9))
  mv /data/theorie/abelbk/bash_train_pyfiles/models/dE_nf-ws2_SI-001/E1_new/nn_rep_$i /data/theorie/abelbk/bash_train_pyfiles/models/dE_nf-ws2_SI-001/E1_new/nn_rep_$j
done