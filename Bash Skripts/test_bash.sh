#!/bin/bash

array_i=3

featSelMeth=("mrmr" "f_classif" "mim_class" "randFor_feat_sel")
featSelMeth_temp=${featSelMeth[$array_i]}
echo $featSelMeth_temp

echo "call python script next"
python ./Predictions/2.2_class_os_days_Radio-pred_script.py -fs $featSelMeth_temp