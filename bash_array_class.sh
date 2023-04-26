#!/bin/bash
#SBATCH --job-name=classific_all_upd
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=batch_72h
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"
array_i=$SLURM_ARRAY_TASK_ID

featSelMeth=("mrmr" "randFor_feat_sel")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/2.2_class_os_days_Radio_pred_script.py -fs $featSelMeth_temp