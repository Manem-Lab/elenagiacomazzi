#!/bin/bash
#SBATCH --job-name=switch-rad-mrmr
#SBATCH --array=0
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

featSelMeth=("mrmr")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/2.1.1_switch_os_days_Radio_pred_script.py  -fs $featSelMeth_temp