#!/bin/bash
#SBATCH --job-name=CI-rad-ecog-eval
#SBATCH --array=0-4
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

featSelMeth=("corr1" "corr2" "mim" "f_reg" "mrmr")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/eval_script_ecog_switch.py  -fs $featSelMeth_temp