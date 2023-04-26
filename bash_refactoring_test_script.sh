#!/bin/bash
#SBATCH --job-name=refactoring-testing
#SBATCH --array=0
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=batch
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
python ./Predictions/0_reg_general.py  -fs $featSelMeth_temp -local False -path /refactoring_test/ -k_max 50 -pred os_days