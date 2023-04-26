#!/bin/bash
#SBATCH --job-name=pdl1-eval-5000gen
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

featSelMeth=("mim" "f_reg")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/eval_script_refactored.py  -fs $featSelMeth_temp -local False -path /pdl1_5000genNEW -k_max 100 -pred pdl1_tps -trainSet CHUM -genomics True -num_gen 5000