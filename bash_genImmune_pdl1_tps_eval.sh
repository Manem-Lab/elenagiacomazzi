#!/bin/bash
#SBATCH --job-name=pdl1-Immune-eval
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --partition=batch_96h
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"
array_i=$SLURM_ARRAY_TASK_ID

featSelMeth=("spearman" "pearson" "mim" "f_reg" "mrmr")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/eval_script_refactored.py  -fs $featSelMeth_temp -local False -path /pdl1_Immune -k_max 50 -pred pdl1_tps -trainSet CHUM -genomics True -num_gen Immune