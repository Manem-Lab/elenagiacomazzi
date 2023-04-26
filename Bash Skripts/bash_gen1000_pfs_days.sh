#!/bin/bash
#SBATCH --job-name=pfs-1000gen
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

featSelMeth=("spearman" "pearson" "mim" "f_reg" "mrmr")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/0_reg_general.py  -fs $featSelMeth_temp -local False -path /pfs_1000genNEW -k_max 100 -pred pfs_days -trainSet CHUM -genomics True -num_gen 1000