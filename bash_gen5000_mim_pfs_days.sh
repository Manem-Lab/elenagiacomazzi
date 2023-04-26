#!/bin/bash
#SBATCH --job-name=pfs-5000gen-mim
#SBATCH --array=0
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

featSelMeth=("mim")
featSelMeth_temp=${featSelMeth[$array_i]}


echo "call python script next"
python ./Predictions/0_reg_general.py  -fs $featSelMeth_temp -local False -path /pfs_5000genNEW -k_max 100 -pred pfs_days -trainSet CHUM -genomics True -num_gen 5000