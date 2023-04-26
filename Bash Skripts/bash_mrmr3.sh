#!/bin/bash
#SBATCH --job-name=mrmr3
#SBATCH --cpus-per-task=6
#SBATCH --time=72:00:00
#SBATCH --partition=batch_72h
#SBATCH --nodelist=ul-val-pr-cpu[01-09]
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"

echo "call python script next"
python ./Predictions/2.1_os_days_Radio_pred_script.py -fs mrmr -ns 3