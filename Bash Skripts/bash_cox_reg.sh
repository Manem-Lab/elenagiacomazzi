#!/bin/bash
#SBATCH --job-name=cox_reg
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --partition=batch_48h
#SBATCH --nodelist=ul-val-pr-cpu06
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"

echo "call python script next"
python ./Predictions/2.1_os_days_Radio_pred_script.py -fs cox_reg