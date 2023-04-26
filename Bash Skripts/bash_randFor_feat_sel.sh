#!/bin/bash
#SBATCH --job-name=randFor_feat_sel
#SBATCH --cpus-per-task=6
#SBATCH --time=96:00:00
#SBATCH --partition=batch_96h
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"

echo "call python script next"
python ./Predictions/2.1_os_days_Radio_pred_script.py -fs randFor_feat_sel