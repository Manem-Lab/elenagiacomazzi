#!/bin/bash
#SBATCH --job-name=my-batch-2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --nodelist=ul-val-pr-cpu90
#SBATCH --mem=200g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"

echo "call python script next"
python ./Predictions/2.1_os_days_Radio_pred_script.py