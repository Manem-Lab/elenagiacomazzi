#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --partition=batch
#SBATCH --nodelist=ul-val-pr-cpu[03-05]
#SBATCH --mem=4g
#SBATCH --output=%x-%j.out


echo "Restoring modules"
module restore mymodules-venv1
# reactivate virtual env
source ~/venvs/venv1/bin/activate
echo "venv activated"

echo "call python script next"
python ./Predictions/3.1_os_days_Radio_eval_script.py 