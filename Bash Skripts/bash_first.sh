#!/bin/bash
#SBATCH --job-name=my-batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --nodelist=ul-val-pr-cpu90
#SBATCH --mem=0
#SBATCH --output=%x-%j.out


echo "Load module"

# load module
module load StdEnv/2020 python/3.8 scipy-stack/2022a
# Create virtual env
virtualenv --no-download ~/venvs/venv1
source ~/venvs/venv1/bin/activate
echo "venv activated"

pip install --no-index --upgrade pip
pip install --no-index scikit_learn lifelines boto3 s3fs sqlalchemy psycopg2 pgpasslib xlrd openpyxl
pip install pymrmre

echo "call python script next"
python Predictions/2.1_os_days_Radio_pred_script.py
