#!/bin/bash
module load StdEnv/2020 python/3.8 scipy-stack/2022a
module save mymodules-venv1

echo "Create virtual env"
virtualenv --no-download ~/venvs/venv1
source ~/venvs/venv1/bin/activate

pip install --no-index --upgrade pip
pip install --no-index scikit_learn lifelines boto3 s3fs
pip install pymrmre