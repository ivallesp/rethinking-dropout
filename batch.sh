#!/bin/bash
set -e

SEED=$1

# cd to directory of this script
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Run experiment for cifar100 with all the models
python main.py --model-name Conv6 --dataset-name cifar100 --seed $SEED
python main.py --model-name Conv4 --dataset-name cifar100 --seed $SEED
python main.py --model-name FC --dataset-name cifar100 --seed $SEED

# Run experiment for cifar10 with all the models
python main.py --model-name Conv6 --dataset-name cifar10 --seed $SEED
python main.py --model-name Conv4 --dataset-name cifar10 --seed $SEED
python main.py --model-name FC --dataset-name cifar10 --seed $SEED

# Run experiment for fashion mnist with all the models
python main.py --model-name Conv6 --dataset-name fmnist --seed $SEED
python main.py --model-name Conv4 --dataset-name fmnist --seed $SEED
python main.py --model-name FC --dataset-name fmnist --seed $SEED
