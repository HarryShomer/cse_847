#!/bin/bash

########################################################
#
# Run a model on a specific dataset for 10 random seeds
#
# Params:
#   $1: Model Name
#   $2: Dataset Name
#
#########################################################

for seed in {10..100..10}
do
   python src/run.py --model $1 --dataset $2 --seed $seed --run-from-config
done
