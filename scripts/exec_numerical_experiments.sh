#!/bin/env bash

# Experiment 1
python3 experiment_1.py

# Experiment 2

## data generation
python3 experiment_2_1_gendata.py

## perform multiple imputation
Rscript experiment_2_2_mi.R

## kpod clustering
Rscript experiment_2_3_kpod.R

## ensemble clustering
python3 experiment_2_4_clustering.py

## evaluation
python3 experiment_2_5_evaluation.py

# plot results
python3 plot.py

# generate Friedman test table
python3 experiment_transform_friedman_tables.py
