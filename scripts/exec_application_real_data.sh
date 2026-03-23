#!/bin/env bash

# PBC Data
## data preparation and multiple imputation
Rscript realdata_pbc_prepare_mi.R

## ensemble clustering
python3 realdata_run_pbc_clustering.py

## kpod clustering
Rscript realdata_pbc_kpod.R

# SUPPORT Data
## data preparation and multiple imputation
Rscript realdata_support_prepare_mi.R

## ensemble clustering
python3 realdata_run_support_clustering.py  

## kpod clustering
Rscript realdata_support_kpod.R

# plot results
python3 realdata_plot.py 
