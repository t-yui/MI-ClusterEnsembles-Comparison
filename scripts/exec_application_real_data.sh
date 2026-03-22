#!/bin/env bash

# PBC Data
## data preparation and multiple imputation
Rscript realdata_pbc_prepare_mi.R

## kpod clustering
Rscript realdata_pbc_kpod.R

## ensemble clustering
python3 realdata_run_pbc_clustering.py

# SUPPORT Data
## data preparation and multiple imputation
Rscript realdata_support_prepare_mi.R

## kpod clustering
Rscript realdata_support_kpod.R

## ensemble clustering
python3 realdata_run_support_clustering.py  

# plot results
python3 realdata_plot.py 
