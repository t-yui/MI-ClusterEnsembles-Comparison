# MI-ClusterEnsembles-Comparison 

Repository for: Comparative Study of Cluster Ensemble Algorithms Integrated with Multiple Imputation for Clustering with Missing Data

# How to conduct simulation

## Experiment 1

```bash
$ python3 experiment_1.py
```

## Experiment 2

### data generation

```bash
$ python3 experiment_2_1_gendata.py
```
### perform multiple imputation

```bash
$ Rscript experiment_2_2_mi.R
```

### kpod clustering

```bash
$ Rscript experiment_2_3_kpod.R
```

### ensemble clustering

```bash
$ python3 experiment_2_4_clustering.py
```

### evaluation

```bash
$ python3 experiment_2_5_evaluation.py
```

## plot results

```bash
$ python3 plot.py
```
