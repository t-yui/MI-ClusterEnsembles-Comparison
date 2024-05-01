import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial import distance
from sklearn import preprocessing
from tqdm import tqdm
from logzero import logger
from modules import (
    ensemble_clustering_nmi_,
    ensemble_clustering_NMF_,
    ensemble_clustering_,
    stab,
    gen_data,
    gen_data_imbalanced,
    evaluate_clustering,
    eval_scores,
    d_V,
)


def setting_of_simulation():
    N = 3  ## number of clusters
    Ks = [10, 20, 40]  ## number of data points per clusters
    DIM = 10  ## data dimension
    STD_DEVs = [1, 1.5, 2, 2.5, 3, 3.5, 4]  ## data standard deviation
    n_sim = 200
    return N, Ks, DIM, STD_DEVs, n_sim


output = True
N, Ks, DIM, STD_DEVs, n_sim = setting_of_simulation()


# instability
## class-balanced scenarios
stab_dict = {}
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}, stab".format(N * K, STD_DEV))
        stab_tmp = eval_scores(
            N, K, DIM, STD_DEV, cls_times=100, imbalance=0, n_clst=3, random_state=1
        )
        stab_dict["n{0}_STDDEV_{1}".format(N * K, STD_DEV)] = stab_tmp
pd.Series(stab_dict).to_csv("../results/stabs_balanced.csv", index=False)

## class-imbalanced scenarios
stab_dict = {}
for K in Ks:
    for STD_DEV in STD_DEVs:
        ogger.info("====== n={0}, STD_DEV:{1}, stab imbalance".format(N * K, STD_DEV))
        stab_tmp = eval_scores(
            N, K, DIM, STD_DEV, cls_times=100, imbalance=1, n_clst=3, random_state=1
        )
        stab_dict["n{0}_STDDEV_{1}".format(N * K, STD_DEV)] = stab_tmp
pd.Series(stab_dict).to_csv("../results/stabs_imbalanced.csv", index=False)


# NMF
## class-balanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}".format(N * K, STD_DEV))
        methname = "NMF"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_NMF_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/balanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )

## class-imbalanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}, imbalanced".format(N * K, STD_DEV))
        methname = "NMF"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_NMF_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/imbalanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )


# AClu
## class-balanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}".format(N * K, STD_DEV))
        methname = "AClu"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/balanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )

## class-imbalanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}, imbalanced".format(N * K, STD_DEV))
        methname = "AClu"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/imbalanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )


# GNMI
## class-balanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}".format(N * K, STD_DEV))
        methname = "GNMI"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_nmi_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/balanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )

## class-imbalanced scenarios
for K in Ks:
    for STD_DEV in STD_DEVs:
        logger.info("====== n={0}, STD_DEV:{1}, imbalanced".format(N * K, STD_DEV))
        methname = "GNMI"
        list_ARI = []
        list_NMI = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_nmi_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/imbalanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )
