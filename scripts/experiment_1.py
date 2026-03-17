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
from scipy.stats import friedmanchisquare
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
    paired_ttest_between_methods,
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
metrics_to_use = ["ARI", "NMI", "SS", "P"]
methods = ["NMF", "AClu", "GNMI"]


def save_paired_ttests(prefix, n, std_dev):
    rows = []
    for metric in metrics_to_use:
        score_df = pd.DataFrame(
            {
                method: pd.read_csv(
                    f"../results/{prefix}_n{n}_STDDEV_{std_dev}_methname{method}.csv"
                )[metric]
                for method in methods
            }
        )
        res = paired_ttest_between_methods(score_df)
        res.insert(0, "metric", metric)
        rows.append(res)
    pd.concat(rows, ignore_index=True).to_csv(
        f"../results/paired_ttest_{prefix}_n{n}_STDDEV_{std_dev}.csv",
        index=False,
    )


def save_panel_friedman(prefix):
    rows = []
    friedman_methods = ["NMF", "GNMI", "AClu"]

    for K in Ks:
        n = N * K
        for metric in metrics_to_use:
            panel_blocks = []

            # 1 panel = fixed (prefix, n), blocks = (STD_DEV, repetition)
            for STD_DEV in STD_DEVs:
                score_df = pd.DataFrame(
                    {
                        method: pd.read_csv(
                            f"../results/{prefix}_n{n}_STDDEV_{STD_DEV}_methname{method}.csv"
                        )[metric]
                        for method in friedman_methods
                    }
                )
                panel_blocks.append(score_df)

            panel_df = pd.concat(panel_blocks, ignore_index=True)
            tmp = panel_df[friedman_methods].dropna()

            if len(tmp) < 2:
                stat, p_value = np.nan, np.nan
                mean_ranks = pd.Series(np.nan, index=friedman_methods)
                best_method = np.nan
            else:
                stat, p_value = friedmanchisquare(
                    *(tmp[m].values for m in friedman_methods)
                )
                # higher is better -> descending rank, so smaller mean rank is better
                mean_ranks = tmp.rank(axis=1, ascending=False, method="average").mean()
                best_method = mean_ranks.idxmin()

            rows.append(
                {
                    "scenario": prefix,
                    "n": n,
                    "metric": metric,
                    "n_blocks": len(tmp),
                    "friedman_chi2": stat,
                    "p_value": p_value,
                    "best_method": best_method,
                    "mean_rank_NMF": mean_ranks["NMF"],
                    "mean_rank_GNMI": mean_ranks["GNMI"],
                    "mean_rank_AClu": mean_ranks["AClu"],
                }
            )

    pd.DataFrame(rows).to_csv(
        f"../results/friedman_panel_{prefix}.csv",
        index=False,
    )


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
        logger.info("====== n={0}, STD_DEV:{1}, stab imbalance".format(N * K, STD_DEV))
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_NMF_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_NMF_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_nmi_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
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
        list_SS = []
        list_P = []
        for i in tqdm(range(n_sim)):
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=i)
            labels_pred, labels_list = ensemble_clustering_nmi_(
                N, K, DIM, STD_DEV, cls_times=30, n_clst=3, random_state=i
            )
            scores = evaluate_clustering(data, labels, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "SS": list_SS,
                "P": list_P,
            }
        )
        print(res_scores.mean())
        res_scores.to_csv(
            "../results/imbalanced_n{0}_STDDEV_{1}_methname{2}.csv".format(
                N * K, STD_DEV, methname
            ),
            index=False,
        )


for K in Ks:
    for STD_DEV in STD_DEVs:
        save_paired_ttests("balanced", N * K, STD_DEV)
        save_paired_ttests("imbalanced", N * K, STD_DEV)

save_panel_friedman("balanced")
save_panel_friedman("imbalanced")
