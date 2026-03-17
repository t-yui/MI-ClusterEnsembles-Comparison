import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import friedmanchisquare
from modules import evaluate_clustering, paired_ttest_between_methods


# parameters
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
tau_values = [0.1, 0.3, 0.5]
scenarios = ["MCAR", "MAR"]
S = 200
metrics_to_use = ["ARI", "NMI", "Silhouette", "Purity"]


def save_scores(list_ARI, list_NMI, list_SS, list_P, outpath):
    res_scores = pd.DataFrame(
        {
            "ARI": list_ARI,
            "NMI": list_NMI,
            "Silhouette": list_SS,
            "Purity": list_P,
        }
    )
    res_scores.to_csv(outpath, index=False)


def append_nan_scores(list_ARI, list_NMI, list_SS, list_P):
    list_ARI.append(np.nan)
    list_NMI.append(np.nan)
    list_SS.append(np.nan)
    list_P.append(np.nan)


def append_scores(list_ARI, list_NMI, list_SS, list_P, data, labels_true, labels_pred):
    labels_pred = np.asarray(labels_pred)
    valid = ~np.isnan(labels_pred)
    if valid.sum() == 0:
        append_nan_scores(list_ARI, list_NMI, list_SS, list_P)
        return
    _data = data.iloc[valid, :]
    scores = evaluate_clustering(_data, labels_true[valid], labels_pred[valid])
    list_ARI.append(scores["Adjusted Rand Index"])
    list_NMI.append(scores["NMI"])
    list_SS.append(scores["Silhouette"])
    list_P.append(scores["Purity"])


def save_paired_ttests(prefix, n, rho, tau, scenario):
    rows = []
    for metric in metrics_to_use:
        score_df = pd.DataFrame(
            {
                "kmeans-full": pd.read_csv(
                    f"../results/res_scores_kmeans_{prefix}_n{n}_rho{rho}.csv"
                )[metric],
                "k-means-CCA": pd.read_csv(
                    f"../results/res_scores_ccakmeanspp_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )[metric],
                "k-pod": pd.read_csv(
                    f"../results/res_scores_kpod_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )[metric],
                "MI-AClu": pd.read_csv(
                    f"../results/res_scores_MICluEnHpp_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )[metric],
                "MI-NMF": pd.read_csv(
                    f"../results/res_scores_MICluEnN_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )[metric],
                "MI-GNMI": pd.read_csv(
                    f"../results/res_scores_MICluEnNMI_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )[metric],
            }
        )
        res = paired_ttest_between_methods(score_df)
        res.insert(0, "metric", metric)
        rows.append(res)
    pd.concat(rows, ignore_index=True).to_csv(
        f"../results/paired_ttest_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv",
        index=False,
    )


def save_panel_friedman(prefix):
    rows = []
    friedman_methods = ["MI-NMF", "MI-GNMI", "MI-AClu"]

    for scenario in scenarios:
        for n in n_values:
            for rho in rho_values:
                for metric in metrics_to_use:
                    panel_blocks = []

                    # 1 panel = fixed (prefix, n, rho, scenario), blocks = (tau, repetition)
                    for tau in tau_values:
                        score_df = pd.DataFrame(
                            {
                                "MI-NMF": pd.read_csv(
                                    f"../results/res_scores_MICluEnN_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                                )[metric],
                                "MI-GNMI": pd.read_csv(
                                    f"../results/res_scores_MICluEnNMI_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                                )[metric],
                                "MI-AClu": pd.read_csv(
                                    f"../results/res_scores_MICluEnHpp_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                                )[metric],
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
                        mean_ranks = tmp.rank(
                            axis=1, ascending=False, method="average"
                        ).mean()
                        best_method = mean_ranks.idxmin()

                    rows.append(
                        {
                            "prefix": prefix,
                            "scenario": scenario,
                            "n": n,
                            "rho": rho,
                            "metric": metric,
                            "n_blocks": len(tmp),
                            "friedman_chi2": stat,
                            "p_value": p_value,
                            "best_method": best_method,
                            "mean_rank_MI_NMF": mean_ranks["MI-NMF"],
                            "mean_rank_MI_GNMI": mean_ranks["MI-GNMI"],
                            "mean_rank_MI_AClu": mean_ranks["MI-AClu"],
                        }
                    )

    pd.DataFrame(rows).to_csv(
        f"../results/friedman_panel_{prefix}.csv",
        index=False,
    )


# k-means for reference
## class-balanced scenarios
for n in n_values:
    for rho in rho_values:
        list_ARI = []
        list_NMI = []
        list_SS = []
        list_P = []
        res_file_name = f"res_kmeans_3c_n{n}_rho{rho}.csv"
        res = pd.read_csv(f"../results/{res_file_name}")
        for s in tqdm(range(S)):
            filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
            data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
            labels_pred = res.values[:, s]
            file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
            labels_true = pd.read_csv(file_name_labels).values[:, 1]
            scores = evaluate_clustering(data, labels_true, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
        res_scores_name = f"res_scores_kmeans_3c_n{n}_rho{rho}.csv"
        res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for n in n_values:
    for rho in rho_values:
        list_ARI = []
        list_NMI = []
        list_SS = []
        list_P = []
        res_file_name = f"res_kmeans_3cib_n{n}_rho{rho}.csv"
        res = pd.read_csv(f"../results/{res_file_name}")
        for s in tqdm(range(S)):
            filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
            data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
            labels_pred = res.values[:, s]
            file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
            labels_true = pd.read_csv(file_name_labels).values[:, 1]
            scores = evaluate_clustering(data, labels_true, labels_pred)
            list_ARI.append(scores["Adjusted Rand Index"])
            list_NMI.append(scores["NMI"])
            list_SS.append(scores["Silhouette"])
            list_P.append(scores["Purity"])
        res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
        res_scores_name = f"res_scores_kmeans_3cib_n{n}_rho{rho}.csv"
        res_scores.to_csv(f"../results/{res_scores_name}")


# complete case analysis
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_ccakmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    _data = data[~np.isnan(labels_pred)]
                    labels_true = labels_true[~np.isnan(labels_pred)]
                    labels_pred = labels_pred[~np.isnan(labels_pred)]
                    if len(labels_true) == 0:
                        pass
                    else:
                        scores = evaluate_clustering(_data, labels_true, labels_pred)
                        list_ARI.append(scores["Adjusted Rand Index"])
                        list_NMI.append(scores["NMI"])
                        list_SS.append(scores["Silhouette"])
                        list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_ccakmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_ccakmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    _data = data[~np.isnan(labels_pred)]
                    labels_true = labels_true[~np.isnan(labels_pred)]
                    labels_pred = labels_pred[~np.isnan(labels_pred)]
                    if len(labels_true) == 0:
                        pass
                    else:
                        scores = evaluate_clustering(_data, labels_true, labels_pred)
                        list_ARI.append(scores["Adjusted Rand Index"])
                        list_NMI.append(scores["NMI"])
                        list_SS.append(scores["Silhouette"])
                        list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_ccakmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


# k-pod
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = f"res_kpod_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(S)):
                    filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s + 1]
                    file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_kpod_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = f"res_kpod_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(S)):
                    filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s + 1]
                    file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_kpod_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


# AClu ensemble
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_MICluEnHpp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnHpp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_MICluEnHpp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnHpp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


# NMF ensemble
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = f"res_MICluEnN_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnN_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_MICluEnN_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnN_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


# GNMI ensemble
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_MICluEnNMI_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3c_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnNMI_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_NMI = []
                list_SS = []
                list_P = []
                res_file_name = (
                    f"res_MICluEnNMI_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res = pd.read_csv(f"../results/{res_file_name}")
                for s in tqdm(range(res.shape[1])):
                    filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
                    data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                    labels_pred = res.values[:, s]
                    file_name_labels = f"../data/labels_3cib_n{n}_rho{rho}_{s}.csv"
                    labels_true = pd.read_csv(file_name_labels).values[:, 1]
                    scores = evaluate_clustering(data, labels_true, labels_pred)
                    list_ARI.append(scores["Adjusted Rand Index"])
                    list_NMI.append(scores["NMI"])
                    list_SS.append(scores["Silhouette"])
                    list_P.append(scores["Purity"])
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI, "Silhouette": list_SS, "Purity": list_P,})
                res_scores_name = (
                    f"res_scores_MICluEnNMI_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                save_paired_ttests("3c", n, rho, tau, scenario)
                save_paired_ttests("3cib", n, rho, tau, scenario)


save_panel_friedman("3c")
save_panel_friedman("3cib")
