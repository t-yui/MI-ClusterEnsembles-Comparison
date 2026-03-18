import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import friedmanchisquare
from modules import evaluate_clustering


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


def save_friedman_by_balance_rho_tau(scenario):
    rows = []
    ensemble_methods = ["MI-NMF", "MI-GNMI", "MI-AClu"]
    method_filekey = {
        "MI-NMF": "MICluEnN",
        "MI-GNMI": "MICluEnNMI",
        "MI-AClu": "MICluEnHpp",
    }

    for prefix in ["3c", "3cib"]:
        balance_label = "balanced" if prefix == "3c" else "imbalanced"

        for rho in rho_values:
            for tau in tau_values:
                for metric in metrics_to_use:
                    block_list = []

                    # fixed: (balance/imbalance, rho, tau)
                    # blocks: (n, repetition)
                    for n in n_values:
                        score_df = pd.DataFrame(
                            {
                                method: pd.read_csv(
                                    f"../results/res_scores_{filekey}_{prefix}_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                                )[metric]
                                for method, filekey in method_filekey.items()
                            }
                        ).dropna()

                        if len(score_df) == 0:
                            continue

                        score_df = score_df.reset_index(drop=True)
                        score_df["block_id"] = [
                            f"{prefix}_rho{rho}_tau{tau}_{scenario}_n{n}_s{i}"
                            for i in range(len(score_df))
                        ]
                        block_list.append(score_df)

                    if len(block_list) == 0:
                        rows.append(
                            {
                                "balance": balance_label,
                                "rho": rho,
                                "tau": tau,
                                "metric": metric,
                                "n_blocks": 0,
                                "statistic": np.nan,
                                "p_value": np.nan,
                                "mean_rank_MI-NMF": np.nan,
                                "mean_rank_MI-GNMI": np.nan,
                                "mean_rank_MI-AClu": np.nan,
                            }
                        )
                        continue

                    block_df = pd.concat(block_list, ignore_index=True)

                    stat, p_value = friedmanchisquare(
                        block_df["MI-NMF"].values,
                        block_df["MI-GNMI"].values,
                        block_df["MI-AClu"].values,
                    )

                    mean_ranks = (
                        block_df[ensemble_methods]
                        .rank(axis=1, ascending=False, method="average")
                        .mean()
                    )

                    rows.append(
                        {
                            "balance": balance_label,
                            "rho": rho,
                            "tau": tau,
                            "metric": metric,
                            "n_blocks": len(block_df),
                            "statistic": stat,
                            "p_value": p_value,
                            "mean_rank_MI-NMF": mean_ranks["MI-NMF"],
                            "mean_rank_MI-GNMI": mean_ranks["MI-GNMI"],
                            "mean_rank_MI-AClu": mean_ranks["MI-AClu"],
                        }
                    )

    pd.DataFrame(rows).to_csv(
        "../results/friedman_by_balance_rho_tau_{0}.csv".format(scenario),
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
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "Silhouette": list_SS,
                "Purity": list_P,
            }
        )
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
        res_scores = pd.DataFrame(
            {
                "ARI": list_ARI,
                "NMI": list_NMI,
                "Silhouette": list_SS,
                "Purity": list_P,
            }
        )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
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
                res_scores = pd.DataFrame(
                    {
                        "ARI": list_ARI,
                        "NMI": list_NMI,
                        "Silhouette": list_SS,
                        "Purity": list_P,
                    }
                )
                res_scores_name = (
                    f"res_scores_MICluEnNMI_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")


for scenario in scenarios:
    save_friedman_by_balance_rho_tau(scenario)
