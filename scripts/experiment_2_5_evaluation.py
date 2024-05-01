import numpy as np
import pandas as pd
from tqdm import tqdm
from modules import evaluate_clustering


# parameters
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
tau_values = [0.1, 0.3, 0.5]
scenarios = ["MCAR", "MAR"]
S = 200


# k-means for reference
## class-balanced scenarios
for n in n_values:
    for rho in rho_values:
        list_ARI = []
        list_NMI = []
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
        res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
        res_scores_name = f"res_scores_kmeans_3c_n{n}_rho{rho}.csv"
        res_scores.to_csv(f"../results/{res_scores_name}")

## class-imbalanced scenarios
for n in n_values:
    for rho in rho_values:
        list_ARI = []
        list_NMI = []
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
        res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
        res_scores_name = f"res_scores_kmeans_3cib_n{n}_rho{rho}.csv"
        res_scores.to_csv(f"../results/{res_scores_name}")


# complete case analysis
## class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                list_ARI = []
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
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
                list_SS = []
                list_CHI = []
                list_DBI = []
                list_P = []
                list_NMI = []
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
                res_scores = pd.DataFrame({"ARI": list_ARI, "NMI": list_NMI})
                res_scores_name = (
                    f"res_scores_MICluEnNMI_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res_scores.to_csv(f"../results/{res_scores_name}")
