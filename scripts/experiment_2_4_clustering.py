import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from os import path
from tqdm import tqdm
from logzero import logger
from modules import (
    apply_kmeans_clustering,
    apply_mi_ensemble_clustering_NMF,
    apply_mi_ensemble_clustering_nmi,
    apply_mi_ensemble_clustering,
)


# parameters
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
tau_values = [0.1, 0.3, 0.5]
scenarios = ["MCAR", "MAR"]
S = 200
k = 3


# k-means for reference
## class-balanced scenarios
for n in n_values:
    for rho in rho_values:
        cluster_labels = []
        for s in tqdm(range(S)):
            filepath_not_missing_data = f"../data/3c_n{n}_rho{rho}_{s}.csv"
            data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
            labels = apply_kmeans_clustering(data, n_clst=3, random_state=s + 1)
            cluster_labels.append(labels)
        res = pd.DataFrame(cluster_labels).T
        output_file_name = f"res_kmeans_3c_n{n}_rho{rho}.csv"
        res.to_csv(f"../results/{output_file_name}", index=False)

## class-imbalanced scenarios
for n in n_values:
    for rho in rho_values:
        cluster_labels = []
        for s in tqdm(range(S)):
            filepath_not_missing_data = f"../data/3cib_n{n}_rho{rho}_{s}.csv"
            data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
            labels = apply_kmeans_clustering(data, n_clst=3, random_state=s + 1)
            cluster_labels.append(labels)
        res = pd.DataFrame(cluster_labels).T
        output_file_name = f"res_kmeans_3cib_n{n}_rho{rho}.csv"
        res.to_csv(f"../results/{output_file_name}", index=False)


# complete case analysis
def exec_clustering(filepath):
    X = pd.read_csv(filepath, index_col=0)
    original_index = X.index
    X_complete = X.dropna()
    if len(X_complete) < k:
        labels = np.full(shape=len(original_index), fill_value=np.nan)
    else:
        X_scaled = scale(X_complete)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10)
        kmeans.fit(X_scaled)
        labels = np.full(shape=len(original_index), fill_value=-1)
        labels[original_index.isin(X_complete.index)] = kmeans.labels_
    return labels


## class-balanced scenarios
for scenario in scenarios:
    for rho in rho_values:
        for tau in tau_values:
            for n in n_values:
                scenario_results = []
                for s in tqdm(range(S), desc=f"n{n}_rho{rho}_tau{tau}"):
                    filepath = f"../data/3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                    labels = exec_clustering(filepath)
                    if labels is not None:
                        scenario_results.append(labels)
                if scenario_results:
                    df_results = pd.DataFrame(scenario_results).T
                    df_results = df_results.replace(-1, np.nan)
                    output_path = f"../results/res_ccakmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                    df_results.to_csv(output_path, index=False)

## class-imbalanced scenarios
for scenario in scenarios:
    for rho in rho_values:
        for tau in tau_values:
            for n in n_values:
                scenario_results = []
                for s in tqdm(range(S), desc=f"n{n}_rho{rho}_tau{tau}"):
                    filepath = f"../data/3cib_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                    labels = exec_clustering(filepath)
                    if labels is not None:
                        scenario_results.append(labels)
                if scenario_results:
                    df_results = pd.DataFrame(scenario_results).T
                    df_results = df_results.replace(-1, np.nan)
                    output_path = f"../results/res_ccakmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                    df_results.to_csv(output_path, index=False)


# cluster ensemble
## base clustering
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                for s in tqdm(range(S)):
                    output_file_name = (
                        f"res_baseKmeans_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                    )
                    outpath = f"../results/{output_file_name}"
                    if os.path.exists(outpath):
                        pass
                    else:
                        filepath_not_missing_data = f"../data_mi/imp_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        data = pd.read_csv(filepath_not_missing_data).iloc[:, 1:]
                        labels = apply_each_clustering(data, cls_times=30, n_clst=3)
                        res = pd.DataFrame(labels).T
                        res.to_csv(outpath, index=False)

## AClu ensemble
### class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3c, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', AClu balanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = f"../results/{res_file_name}"
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnHpp_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

### class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3cib, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', AClu imbalanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = (
                            f"../results_audigier_base_clustering/{res_file_name}"
                        )
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering_nmi(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnHpp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

## NMF ensemble
### class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3c, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', NMF balanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = f"../results/{res_file_name}"
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering_NMF(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnN_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

### class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3cib, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', NMF imbalanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = f"../results/{res_file_name}"
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering_NMF(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnN_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

## GNMI ensemble
### class-balanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3c, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', GNMI balanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = f"../results/{res_file_name}"
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering_nmi(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnNMI_3c_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

### class-imbalanced scenarios
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                print(
                    f"====== 3cib, n={n}; rho={rho}; tau={tau}, scenario='{scenario}', GNMI imbalanced"
                )
                cluster_labels = []
                for s in tqdm(range(S)):
                    try:
                        res_file_name = f"res_baseKmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                        res_path = f"../results/{res_file_name}"
                        res_bc = pd.read_csv(res_path)
                        labels = apply_mi_ensemble_clustering_nmi(res_bc, n_clst=3)
                        cluster_labels.append(labels)
                    except:
                        pass
                res = pd.DataFrame(cluster_labels).T
                output_file_name = (
                    f"res_MICluEnNMI_3cib_n{n}_rho{rho}_tau{tau}_{scenario}.csv"
                )
                res.to_csv(f"../results/{output_file_name}", index=False)

# instability
## class-balanced scenarios
stability_dict = {}
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                stability_list = []
                output_file_name = f"res_stability_3c.csv"
                outpath = f"../results/{output_file_name}"
                for s in tqdm(range(S)):
                    base_file_name = (
                        f"res_baseKmeanspp_3c_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                    )
                    filepath = f"../results/{base_file_name}"
                    res = pd.read_csv(filepath)
                    labels_list = res.values.T
                    stability_list.append(stab(labels_list))
                stability_dict[f"n{n}_rho{rho}_tau{tau}_{scenario}"] = np.mean(
                    stability_list
                )
pd.Series(stability_dict).to_csv(outpath)

## class-imbalanced scenarios
stability_dict = {}
for scenario in scenarios:
    for n in n_values:
        for rho in rho_values:
            for tau in tau_values:
                stability_list = []
                output_file_name = f"res_stability_3cib.csv"
                outpath = f"../results/{output_file_name}"
                for s in tqdm(range(S)):
                    base_file_name = f"res_baseKmeanspp_3cib_n{n}_rho{rho}_tau{tau}_{scenario}_{s}.csv"
                    filepath = f"../results/{base_file_name}"
                    res = pd.read_csv(filepath)
                    labels_list = res.values.T
                    stability_list.append(stab(labels_list))
                stability_dict[f"n{n}_rho{rho}_tau{tau}_{scenario}"] = np.mean(
                    stability_list
                )
pd.Series(stability_dict).to_csv(outpath)
