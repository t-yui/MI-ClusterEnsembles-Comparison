import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as ssd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
)
from scipy.cluster.hierarchy import linkage, fcluster
import json
import os
import subprocess
from modules import (
    stab,
    apply_mi_ensemble_clustering,
    apply_mi_ensemble_clustering_NMF,
    apply_mi_ensemble_clustering_nmi,
)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# parameters
input_dir = "../realdata/pbc"
output_dir = "../realdata/pbc/results"
#k = 3
k = None
candidate_k_values = list(range(2, 11))
seed_num = 1
kpod_script = "realdata_pbc_kpod.R"
figure_formats = ["png", "eps"]

method_order = ["ccakmeanspp", "kpod", "MICluEnHpp", "MICluEnN", "MICluEnNMI"]
method_rename = {
    "ccakmeanspp": "Complete-case",
    "kpod": "k-pod",
    "MICluEnHpp": "MI-AClu",
    "MICluEnN": "MI-NMF",
    "MICluEnNMI": "MI-GNMI",
}

BOXPLOT_STYLE = dict(
    linewidth=4,
    fliersize=10,
    saturation=1,
    boxprops=dict(edgecolor="black"),
    whiskerprops=dict(color="black", linewidth=3),
    capprops=dict(color="black", linewidth=3),
    medianprops=dict(color="black", linewidth=4),
)


def safe_silhouette_score_(data, labels):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return silhouette_score(np.asarray(data), labels)
    except:
        return np.nan


def scale_matrix(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def evaluate_internal_scores(data, labels):
    return {
        "Silhouette": safe_silhouette_score_(data, labels),
    }


def apply_kmeans_clustering_(data, n_clst=3, random_state=1, return_inertia=False):
    km = KMeans(
        n_clusters=n_clst,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    labels = km.fit_predict(data)
    if return_inertia:
        return labels, km.inertia_
    return labels


def detect_elbow_k(summary_df, value_col="Inertia_median"):
    tmp = summary_df.sort_values("k").reset_index(drop=True)

    x = tmp["k"].to_numpy(dtype=float)
    y = tmp[value_col].to_numpy(dtype=float)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    pts = np.column_stack([x_norm, y_norm])

    numerator = np.abs(
        (p2[0] - p1[0]) * (p1[1] - pts[:, 1])
        - (p1[0] - pts[:, 0]) * (p2[1] - p1[1])
    )
    denominator = np.linalg.norm(p2 - p1)
    distances = numerator / denominator

    elbow_idx = int(np.argmax(distances))
    selected_k = int(tmp.loc[elbow_idx, "k"])

    return selected_k, distances


def save_figure(fig, outpath_base):
    for ext in figure_formats:
        fig.savefig(f"{outpath_base}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_k_selection_elbow(summary_df, selected_k, outpath_base):
    fig, ax = plt.subplots(figsize=(9, 6))

    tmp = summary_df.sort_values("k").reset_index(drop=True)

    ax.plot(
        tmp["k"],
        tmp["Inertia_median"],
        marker="o",
        linewidth=3,
        markersize=8,
    )

    y_sel = tmp.loc[tmp["k"] == selected_k, "Inertia_median"].iloc[0]
    ax.axvline(selected_k, linestyle="dashed", color="black", linewidth=2)
    ax.scatter([selected_k], [y_sel], s=120, color="black", zorder=5)

    ax.set_xlabel("Number of clusters, $k$", fontsize=20)
    ax.set_ylabel("Median inertia", fontsize=20)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    #ax.set_title("k selection by elbow method", fontsize=20)

    ax.text(
        selected_k + 0.1,
        y_sel,
        f"selected k = {selected_k}",
        fontsize=14,
        va="bottom",
    )
    save_figure(fig, outpath_base)


def make_agreement_matrix(labels_dict):
    mat = pd.DataFrame(np.nan, index=method_order, columns=method_order)
    for meth in method_order:
        if meth in labels_dict:
            mat.loc[meth, meth] = 1.0
    for meth_1 in method_order:
        for meth_2 in method_order:
            if method_order.index(meth_1) < method_order.index(meth_2):
                left = labels_dict[meth_1].dropna().astype(int)
                right = labels_dict[meth_2].dropna().astype(int)
                common = left.index.intersection(right.index)
                if len(common) == 0:
                    score = np.nan
                else:
                    score = normalized_mutual_info_score(
                        left.loc[common].values,
                        right.loc[common].values,
                        average_method="max",
                    )
                mat.loc[meth_1, meth_2] = score
                mat.loc[meth_2, meth_1] = score
    return mat


def run_kpod_clustering(selected_k):
    selected_k_path = f"{output_dir}/pbc_selected_k.txt"
    with open(selected_k_path, "w") as f:
        f.write(f"{selected_k}\n")
    subprocess.run(["Rscript", kpod_script], check=True)

    labels_df = pd.read_csv(f"{output_dir}/pbc_kpod_labels.csv")
    completed_df = pd.read_csv(f"{output_dir}/pbc_kpod_completed_scaled.csv")

    labels_df["id"] = labels_df["id"].astype(int)
    completed_df["id"] = completed_df["id"].astype(int)

    return labels_df, completed_df


# load data
missing_df = pd.read_csv(f"{input_dir}/pbc_missing_data.csv")
imp_df = pd.read_csv(f"{input_dir}/imp_pbc.csv")
meta_df = pd.read_csv(f"{input_dir}/pbc_meta.csv")

missing_df["id"] = missing_df["id"].astype(int)
imp_df[".id"] = imp_df[".id"].astype(int)
imp_df[".imp"] = imp_df[".imp"].astype(int)
meta_df["id"] = meta_df["id"].astype(int)
feature_cols = missing_df.columns[1:].tolist()
imp_values = sorted(imp_df[".imp"].unique())
ids = (
    imp_df.loc[imp_df[".imp"] == imp_values[0], [".id"]]
    .sort_values(".id")
    .values.ravel()
)

os.makedirs(output_dir, exist_ok=True)


# cluster number selection
selected_k = None
labels_store = {}

if k is None:
    rows_kselect = []

    for imp_no in tqdm(imp_values, desc="k selection"):
        data_tmp = imp_df.loc[imp_df[".imp"] == imp_no, :].sort_values(".id")
        data_tmp = data_tmp.loc[:, feature_cols]
        data_scaled = scale_matrix(data_tmp)

        for k_tmp in candidate_k_values:
            labels_tmp, inertia_tmp = apply_kmeans_clustering_(
                data_scaled,
                n_clst=k_tmp,
                random_state=seed_num + imp_no,
                return_inertia=True,
            )

            rows_kselect.append(
                {
                    "imp": imp_no,
                    "k": k_tmp,
                    "Inertia": inertia_tmp,
                }
            )
            labels_store[(imp_no, k_tmp)] = labels_tmp.copy()

    res_kselect = pd.DataFrame(rows_kselect)
    res_kselect.to_csv(f"{output_dir}/pbc_k_details.csv", index=False)

    res_kselect_summary = (
        res_kselect.groupby("k")
        .agg(
            Inertia_median=("Inertia", "median"),
            Inertia_mean=("Inertia", "mean"),
            Inertia_sd=("Inertia", "std"),
        )
        .reset_index()
    )

    selected_k, elbow_distances = detect_elbow_k(
        res_kselect_summary,
        value_col="Inertia_median",
    )

    res_kselect_summary["elbow_distance"] = elbow_distances
    res_kselect_summary["selected"] = (
        res_kselect_summary["k"] == selected_k
    ).astype(int)

    res_kselect_summary.to_csv(
        f"{output_dir}/pbc_k_selection_summary.csv",
        index=False,
    )

    figures_dir = f"{output_dir}/report/figures"
    os.makedirs(figures_dir, exist_ok=True)

    plot_k_selection_elbow(
        res_kselect_summary,
        selected_k,
        f"{figures_dir}/pbc_k_selection_elbow",
    )

else:
    selected_k = int(k)

print(f"Selected k for PBC = {selected_k}")


# base clustering for the selected k
base_clusterings = []
for imp_no in imp_values:
    if k is None and (imp_no, selected_k) in labels_store:
        labels_tmp = labels_store[(imp_no, selected_k)]
    else:
        data_tmp = imp_df.loc[imp_df[".imp"] == imp_no, :].sort_values(".id")
        data_tmp = data_tmp.loc[:, feature_cols]
        data_scaled = scale_matrix(data_tmp)
        labels_tmp = apply_kmeans_clustering_(
            data_scaled,
            n_clst=selected_k,
            random_state=seed_num + imp_no,
        )
    base_clusterings.append(pd.Series(labels_tmp, index=ids, name=f"imp_{imp_no}"))
base_clusterings = pd.concat(base_clusterings, axis=1)
base_clusterings.index.name = "id"
base_clusterings.reset_index().to_csv(f"{output_dir}/pbc_base_clusterings.csv", index=False)
base_stability = stab(base_clusterings.values.T)


# complete-case analysis
cc_df = missing_df.dropna().copy()
cc_labels = pd.Series(np.nan, index=missing_df["id"].values, name="ccakmeanspp")
if cc_df.shape[0] >= selected_k:
    cc_scaled = scale_matrix(cc_df.loc[:, feature_cols])
    cc_pred = apply_kmeans_clustering_(cc_scaled, n_clst=selected_k, random_state=seed_num)
    cc_labels.loc[cc_df["id"].values] = cc_pred
else:
    cc_scaled = np.empty((0, len(feature_cols)))


# k-pod
kpod_labels_df, kpod_completed_df = run_kpod_clustering(selected_k)
kpod_labels = (
    kpod_labels_df
    .set_index("id")
    .loc[missing_df["id"].values, "kpod"]
    .astype(int)
)
kpod_labels.name = "kpod"
kpod_completed = (
    kpod_completed_df
    .set_index("id")
    .loc[missing_df["id"].values, feature_cols]
    .values
)


# cluster ensemble
print("====== PBC, MI-AClu")
labels_aclu = apply_mi_ensemble_clustering(base_clusterings, n_clst=selected_k)
print("====== PBC, MI-NMF")
labels_nmf = apply_mi_ensemble_clustering_NMF(base_clusterings, n_clst=selected_k, random_state=seed_num)
print("====== PBC, MI-GNMI")
labels_gnmi = apply_mi_ensemble_clustering_nmi(base_clusterings, n_clst=selected_k)

labels_dict = {
    "ccakmeanspp": cc_labels,
    "kpod": kpod_labels,
    "MICluEnHpp": pd.Series(labels_aclu, index=ids, name="MICluEnHpp"),
    "MICluEnN": pd.Series(labels_nmf, index=ids, name="MICluEnN"),
    "MICluEnNMI": pd.Series(labels_gnmi, index=ids, name="MICluEnNMI"),
}
labels_df = pd.concat(labels_dict.values(), axis=1)
labels_df.index.name = "id"
labels_df.reset_index().to_csv(f"{output_dir}/pbc_consensus_labels.csv", index=False)


# internal validation scores
rows_scores = []
if cc_df.shape[0] >= selected_k and cc_labels.dropna().nunique() >= 2:
    cc_valid = cc_labels.dropna().astype(int)
    cc_scores = evaluate_internal_scores(cc_scaled, cc_valid.values)
    rows_scores.append(
        {
            "method": "ccakmeanspp",
            "imp": 0,
            "Silhouette": cc_scores["Silhouette"],
        }
    )

kpod_scores = evaluate_internal_scores(kpod_completed, kpod_labels.values)
rows_scores.append(
    {
        "method": "kpod",
        "imp": 0,
        "Silhouette": kpod_scores["Silhouette"],
    }
)

for methname in ["MICluEnHpp", "MICluEnN", "MICluEnNMI"]:
    labels_tmp = labels_dict[methname].astype(int)
    for imp_no in imp_values:
        data_tmp = imp_df.loc[imp_df[".imp"] == imp_no, :].sort_values(".id")
        data_tmp = data_tmp.loc[:, feature_cols]
        data_scaled = scale_matrix(data_tmp)
        scores_tmp = evaluate_internal_scores(data_scaled, labels_tmp.values)
        rows_scores.append(
            {
                "method": methname,
                "imp": imp_no,
                "Silhouette": scores_tmp["Silhouette"],
            }
        )
res_scores = pd.DataFrame(rows_scores)
res_scores.to_csv(f"{output_dir}/pbc_internal_values.csv", index=False)


# method summary
summary_rows = []
for methname in method_order:
    tmp = res_scores.loc[res_scores["method"] == methname, :].copy()
    labels_tmp = labels_dict[methname]
    summary_rows.append(
        {
            "method": methname,
            "selected_k": selected_k,
            "n_clustered": int(labels_tmp.notna().sum()),
            "coverage": float(labels_tmp.notna().mean()),
            "n_clusters": int(labels_tmp.dropna().nunique()),
            "Silhouette_mean": tmp["Silhouette"].mean(),
            "Silhouette_sd": tmp["Silhouette"].std(),
            "base_stability": base_stability if methname in ["MICluEnHpp", "MICluEnN", "MICluEnNMI"] else np.nan,
            "mean_pairwise_nmi": 1 - base_stability if methname in ["MICluEnHpp", "MICluEnN", "MICluEnNMI"] else np.nan,
        }
    )
method_summary = pd.DataFrame(summary_rows)
method_summary.to_csv(f"{output_dir}/pbc_method_summary.csv", index=False)


# cluster sizes
rows_cluster_sizes = []
for methname in method_order:
    labels_tmp = labels_dict[methname].dropna().astype(int)
    clst_sizes = labels_tmp.value_counts().sort_index()
    for clst in clst_sizes.index:
        rows_cluster_sizes.append(
            {
                "method": methname,
                "cluster": int(clst),
                "n": int(clst_sizes.loc[clst]),
                "prop": float(clst_sizes.loc[clst] / len(labels_tmp)),
            }
        )
res_cluster_sizes = pd.DataFrame(rows_cluster_sizes)
res_cluster_sizes.to_csv(f"{output_dir}/pbc_cluster_sizes.csv", index=False)


# method agreement
agreement_matrix = make_agreement_matrix(labels_dict)
agreement_matrix.to_csv(f"{output_dir}/pbc_method_agreement.csv")


# merged output
res_merged = pd.merge(meta_df, labels_df.reset_index(), on="id", how="left")
res_merged.to_csv(f"{output_dir}/pbc_clusters_with_meta.csv", index=False)

manifest = {
    "dataset": "PBC",
    "k_input": k,
    "selected_k": selected_k,
    "k_selection_mode": "auto" if k is None else "fixed",
    "k_values": candidate_k_values,
    "seed": seed_num,
    "n_rows": int(missing_df.shape[0]),
    "n_features": int(len(feature_cols)),
    "n_imputations": int(len(imp_values)),
    "feature_cols": feature_cols,
}
with open(f"{output_dir}/pbc_run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("Finished: PBC clustering")
print(f"base stability = {base_stability:.4f}")
