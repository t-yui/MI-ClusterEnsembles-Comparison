import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"


# parameters
seed_num = 1
figure_formats = ["png", "eps"]
reference_method = "MICluEnHpp"
outcome_cols = ["stage", "status", "time"]
method_order = ["ccakmeanspp", "kpod", "MICluEnNMI", "MICluEnN", "MICluEnHpp"]
panel_method_order = ["ccakmeanspp", "kpod", "MICluEnHpp", "MICluEnN", "MICluEnNMI"]

method_rename = {
    "ccakmeanspp": "Complete-case",
    "kpod": "k-pod",
    "MICluEnHpp": "MI-AClu",
    "MICluEnN": "MI-NMF",
    "MICluEnNMI": "MI-GNMI",
}

exp2_hue_order = [
    "k-means-full",
    "k-means-CCA",
    "k-pod",
    "MI-GNMI",
    "MI-NMF",
    "MI-AClu",
]
exp2_palette = dict(zip(exp2_hue_order, sns.color_palette("Set2", len(exp2_hue_order))))
method_palette = {
    "Complete-case": exp2_palette["k-means-CCA"],
    "k-pod": exp2_palette["k-pod"],
    "MI-GNMI": exp2_palette["MI-GNMI"],
    "MI-NMF": exp2_palette["MI-NMF"],
    "MI-AClu": exp2_palette["MI-AClu"],
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


def scale_matrix(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def compute_lowdim_embedding(data, method="pca", random_state=1):
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(data)
    if method == "tsne":
        perplexity = 30
        return TSNE(
            n_components=2,
            init="random",
            learning_rate="auto",
            random_state=random_state,
            perplexity=perplexity,
        ).fit_transform(data)
    if method == "mds":
        return MDS(
            n_components=2, random_state=random_state, normalized_stress="auto"
        ).fit_transform(data)
    if method == "umap":
        import umap

        return umap.UMAP(
            n_components=2, init="random", random_state=random_state
        ).fit_transform(data)
    raise ValueError("Unsupported embedding method")


def normalize_embedding(embedding):
    y = embedding - embedding.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(y)
    if norm == 0:
        return y
    return y / norm


def geometric_median_matrices(matrices, eps=1e-15, tol=1e-05, max_iter=500):
    z = np.mean(np.stack(matrices, axis=0), axis=0)
    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(x - z, ord="fro") for x in matrices])
        weights = 1.0 / (distances + eps)
        z_new = np.tensordot(weights, np.stack(matrices, axis=0), axes=(0, 0)) / np.sum(
            weights
        )
        if np.linalg.norm(z_new - z, ord="fro") < tol:
            return z_new
        z = z_new
    return z


def classical_mds(distance_matrix, n_components=2):
    n = distance_matrix.shape[0]
    d2 = distance_matrix**2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j.dot(d2).dot(j)
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.maximum(eigvals[idx], 0)
    eigvecs = eigvecs[:, idx]
    return eigvecs * np.sqrt(eigvals)


def median_consensus_embedding(base_embeddings):
    distance_matrices = []
    for emb in base_embeddings:
        emb_norm = normalize_embedding(emb)
        distance_matrices.append(pairwise_distances(emb_norm, metric="euclidean"))
    consensus_x = geometric_median_matrices(distance_matrices)
    consensus_y = classical_mds(consensus_x, n_components=2)
    return consensus_y


def load_manifest(path):
    with open(path, "r") as f:
        return json.load(f)


def align_series_to_reference(reference, target):
    ref = reference.dropna().astype(int)
    tgt = target.dropna().astype(int)
    common = ref.index.intersection(tgt.index)
    if len(common) == 0:
        return target.copy()
    if ref.nunique() <= 1 or tgt.nunique() <= 1:
        return target.copy()

    ref_vals = np.sort(ref.loc[common].unique())
    tgt_vals = np.sort(tgt.loc[common].unique())
    contingency = np.zeros((len(ref_vals), len(tgt_vals)), dtype=int)
    ref_common = ref.loc[common].values
    tgt_common = tgt.loc[common].values
    for i, rv in enumerate(ref_vals):
        for j, tv in enumerate(tgt_vals):
            contingency[i, j] = np.sum((ref_common == rv) & (tgt_common == tv))
    cost = contingency.max() - contingency
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {int(tgt_vals[j]): int(ref_vals[i]) for i, j in zip(row_ind, col_ind)}

    unused_ref = [int(x) for x in ref_vals if int(x) not in mapping.values()]
    next_label = int(max(ref_vals.max(), tgt_vals.max()) + 1)
    for tv in tgt_vals:
        tv = int(tv)
        if tv not in mapping:
            if len(unused_ref) > 0:
                mapping[tv] = unused_ref.pop(0)
            else:
                mapping[tv] = next_label
                next_label += 1

    aligned = target.copy()
    mask = aligned.notna()
    aligned.loc[mask] = aligned.loc[mask].astype(int).map(mapping)
    return aligned


def align_all_labels(labels_dict, reference_method):
    aligned_dict = {}
    if reference_method not in labels_dict:
        reference_method = list(labels_dict.keys())[0]
    reference = labels_dict[reference_method]
    for methname in labels_dict.keys():
        if methname == reference_method:
            aligned_dict[methname] = labels_dict[methname].copy()
        else:
            aligned_dict[methname] = align_series_to_reference(
                reference, labels_dict[methname]
            )
    return aligned_dict


def save_figure(fig, outpath_base):
    for ext in figure_formats:
        fig.savefig(f"{outpath_base}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_missingness(missing_summary, outpath_base):
    tmp = missing_summary.sort_values("missing_prop", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.35 * len(tmp))))
    ax.barh(tmp["variable"], 100.0 * tmp["missing_prop"].values)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missingness (%)", fontsize=18)
    ax.set_ylabel("", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    # ax.set_title(dataset_title, fontsize=20)
    save_figure(fig, outpath_base)


def plot_agreement_heatmap(agreement_matrix, outpath_base):
    disp_labels = [method_rename[x] for x in method_order]
    mat = agreement_matrix.loc[method_order, method_order].values.astype(float)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.grid(False)
    ax.set_xticks(range(len(disp_labels)))
    ax.set_yticks(range(len(disp_labels)))
    ax.set_xticklabels(disp_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticklabels(disp_labels, fontsize=12)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            text = "--" if np.isnan(val) else f"{val:.2f}"
            color = "white" if np.isnan(val) or val < 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NMI")
    # ax.set_title(dataset_title, fontsize=20)
    save_figure(fig, outpath_base)


def plot_silhouette_boxplot(res_scores, outpath_base):
    tmp = res_scores.copy()
    tmp["Method"] = tmp["method"].replace(method_rename)
    order = [method_rename[x] for x in method_order]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="Method",
        y="Silhouette",
        data=tmp,
        order=order,
        palette=method_palette,
        ax=ax,
        **BOXPLOT_STYLE,
    )
    ax.set_xlabel("", fontsize=18)
    ax.set_ylabel("Silhouette score", fontsize=20)
    ax.tick_params(axis="x", labelsize=15, rotation=25)
    ax.tick_params(axis="y", labelsize=16)
    # ax.set_title(dataset_title, fontsize=20)
    save_figure(fig, outpath_base)


def compute_mce_embedding(imp_df, feature_cols):
    base_embeddings = []
    for imp_no in sorted(imp_df[".imp"].unique()):
        data_tmp = imp_df.loc[imp_df[".imp"] == imp_no, :].sort_values(".id")
        data_tmp = data_tmp.loc[:, feature_cols]
        data_scaled = scale_matrix(data_tmp)
        emb = compute_lowdim_embedding(
            data_scaled, method=embedding_method, random_state=seed_num + int(imp_no)
        )
        base_embeddings.append(emb)
    coords = median_consensus_embedding(base_embeddings)
    ids = (
        imp_df.loc[imp_df[".imp"] == sorted(imp_df[".imp"].unique())[0], ".id"]
        .sort_values()
        .values
    )
    return pd.DataFrame({"id": ids, "dim1": coords[:, 0], "dim2": coords[:, 1]})


def plot_mce_embedding_panels(embedding_df, labels_dict, method_summary, outpath_base):
    ncols = 3
    nrows = int(math.ceil(len(panel_method_order) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False
    )
    embedding_df = embedding_df.set_index("id")

    all_clusters = []
    for methname in panel_method_order:
        all_clusters.extend(
            labels_dict[methname].dropna().astype(int).unique().tolist()
        )
    all_clusters = sorted(list(set(all_clusters)))
    cluster_colors = {
        clst: plt.get_cmap("tab10")(i % 10) for i, clst in enumerate(all_clusters)
    }

    for ax, methname in zip(axes.ravel(), panel_method_order):
        labels = labels_dict[methname]
        valid = labels.dropna().astype(int)
        invalid = labels[labels.isna()].index
        if len(invalid) > 0:
            ax.scatter(
                embedding_df.loc[invalid, "dim1"],
                embedding_df.loc[invalid, "dim2"],
                s=12,
                color="lightgray",
                alpha=0.40,
            )
        for clst in sorted(valid.unique()):
            idx = valid.index[valid == clst]
            ax.scatter(
                embedding_df.loc[idx, "dim1"],
                embedding_df.loc[idx, "dim2"],
                s=15,
                color=cluster_colors[clst],
                alpha=0.85,
                label=f"C{clst}",
            )
        selected_k = int(
            method_summary.loc[method_summary["method"] == methname, "selected_k"].iloc[
                0
            ]
        )
        ax.set_title(f"{method_rename[methname]}", fontsize=16)
        ax.set_xlabel("MCE dimension 1", fontsize=14)
        ax.set_ylabel("MCE dimension 2", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

    for ax in axes.ravel()[len(panel_method_order) :]:
        ax.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=cluster_colors[clst],
            label=f"C{clst}",
        )
        for clst in all_clusters
    ]
    if labels_dict["ccakmeanspp"].isna().any():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color="lightgray",
                label="Unclustered",
            )
        )
    target_ax = axes.ravel()[-1]
    target_ax.legend(
        handles=legend_handles, loc="center", ncol=1, fontsize=20, frameon=False
    )
    # fig.suptitle(f"{dataset_title}: median consensus embedding", fontsize=20, y=1.02)
    fig.tight_layout()
    save_figure(fig, outpath_base)


def get_average_cluster_profile(imp_df, feature_cols, labels):
    frames = []
    for imp_no in sorted(imp_df[".imp"].unique()):
        data_tmp = imp_df.loc[imp_df[".imp"] == imp_no, :].sort_values(".id")
        data_tmp = data_tmp.set_index(".id")
        data_scaled = pd.DataFrame(
            scale_matrix(data_tmp.loc[:, feature_cols]),
            index=data_tmp.index,
            columns=feature_cols,
        )
        merged = data_scaled.join(labels.rename("cluster"), how="inner")
        merged = merged.loc[merged["cluster"].notna(), :]
        if merged.shape[0] == 0:
            continue
        profile = merged.groupby("cluster")[feature_cols].mean()
        frames.append(profile)
    if len(frames) == 0:
        return pd.DataFrame(columns=feature_cols)
    profile_mean = (
        pd.concat(frames, keys=range(len(frames)))
        .groupby("cluster")
        .mean()
        .sort_index()
    )
    return profile_mean


def plot_cluster_profile_heatmaps(imp_df, feature_cols, labels_dict, outpath_base):
    profile_dict = {}
    all_values = []
    for methname in panel_method_order:
        profile = get_average_cluster_profile(
            imp_df, feature_cols, labels_dict[methname]
        )
        profile_dict[methname] = profile
        if profile.shape[0] > 0:
            all_values.append(profile.values)

    vmax = 1.0
    if len(all_values) > 0:
        vmax = max(1.0, np.nanmax(np.abs(np.concatenate(all_values))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        len(panel_method_order),
        figsize=(4.5 * len(panel_method_order), 6),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    im = None

    for ax, methname in zip(axes, panel_method_order):
        profile = profile_dict[methname]
        if profile.shape[0] == 0:
            ax.axis("off")
            continue
        im = ax.imshow(profile.values, aspect="auto", cmap="coolwarm", norm=norm)
        ax.set_title(method_rename[methname], fontsize=16)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(feature_cols, rotation=90, fontsize=11)
        ax.set_yticks(range(profile.shape[0]))
        ax.set_yticklabels([f"C{int(x) + 1}" for x in profile.index], fontsize=12)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.04, aspect=30)
        cbar.set_label("Standardized mean", fontsize=14)

    # fig.suptitle(f"{dataset_title}: imputation-averaged cluster profiles", fontsize=20)

    save_figure(fig, outpath_base)


def make_overview_table(missing_df, imp_df, method_summary):
    total_missing = missing_df.iloc[:, 1:].isna().sum().sum()
    total_cells = missing_df.iloc[:, 1:].shape[0] * missing_df.iloc[:, 1:].shape[1]
    selected_k = int(method_summary["selected_k"].iloc[0])
    res = pd.DataFrame(
        {
            "Dataset": [dataset_title],
            "$n$": [int(missing_df.shape[0])],
            "$p$": [int(missing_df.shape[1] - 1)],
            "Complete cases": [int(missing_df.iloc[:, 1:].dropna().shape[0])],
            "$m$": [int(imp_df[".imp"].nunique())],
            "Selected $k$": [selected_k],
            "Missingness": [f"{100.0 * total_missing / total_cells:.1f}\\%"],
        }
    )
    return res


def make_missingness_table(missing_summary):
    tmp = missing_summary.copy().sort_values("missing_prop", ascending=False)
    tmp["Missing proportion"] = tmp["missing_prop"].map(lambda x: f"{100.0 * x:.1f}\\%")
    tmp = tmp.rename(
        columns={
            "variable": "Variable",
            "missing_n": "Missing $n$",
            "nonmissing_n": "Observed $n$",
        }
    )
    return tmp.loc[:, ["Variable", "Missing $n$", "Observed $n$", "Missing proportion"]]


def make_method_summary_table(method_summary):
    rows = []
    for methname in method_order:
        tmp = method_summary.loc[method_summary["method"] == methname, :].iloc[0]
        rows.append(
            {
                "Method": method_rename[methname],
                "$k$": int(tmp["selected_k"]),
                "$n$ clustered": int(tmp["n_clustered"]),
                "Coverage": f"{100.0 * tmp['coverage']:.1f}\\%",
                "Silhouette": (
                    "--"
                    if pd.isna(tmp["Silhouette_mean"])
                    else f"{tmp['Silhouette_mean']:.3f}"
                ),
                "Silhouette SD": (
                    "--"
                    if pd.isna(tmp["Silhouette_sd"])
                    else f"{tmp['Silhouette_sd']:.3f}"
                ),
                "Base instability": (
                    "--"
                    if pd.isna(tmp["base_stability"])
                    else f"{tmp['base_stability']:.3f}"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_agreement_table(agreement_matrix):
    tmp = agreement_matrix.loc[method_order, method_order].copy()
    tmp.index = [method_rename[x] for x in tmp.index]
    tmp.columns = [method_rename[x] for x in tmp.columns]
    for col in tmp.columns:
        tmp[col] = tmp[col].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    tmp.insert(0, "Method", tmp.index)
    return tmp.reset_index(drop=True)


def make_cluster_size_table(labels_dict):
    rows = []
    for methname in method_order:
        labels = labels_dict[methname].dropna().astype(int)
        sizes = labels.value_counts().sort_index()
        for clst in sizes.index:
            rows.append(
                {
                    "Method": method_rename[methname],
                    "Cluster": f"C{int(clst) + 1}",
                    "$n$": int(sizes.loc[clst]),
                    "Proportion": f"{100.0 * sizes.loc[clst] / len(labels):.1f}\\%",
                }
            )
    return pd.DataFrame(rows)


def make_outcome_summary_table(meta_df, labels_dict):
    meta = meta_df.set_index("id")
    rows = []
    for methname in method_order:
        labels = labels_dict[methname].rename("cluster")
        merged = meta.join(labels, how="inner")
        merged = merged.loc[merged["cluster"].notna(), :]
        for outcome in outcome_cols:
            if outcome not in merged.columns:
                continue
            vals = pd.to_numeric(merged[outcome], errors="coerce")
            tmp = pd.DataFrame(
                {"cluster": merged["cluster"].astype(int), "value": vals}
            )
            tmp = tmp.dropna()
            if tmp.shape[0] == 0:
                continue
            res = (
                tmp.groupby("cluster")["value"]
                .agg(["count", "mean", "median"])
                .reset_index()
            )
            for _, row in res.iterrows():
                rows.append(
                    {
                        "Method": method_rename[methname],
                        "Outcome": outcome,
                        "Cluster": f"C{int(row['cluster']) + 1}",
                        "$n$": int(row["count"]),
                        "Mean": f"{row['mean']:.3f}",
                        "Median": f"{row['median']:.3f}",
                    }
                )
    return pd.DataFrame(rows)


def print_latex_table(title, df):
    print(f"\n% {title}")
    print(df.to_latex(index=False, escape=False))


# load data for pbc
dataset_prefix = "pbc"
dataset_title = "PBC"
embedding_method = "tsne"
input_dir = "../data"
results_dir = "../results"
output_dir = "../plots"
missing_df = pd.read_csv(f"../data/pbc_missing_data.csv")
imp_df = pd.read_csv("../data_mi/imp_pbc.csv")
meta_df = pd.read_csv(f"{input_dir}/pbc_meta.csv")
missing_summary = pd.read_csv(f"{results_dir}/pbc_missingness_summary.csv")
labels_df = pd.read_csv(f"{results_dir}/pbc_consensus_labels.csv")
method_summary = pd.read_csv(f"{results_dir}/pbc_method_summary.csv")
res_scores = pd.read_csv(f"{results_dir}/pbc_internal_values.csv")
agreement_matrix = pd.read_csv(f"{results_dir}/pbc_method_agreement.csv", index_col=0)
manifest = load_manifest(f"{results_dir}/pbc_run_manifest.json")
feature_cols = manifest["feature_cols"]

kselect_details_path = f"{results_dir}/pbc_k_selection_details.csv"
if os.path.exists(kselect_details_path):
    res_kselect = pd.read_csv(kselect_details_path)
else:
    res_kselect = None

missing_df["id"] = missing_df["id"].astype(int)
imp_df[".id"] = imp_df[".id"].astype(int)
imp_df[".imp"] = imp_df[".imp"].astype(int)
meta_df["id"] = meta_df["id"].astype(int)
labels_df["id"] = labels_df["id"].astype(int)

figures_dir = "../plots"
os.makedirs(figures_dir, exist_ok=True)

labels_dict = {}
for methname in ["ccakmeanspp", "kpod", "MICluEnHpp", "MICluEnN", "MICluEnNMI"]:
    labels_dict[methname] = labels_df.set_index("id")[methname].copy()
aligned_labels = align_all_labels(labels_dict, reference_method)

embedding_df = compute_mce_embedding(imp_df, feature_cols)
plot_mce_embedding_panels(
    embedding_df,
    aligned_labels,
    method_summary,
    f"{figures_dir}/pbc_mce_embedding_panels",
)
plot_cluster_profile_heatmaps(
    imp_df, feature_cols, aligned_labels, f"{figures_dir}/pbc_cluster_profile_heatmaps"
)
plot_agreement_heatmap(agreement_matrix, f"{figures_dir}/pbc_agreement_heatmap")
plot_missingness(missing_summary, f"{figures_dir}/pbc_missingness")
plot_silhouette_boxplot(res_scores, f"{figures_dir}/pbc_silhouette_boxplot")

overview_table = make_overview_table(missing_df, imp_df, method_summary)
missingness_table = make_missingness_table(missing_summary)
method_table = make_method_summary_table(method_summary)
agreement_table = make_agreement_table(agreement_matrix)
cluster_size_table = make_cluster_size_table(aligned_labels)
outcome_table = make_outcome_summary_table(meta_df, aligned_labels)

print_latex_table("PBC overview", overview_table)
print_latex_table("PBC method summary", method_table)
print_latex_table("PBC agreement matrix", agreement_table)
print_latex_table("PBC cluster sizes", cluster_size_table)
print_latex_table("PBC missingness summary", missingness_table)
if outcome_table.shape[0] > 0:
    print_latex_table("PBC outcome summary", outcome_table)


# load data for support
dataset_prefix = "support"
dataset_title = "SUPPORT"
input_dir = "../data"
results_dir = "../results"
output_dir = "../plots"
missing_df = pd.read_csv(f"../data/support_missing_data.csv")
imp_df = pd.read_csv("../data_mi/imp_support.csv")
meta_df = pd.read_csv(f"{input_dir}/support_meta.csv")
missing_summary = pd.read_csv(f"{results_dir}/support_missingness_summary.csv")
labels_df = pd.read_csv(f"{results_dir}/support_consensus_labels.csv")
method_summary = pd.read_csv(f"{results_dir}/support_method_summary.csv")
res_scores = pd.read_csv(f"{results_dir}/support_internal_values.csv")
agreement_matrix = pd.read_csv(
    f"{results_dir}/support_method_agreement.csv", index_col=0
)
manifest = load_manifest(f"{results_dir}/support_run_manifest.json")
feature_cols = manifest["feature_cols"]

kselect_details_path = f"{results_dir}/support_k_selection_details.csv"
if os.path.exists(kselect_details_path):
    res_kselect = pd.read_csv(kselect_details_path)
else:
    res_kselect = None

missing_df["id"] = missing_df["id"].astype(int)
imp_df[".id"] = imp_df[".id"].astype(int)
imp_df[".imp"] = imp_df[".imp"].astype(int)
meta_df["id"] = meta_df["id"].astype(int)
labels_df["id"] = labels_df["id"].astype(int)

figures_dir = "../plots"
os.makedirs(figures_dir, exist_ok=True)

labels_dict = {}
for methname in ["ccakmeanspp", "kpod", "MICluEnHpp", "MICluEnN", "MICluEnNMI"]:
    labels_dict[methname] = labels_df.set_index("id")[methname].copy()
aligned_labels = align_all_labels(labels_dict, reference_method)

embedding_df = compute_mce_embedding(imp_df, feature_cols)
plot_mce_embedding_panels(
    embedding_df,
    aligned_labels,
    method_summary,
    f"{figures_dir}/support_mce_embedding_panels",
)
plot_cluster_profile_heatmaps(
    imp_df,
    feature_cols,
    aligned_labels,
    f"{figures_dir}/support_cluster_profile_heatmaps",
)
plot_agreement_heatmap(agreement_matrix, f"{figures_dir}/support_agreement_heatmap")
plot_missingness(missing_summary, f"{figures_dir}/support_missingness")
plot_silhouette_boxplot(res_scores, f"{figures_dir}/support_silhouette_boxplot")

overview_table = make_overview_table(missing_df, imp_df, method_summary)
missingness_table = make_missingness_table(missing_summary)
method_table = make_method_summary_table(method_summary)
agreement_table = make_agreement_table(agreement_matrix)
cluster_size_table = make_cluster_size_table(aligned_labels)
outcome_table = make_outcome_summary_table(meta_df, aligned_labels)

print_latex_table("SUPPORT overview", overview_table)
print_latex_table("SUPPORT method summary", method_table)
print_latex_table("SUPPORT agreement matrix", agreement_table)
print_latex_table("SUPPORT cluster sizes", cluster_size_table)
print_latex_table("SUPPORT missingness summary", missingness_table)
if outcome_table.shape[0] > 0:
    print_latex_table("SUPPORT outcome summary", outcome_table)
