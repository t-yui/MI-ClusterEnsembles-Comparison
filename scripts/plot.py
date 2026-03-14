import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"


metrics_to_plot = ["ARI", "NMI", "Silhouette", "Purity"]

metric_labels = {
    "ARI": "ARI",
    "NMI": "NMI",
    "Silhouette": "Silhouette score",
    "Purity": "Purity",
}

metric_ylims = {
    "ARI": (-0.1, 1.05),
    "NMI": (-0.1, 1.05),
    "Silhouette": (-1.05, 1.05),
    "Purity": (-0.1, 1.05),
}

metric_yticks = {
    "ARI": [0, 0.25, 0.5, 0.75, 1],
    "NMI": [0, 0.25, 0.5, 0.75, 1],
    "Silhouette": [-1, -0.5, 0, 0.5, 1],
    "Purity": [0, 0.25, 0.5, 0.75, 1],
}

method_rename = {
    "ccakmeanspp": "k-means-CCA",
    "kpod": "k-pod",
    "MICluEnHpp": "MI-AClu",
    "MICluEnNMI": "MI-GNMI",
    "MICluEnN": "MI-NMF",
}


def get_stabs(stab_file, n, rho, scenario):
    stab = pd.read_csv(stab_file, index_col=0).iloc[:, 0]
    return [stab.loc[f"n{n}_rho{rho}_tau{tau}_{scenario}"] for tau in tau_values]


def load_exp1_metric(K, scenario, metric):
    frames = []
    for STD_DEV in std_devs:
        for methname in methods:
            file_name = f"../results/{scenario}_n{N*K}_STDDEV_{STD_DEV}_methname{methname}.csv"
            vals = pd.read_csv(file_name)[metric]
            frames.append(
                pd.DataFrame(
                    {"Method": methname, "Score": vals, "sigma": STD_DEV}
                )
            )
    return pd.concat(frames, ignore_index=True)


def load_exp2_metric(prefix, n, rho_value, scenario, metric):
    frames = []
    for tau in tau_values:
        kmeans_file = f"../results/res_scores_kmeans_{prefix}_n{n}_rho{rho_value}.csv"
        frames.append(
            pd.DataFrame(
                {
                    "Method": "k-means-full",
                    "Score": pd.read_csv(kmeans_file)[metric],
                    "tau": tau,
                }
            )
        )
        for method in ["ccakmeanspp", "kpod", "MICluEnNMI", "MICluEnN", "MICluEnHpp"]:
            file_name = f"../results/res_scores_{method}_{prefix}_n{n}_rho{rho_value}_tau{tau}_{scenario}.csv"
            frames.append(
                pd.DataFrame(
                    {
                        "Method": method,
                        "Score": pd.read_csv(file_name)[metric],
                        "tau": tau,
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    df["Method"] = df["Method"].replace(method_rename)
    return df


# experiment 1
## parameters
Ks = [10, 20, 40]
N = 3
scenarios = ["balanced", "imbalanced"]
std_devs = [1, 1.5, 2, 2.5, 3, 3.5, 4]
methods = ["GNMI", "NMF", "AClu"]
hatches = ["-", "+", "x"]
colors = ["#000000"] * 100

# plot figure 1
for metric in metrics_to_plot:
    fig, axes = plt.subplots(
        nrows=len(Ks), ncols=len(scenarios), figsize=(24 * 2, 10 * 2)
    )
    for i, K in enumerate(Ks):
        for j, scenario in enumerate(scenarios):
            score_data = load_exp1_metric(K, scenario, metric)
            stabs = pd.read_csv(f"../results/stabs_{scenario}.csv").values[
                i * 7 : (i + 1) * 7
            ]

            ax1 = axes[i, j]
            sns.boxplot(
                x="sigma",
                y="Score",
                hue="Method",
                data=score_data,
                palette=colors,
                linewidth=2 * 2,
                fliersize=5 * 2,
                ax=ax1,
            )

            for k, artist in enumerate(ax1.artists):
                col = colors[k % len(colors)]
                artist.set_edgecolor(col)
                artist.set_facecolor("none")
                artist.set_hatch(hatches[k % len(hatches)])

            ax1.set_xlabel("$\\sigma^2$", fontsize=18 * 2)
            ax1.set_ylabel(metric_labels[metric], fontsize=20 * 2)
            ax1.set_xticks(range(len(std_devs)))
            ax1.set_xticklabels([str(sd) for sd in std_devs], fontsize=18 * 2)
            ax1.set_yticks(metric_yticks[metric])
            ax1.set_ylim(*metric_ylims[metric])
            ax1.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
            ax1.set_title(r"$n = {0}$, {1}".format(N * K, scenario), fontsize=20 * 2)

            ax2 = ax1.twinx()
            ax2.plot(
                stabs,
                marker="o",
                linestyle="dashed",
                linewidth=1 * 2,
                markersize=6 * 2,
                label="Instability",
                color="black",
            )
            ax2.set_ylabel("Instability", fontsize=20 * 2, color="black")
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax2.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
            ax2.set_ylim(-0.1, 1.05)

            legend_patches = [
                Patch(
                    facecolor="white",
                    edgecolor=colors[k],
                    hatch=hatches[k % len(hatches)],
                    label=methname,
                )
                for k, methname in enumerate(score_data["Method"].unique())
            ]
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                handles=legend_patches + lines2,
                labels=[lp.get_label() for lp in legend_patches] + labels2,
                loc="upper left",
                bbox_to_anchor=(1.1, 1),
                fontsize=16 * 2,
            )

    plt.tight_layout()
    plt.savefig(f"../plots/figure_1_{metric}.eps")
    plt.clf()
    plt.close()


# experiment 2
## parameters
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
tau_values = [0.1, 0.3, 0.5]
scenarios = ["MCAR", "MAR"]
S = 200
hatches = ["/", "\\", "|", "-", "+", "x"]
colors = ["#000000"] * 100

# plot figure 2 and 3
for metric in metrics_to_plot:
    score_data = load_exp2_metric("3c", n, rho_value, scenario, metric)
    stabs_tmp = get_stabs("../results/res_stability_3c.csv", n, rho_value, scenario)
    #stab_df = pd.DataFrame(pd.read_csv("../results/res_stability_3c.csv").values)
    nfig = 2
    for rho_index, rho_value in enumerate(rho_values):
        fig, axes = plt.subplots(
            nrows=len(n_values), ncols=len(scenarios), figsize=(24 * 2, 10 * 2)
        )
        for n_index, n in enumerate(n_values):
            for scenario_index, scenario in enumerate(scenarios):
                ari_data = pd.DataFrame()
                for tau in tau_values:
                    tau_value = tau
                    kmeans_file = f"res_scores_kmeans_3c_n{n}_rho{rho_value}.csv"
                    kmeans_ari = pd.read_csv(f"../results/{kmeans_file}")["ARI"]
                    ari_data = ari_data.append(
                        pd.DataFrame(
                            {"Method": "k-means-full", "ARI": kmeans_ari, "tau": tau_value}
                        )
                    )

                    for method in [
                        "ccakmeanspp",
                        "kpod",
                        "MICluEnNMI",
                        "MICluEnN",
                        "MICluEnHpp",
                    ]:
                        file_name = f"res_scores_{method}_3c_n{n}_rho{rho_value}_tau{tau_value}_{scenario}.csv"
                        method_ari = pd.read_csv(f"../results/{file_name}")["ARI"]
                        ari_data = ari_data.append(
                            pd.DataFrame(
                                {"Method": method, "ARI": method_ari, "tau": tau_value}
                            )
                        )

                method_rename = {
                    "ccakmeanspp": "k-means-CCA",
                    "kpod": "k-pod",
                    "MICluEnHpp": "MI-AClu",
                    "MICluEnNMI": "MI-GNMI",
                    "MICluEnN": "MI-NMF",
                }
                ari_data["Method"] = ari_data["Method"].replace(method_rename)

                ax1 = axes[n_index, scenario_index]
                sns.boxplot(
                    x="tau",
                    y="Score",
                    hue="Method",
                    data=score_data,
                    linewidth=2 * 2,
                    fliersize=5 * 2,
                    ax=ax1,
                )
                ax1.set_ylabel(metric_labels[metric], fontsize=20 * 2)
                ax1.set_yticks(metric_yticks[metric])
                ax1.set_ylim(*metric_ylims[metric])
                # sns_plot = sns.boxplot(
                #     x="tau",
                #     y="ARI",
                #     hue="Method",
                #     data=ari_data,
                #     linewidth=2 * 2,
                #     fliersize=5 * 2,
                #     ax=ax1,
                # )
                # ax1.set_xlabel("", fontsize=18 * 2)
                # ax1.set_ylabel("ARI", fontsize=20 * 2)
                # ax1.set_xticks([0, 1, 2])
                # ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
                # ax1.set_ylim(-0.1, 1.05)
                ax1.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
                ax1.set_xticklabels(
                    [r"$\tau=0.1$", r"$\tau=0.3$", r"$\tau=0.5$"], fontsize=20 * 2
                )
                ax1.set_title(
                    r"$n = {0}$, ".format(n)
                    + r"$\rho = {0}$, ".format(rho_value)
                    + f"{scenario}",
                    fontsize=20 * 2,
                )
                ax1.set_ylim(-0.1, 1.05)

                # ax2 = ax1.twinx()
                # ax1.plot(
                #     stabs_tmp,
                #     marker="o",
                #     linestyle="dashed",
                #     linewidth=1 * 2,
                #     markersize=6 * 2,
                #     label="instability",
                #     color="black",
                # )
                ax2 = ax1.twinx()
                ax2.plot(
                    stabs_tmp,
                    marker="o",
                    linestyle="dashed",
                    linewidth=1 * 2,
                    markersize=6 * 2,
                    label="instability",
                    color="black",
                )
                ax2.set_ylabel("Instability", fontsize=20 * 2, color="black")
                ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax2.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
                ax2.set_ylim(-0.1, 1.05)

                for i, artist in enumerate(ax1.artists):
                    col = colors[i % len(colors)]
                    artist.set_edgecolor(col)
                    artist.set_facecolor("none")
                    artist.set_hatch(hatches[i % len(hatches)])

                legend_patches = [
                    Patch(
                        facecolor="white",
                        edgecolor=colors[i],
                        hatch=hatches[i % len(hatches)],
                        label=method,
                    )
                    for i, method in enumerate(ari_data["Method"].unique())
                ]
                lines2, labels2 = ax1.get_legend_handles_labels()
                ax1.legend(
                    handles=legend_patches + [lines2[0]],
                    labels=[lp.get_label() for lp in legend_patches] + [labels2[0]],
                    loc="upper left",
                    bbox_to_anchor=(1.1, 1),
                    fontsize=32,
                )

        plt.tight_layout()
        #plt.savefig("../plots/figure_{0}.eps".format(nfig))
        plt.savefig(f"../plots/figure_{nfig}_{metric}.eps")
        nfig += 1
        plt.show()
        plt.clf()
        plt.close()

# plot figure 4 and 5
for metric in metrics_to_plot:
    score_data = load_exp2_metric("3c", n, rho_value, scenario, metric)
    stabs_tmp = get_stabs("../results/res_stability_3cib.csv", n, rho_value, scenario)
    #stab_df = pd.DataFrame(pd.read_csv("../results/res_stability_3cib.csv").values)
    nfig = 4
    for rho_index, rho_value in enumerate(rho_values):
        fig, axes = plt.subplots(
            nrows=len(n_values), ncols=len(scenarios), figsize=(24 * 2, 10 * 2)
        )
        for n_index, n in enumerate(n_values):
            for scenario_index, scenario in enumerate(scenarios):
                ari_data = pd.DataFrame()
                for tau in tau_values:
                    tau_value = tau
                    kmeans_file = f"res_scores_kmeans_3cib_n{n}_rho{rho_value}.csv"
                    kmeans_ari = pd.read_csv(f"../results/{kmeans_file}")["ARI"]
                    ari_data = ari_data.append(
                        pd.DataFrame(
                            {"Method": "k-means-full", "ARI": kmeans_ari, "tau": tau_value}
                        )
                    )

                    for method in [
                        "ccakmeanspp",
                        "kpod",
                        "MICluEnNMI",
                        "MICluEnN",
                        "MICluEnHpp",
                    ]:
                        file_name = f"res_scores_{method}_3cib_n{n}_rho{rho_value}_tau{tau_value}_{scenario}.csv"
                        method_ari = pd.read_csv(f"../results/{file_name}")["ARI"]
                        ari_data = ari_data.append(
                            pd.DataFrame(
                                {"Method": method, "ARI": method_ari, "tau": tau_value}
                            )
                        )

                method_rename = {
                    "ccakmeanspp": "k-means-CCA",
                    "kpod": "k-pod",
                    "MICluEnHpp": "MI-AClu",
                    "MICluEnNMI": "MI-GNMI",
                    "MICluEnN": "MI-NMF",
                }
                ari_data["Method"] = ari_data["Method"].replace(method_rename)

                ax1 = axes[n_index, scenario_index]
                # sns_plot = sns.boxplot(
                #     x="tau",
                #     y="ARI",
                #     hue="Method",
                #     data=ari_data,
                #     linewidth=2 * 2,
                #     fliersize=5 * 2,
                #     ax=ax1,
                # )
                # ax1.set_xlabel("", fontsize=18 * 2)
                # ax1.set_ylabel("ARI", fontsize=20 * 2)
                # ax1.set_xticks([0, 1, 2])
                # ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
                # ax1.set_ylim(-0.1, 1.05)
                sns.boxplot(
                    x="tau",
                    y="Score",
                    hue="Method",
                    data=score_data,
                    linewidth=2 * 2,
                    fliersize=5 * 2,
                    ax=ax1,
                )
                ax1.set_ylabel(metric_labels[metric], fontsize=20 * 2)
                ax1.set_yticks(metric_yticks[metric])
                ax1.set_ylim(*metric_ylims[metric])
                ax1.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
                ax1.set_xticklabels(
                    [r"$\tau=0.1$", r"$\tau=0.3$", r"$\tau=0.5$"], fontsize=20 * 2
                )
                ax1.set_title(
                    r"$n = {0}$, ".format(n)
                    + r"$\rho = {0}$, ".format(rho_value)
                    + f"{scenario}",
                    fontsize=20 * 2,
                )
                ax1.set_ylim(-0.1, 1.05)

                # ax2 = ax1.twinx()
                # ax1.plot(
                #     stabs_tmp,
                #     marker="o",
                #     linestyle="dashed",
                #     linewidth=1 * 2,
                #     markersize=6 * 2,
                #     label="instability",
                #     color="black",
                # )
                ax2 = ax1.twinx()
                ax2.plot(
                    stabs_tmp,
                    marker="o",
                    linestyle="dashed",
                    linewidth=1 * 2,
                    markersize=6 * 2,
                    label="instability",
                    color="black",
                )
                ax2.set_ylabel("Instability", fontsize=20 * 2, color="black")
                ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax2.tick_params(axis="y", labelcolor="black", labelsize=18 * 2)
                ax2.set_ylim(-0.1, 1.05)

                for i, artist in enumerate(ax1.artists):
                    col = colors[i % len(colors)]
                    artist.set_edgecolor(col)
                    artist.set_facecolor("none")
                    artist.set_hatch(hatches[i % len(hatches)])

                legend_patches = [
                    Patch(
                        facecolor="white",
                        edgecolor=colors[i],
                        hatch=hatches[i % len(hatches)],
                        label=method,
                    )
                    for i, method in enumerate(ari_data["Method"].unique())
                ]
                lines2, labels2 = ax1.get_legend_handles_labels()
                ax1.legend(
                    handles=legend_patches + [lines2[0]],
                    labels=[lp.get_label() for lp in legend_patches] + [labels2[0]],
                    loc="upper left",
                    bbox_to_anchor=(1.1, 1),
                    fontsize=32,
                )

        plt.tight_layout()
        #plt.savefig("../plots/figure_{0}.eps".format(nfig))
        plt.savefig(f"../plots/figure_{nfig}_{metric}.eps")
        nfig += 1
        plt.show()
        plt.clf()
        plt.close()


# plot figure 6
import preprocessing
from modules import gen_data, get_km_, symmetric_nmf

np.random.seed(10000)
N = 3
K = 40
DIM = 10
STD_DEV = 1.0
n_clst = 3
cls_times = 30
n = N * K
cM = np.zeros([n, n])
y_preds = []
for l in range(cls_times):
    data, labels = gen_data(N, K, DIM, STD_DEV, random_state=l)
    data_tmp = data.copy()
    scaler = preprocessing.StandardScaler()
    scaler.fit(data_tmp)
    data_tmp = pd.DataFrame(scaler.transform(data_tmp))
    km = get_km_(k=n_clst, X=data_tmp)
    y_pred = km.predict(data_tmp)
    y_preds.append(y_pred)
    for i in range(n):
        for j in range(n):
            if y_pred[i] == y_pred[j]:
                cM[i, j] += 1

Q, S = symmetric_nmf(cM / cls_times, 3)
labels = get_final_partition(Q)

## figure 6.1
plt.figure(figsize=(10, 8))
sns.heatmap(cM / cls_times, cmap="gray_r")
plt.savefig("../plots/figure_6_1.eps")
plt.show()
plt.clf()
plt.close()

## figure 6.2
plt.figure(figsize=(10, 8))
sns.heatmap(Q @ S @ Q.T, cmap="gray_r")
plt.savefig("../plots/figure_6_2.eps")
plt.show()
plt.clf()
plt.close()
