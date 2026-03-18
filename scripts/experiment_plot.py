import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"

# common settings
metrics_to_plot = ["ARI", "NMI", "SS", "P"]

metric_labels = {
    "ARI": "ARI",
    "NMI": "NMI",
    "SS": "Silhouette score",
    "P": "Purity",
}

metric_ylims = {
    "ARI": (-0.1, 1.05),
    "NMI": (-0.1, 1.05),
    "SS": (-1.05, 1.05),
    "P": (-0.1, 1.05),
}

metric_yticks = {
    "ARI": [0, 0.25, 0.5, 0.75, 1],
    "NMI": [0, 0.25, 0.5, 0.75, 1],
    "SS": [-1, -0.5, 0, 0.5, 1],
    "P": [0, 0.25, 0.5, 0.75, 1],
}

method_rename = {
    "ccakmeanspp": "k-means-CCA",
    "kpod": "k-pod",
    "MICluEnHpp": "MI-AClu",
    "MICluEnNMI": "MI-GNMI",
    "MICluEnN": "MI-NMF",
}

# parameters: experiment 1
Ks = [10, 20, 40]
N = 3
exp1_scenarios = ["balanced", "imbalanced"]
std_devs = [1, 1.5, 2, 2.5, 3, 3.5, 4]
methods = ["GNMI", "NMF", "AClu"]

# parameters: experiment 2
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
tau_values = [0.1, 0.3, 0.5]
exp2_scenarios = ["MCAR", "MAR"]

# color palettes
exp1_hue_order = ["GNMI", "NMF", "AClu"]
exp1_palette = dict(zip(exp1_hue_order, sns.color_palette("Set2", len(exp1_hue_order))))

exp2_hue_order = [
    "k-means-full",
    "k-means-CCA",
    "k-pod",
    "MI-GNMI",
    "MI-NMF",
    "MI-AClu",
]
exp2_palette = dict(zip(exp2_hue_order, sns.color_palette("Set2", len(exp2_hue_order))))

BOXPLOT_STYLE = dict(
    linewidth=4,
    fliersize=10,
    saturation=1,
    boxprops=dict(edgecolor="black"),
    whiskerprops=dict(color="black", linewidth=3),
    capprops=dict(color="black", linewidth=3),
    medianprops=dict(color="black", linewidth=4),
)

LINE_STYLE = dict(
    marker="o",
    linestyle="dashed",
    linewidth=2,
    markersize=12,
    color="black",
    label="Instability",
)


def add_combined_legend(ax1, ax2, fontsize=32):
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    if ax1.legend_ is not None:
        ax1.legend_.remove()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
        fontsize=fontsize,
    )


def get_stabs(stab_file, n, rho, scenario):
    stab = pd.read_csv(stab_file, index_col=0).iloc[:, 0]
    return np.array(
        [stab.loc[f"n{n}_rho{rho}_tau{tau}_{scenario}"] for tau in tau_values],
        dtype=float,
    )


def load_exp1_metric(K, scenario, metric):
    frames = []
    for STD_DEV in std_devs:
        for methname in methods:
            file_name = (
                f"../results/{scenario}_n{N*K}_STDDEV_{STD_DEV}_methname{methname}.csv"
            )
            vals = pd.read_csv(file_name)[metric]
            frames.append(
                pd.DataFrame(
                    {
                        "Method": methname,
                        "Score": vals,
                        "sigma": STD_DEV,
                    }
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
            file_name = (
                f"../results/res_scores_{method}_{prefix}_n{n}_rho{rho_value}"
                f"_tau{tau}_{scenario}.csv"
            )
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


# experiment 1 -> figure 1
for metric in metrics_to_plot:
    fig, axes = plt.subplots(
        nrows=len(Ks),
        ncols=len(exp1_scenarios),
        figsize=(24 * 2, 10 * 2),
    )

    for i, K in enumerate(Ks):
        for j, scenario in enumerate(exp1_scenarios):
            score_data = load_exp1_metric(K, scenario, metric)
            stabs = (
                pd.read_csv(f"../results/stabs_{scenario}.csv")
                .values[i * len(std_devs) : (i + 1) * len(std_devs)]
                .ravel()
            )

            ax1 = axes[i, j]
            sns.boxplot(
                x="sigma",
                y="Score",
                hue="Method",
                data=score_data,
                order=std_devs,
                hue_order=exp1_hue_order,
                palette=exp1_palette,
                ax=ax1,
                **BOXPLOT_STYLE,
            )

            ax1.set_xlabel("$\\sigma^2$", fontsize=36)
            ax1.set_ylabel(metric_labels[metric], fontsize=40)
            ax1.set_xticks(range(len(std_devs)))
            ax1.set_xticklabels([str(sd) for sd in std_devs], fontsize=36)
            ax1.set_yticks(metric_yticks[metric])
            ax1.set_ylim(*metric_ylims[metric])
            ax1.tick_params(axis="y", labelcolor="black", labelsize=36)
            ax1.set_title(r"$n = {0}$, {1}".format(N * K, scenario), fontsize=40)

            ax2 = ax1.twinx()
            ax2.plot(range(len(std_devs)), stabs, **LINE_STYLE)
            ax2.set_ylabel("Instability", fontsize=40, color="black")
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax2.tick_params(axis="y", labelcolor="black", labelsize=36)
            ax2.set_ylim(-0.1, 1.05)

            add_combined_legend(ax1, ax2, fontsize=32)

    plt.tight_layout()
    plt.savefig(f"../plots/figure_1_{metric}.eps", bbox_inches="tight")
    plt.savefig(f"../plots/figure_1_{metric}.png", bbox_inches="tight")
    plt.close(fig)


# experiment 2 -> figures 2-5
def plot_exp2_family(prefix, stab_file, first_fig_no):
    for metric in metrics_to_plot:
        if metric == "SS":
            metric_full = "Silhouette"
        elif metric == "P":
            metric_full = "Purity"
        else:
            metric_full = metric
        nfig = first_fig_no

        for rho_value in rho_values:
            fig, axes = plt.subplots(
                nrows=len(n_values),
                ncols=len(exp2_scenarios),
                figsize=(24 * 2, 10 * 2),
            )

            for n_index, n in enumerate(n_values):
                for scenario_index, scenario in enumerate(exp2_scenarios):
                    score_data = load_exp2_metric(
                        prefix, n, rho_value, scenario, metric_full
                    )
                    stabs_tmp = get_stabs(stab_file, n, rho_value, scenario)

                    ax1 = axes[n_index, scenario_index]
                    sns.boxplot(
                        x="tau",
                        y="Score",
                        hue="Method",
                        data=score_data,
                        order=tau_values,
                        hue_order=exp2_hue_order,
                        palette=exp2_palette,
                        ax=ax1,
                        **BOXPLOT_STYLE,
                    )

                    ax1.set_xlabel("", fontsize=36)
                    ax1.set_ylabel(metric_labels[metric], fontsize=40)
                    ax1.set_xticks(range(len(tau_values)))
                    ax1.set_xticklabels(
                        [r"$\tau=0.1$", r"$\tau=0.3$", r"$\tau=0.5$"],
                        fontsize=40,
                    )
                    ax1.set_yticks(metric_yticks[metric])
                    ax1.set_ylim(*metric_ylims[metric])
                    ax1.tick_params(axis="y", labelcolor="black", labelsize=36)
                    ax1.set_title(
                        r"$n = {0}$, ".format(n)
                        + r"$\rho = {0}$, ".format(rho_value)
                        + f"{scenario}",
                        fontsize=40,
                    )

                    ax2 = ax1.twinx()
                    ax2.plot(range(len(tau_values)), stabs_tmp, **LINE_STYLE)
                    ax2.set_ylabel("Instability", fontsize=40, color="black")
                    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
                    ax2.tick_params(axis="y", labelcolor="black", labelsize=36)
                    ax2.set_ylim(-0.1, 1.05)

                    add_combined_legend(ax1, ax2, fontsize=32)

            plt.tight_layout()
            plt.savefig(f"../plots/figure_{nfig}_{metric}.eps", bbox_inches="tight")
            plt.savefig(f"../plots/figure_{nfig}_{metric}.png", bbox_inches="tight")
            plt.close(fig)

            nfig += 1


# balanced scenario -> figure 2, 3
plot_exp2_family("3c", "../results/res_stability_3c.csv", 2)

# imbalanced scenario -> figure 4, 5
plot_exp2_family("3cib", "../results/res_stability_3cib.csv", 4)


# plot figure 6
from sklearn import preprocessing
from modules import gen_data_imbalanced, get_km_, symmetric_nmf, get_final_partition

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
    data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=l)
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
# labels = get_final_partition(Q)

## figure 6.1
plt.figure(figsize=(10, 8))
sns.heatmap(cM / cls_times, cmap="coolwarm")
plt.savefig("../plots/figure_6_1.eps")
plt.savefig("../plots/figure_6_1.png")
plt.clf()
plt.close()

## figure 6.2
plt.figure(figsize=(10, 8))
sns.heatmap(Q @ S @ Q.T, cmap="coolwarm")
plt.savefig("../plots/figure_6_2.eps")
plt.savefig("../plots/figure_6_2.png")
plt.clf()
plt.close()
