import pandas as pd
import math

FILES = [
    (
        "../results/friedman_panel_balanced.csv",
        "Friedman test results for Experiment 1 (balanced)",
        "tab:friedman-panel-balanced",
    ),
    (
        "../results/friedman_panel_imbalanced.csv",
        "Friedman test results for Experiment 1 (imbalanced)",
        "tab:friedman-panel-imbalanced",
    ),
    (
        "../results/friedman_by_balance_rho_tau_MCAR.csv",
        "Friedman test results for Experiment 2 (MCAR)",
        "tab:friedman-mcar",
    ),
    (
        "../results/friedman_by_balance_rho_tau_MAR.csv",
        "Friedman test results for Experiment 2 (MAR)",
        "tab:friedman-mar",
    ),
]

METRIC_MAP = {
    "ARI": "ARI",
    "NMI": "NMI",
    "P": "Purity",
    "SS": "Silhouette",
    "Purity": "Purity",
    "Silhouette": "Silhouette",
}

METRIC_ORDER = {
    "ARI": 0,
    "NMI": 1,
    "Purity": 2,
    "Silhouette": 3,
}

SCENARIO_ORDER = {"balanced": 0, "imbalanced": 1}
N_ORDER = {30: 0, 60: 1, 120: 2}
RHO_ORDER = {0.3: 0, 0.6: 1}
TAU_ORDER = {0.1: 0, 0.3: 1, 0.5: 2}


def fmt_num(x, digits=3):
    return f"{x:.{digits}f}"


def fmt_p(x):
    if x < 1e-3:
        e = math.floor(math.log10(x))
        m = x / (10**e)
        return rf"${m:.2f}\times 10^{{{e}}}$"
    return f"{x:.3f}"


def best_methods(row, rank_cols, tol=1e-12):
    vals = {c.replace("mean_rank_", ""): row[c] for c in rank_cols}
    m = min(vals.values())
    return [k for k, v in vals.items() if abs(v - m) <= tol]


def tex_escape(s):
    return str(s).replace("_", r"\_")


def metric_name(x):
    return METRIC_MAP.get(x, x)


def sort_key_exp1(row):
    return (
        METRIC_ORDER.get(metric_name(row["metric"]), 99),
        SCENARIO_ORDER.get(row["scenario"], 99),
        N_ORDER.get(row["n"], 99),
    )


def sort_key_exp2(row):
    return (
        METRIC_ORDER.get(metric_name(row["metric"]), 99),
        SCENARIO_ORDER.get(row["balance"], 99),
        RHO_ORDER.get(row["rho"], 99),
        TAU_ORDER.get(row["tau"], 99),
    )


def print_table(df, caption, label):
    rank_cols = [c for c in df.columns if c.startswith("mean_rank_")]
    rank_names = [c.replace("mean_rank_", "") for c in rank_cols]

    if "friedman_chi2" in df.columns:
        stat_col = "friedman_chi2"
        headers = (
            ["Metric", "Scenario", "$n$", "Friedman $\\chi^2$", "p-value"]
            + rank_names
            + ["Best"]
        )
        align = "c" + "c" * (2 + len(rank_cols)) + "c"
        rows = []
        for _, row in df.iterrows():
            best = best_methods(row, rank_cols)

            ranks = []
            for c, name in zip(rank_cols, rank_names):
                s = fmt_num(row[c], 3)
                if name in best:
                    s = rf"\textbf{{{s}}}"
                ranks.append(s)

            cells = (
                [
                    tex_escape(metric_name(row["metric"])),
                    tex_escape(row["scenario"]),
                    str(row["n"]),
                    fmt_num(row[stat_col], 2),
                    fmt_p(row["p_value"]),
                ]
                + ranks
                + [tex_escape("/".join(best))]
            )

            rows.append((sort_key_exp1(row), metric_name(row["metric"]), cells))
    else:
        stat_col = "statistic"
        headers = (
            ["Metric", "Balance", "$\\rho$", "$\\tau$", "Friedman $\\chi^2$", "p-value"]
            + rank_names
            + ["Best"]
        )
        align = "cccc" + "c" * (2 + len(rank_cols)) + "c"
        rows = []
        for _, row in df.iterrows():
            best = best_methods(row, rank_cols)

            ranks = []
            for c, name in zip(rank_cols, rank_names):
                s = fmt_num(row[c], 3)
                if name in best:
                    s = rf"\textbf{{{s}}}"
                ranks.append(s)

            cells = (
                [
                    tex_escape(metric_name(row["metric"])),
                    tex_escape(row["balance"]),
                    fmt_num(row["rho"], 1),
                    fmt_num(row["tau"], 1),
                    fmt_num(row[stat_col], 2),
                    fmt_p(row["p_value"]),
                ]
                + ranks
                + [tex_escape("/".join(best))]
            )

            rows.append((sort_key_exp2(row), metric_name(row["metric"]), cells))

    rows.sort(key=lambda x: x[0])

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(rf"\caption{{{caption}. Best is defined by the smallest mean rank.}}")
    print(rf"\label{{{label}}}")
    print(rf"\begin{{tabular}}{{{align}}}")
    print(r"\toprule")
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")

    prev_metric = None
    for i, (_, metric, cells) in enumerate(rows):
        if i > 0 and metric != prev_metric:
            print(r"\midrule")
        print(" & ".join(map(str, cells)) + r" \\")
        prev_metric = metric

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


for path, caption, label in FILES:
    print_table(pd.read_csv(path), caption, label)
