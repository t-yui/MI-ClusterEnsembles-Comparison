import numpy as np
import pandas as pd


def generate_covariance(rho):
    Sigma = np.zeros((10, 10))
    Sigma[:5, :5] = np.identity(5)
    for i in range(5, 10):
        for j in range(5, 10):
            if i == j:
                Sigma[i, j] = 1
            else:
                Sigma[i, j] = rho
    return Sigma


def generate_missing_data(data, tau, mechanism="MCAR", a_tau=None):
    missing_data = data.copy()
    while True:
        if mechanism == "MCAR":
            missing_mask = np.random.rand(*data.shape) < tau
            missing_data[missing_mask] = np.nan
        elif mechanism == "MAR" and a_tau is not None:
            prob_missing = norm.cdf(a_tau + data.iloc[:, 0].to_numpy()[:, np.newaxis])
            prob_missing = np.column_stack(
                [
                    np.zeros(prob_missing.shape[0]),
                    np.repeat(prob_missing, data.shape[1] - 1, axis=1),
                ]
            )
            missing_mask = np.random.rand(*data.shape) < prob_missing
            missing_data[missing_mask] = np.nan
        complete_rows = (~missing_data.isna()).sum(axis=1)
        if (complete_rows >= 3).sum() >= 2:
            break
    return missing_data


## parameters
p = 10
mu1 = np.zeros(p)
mu2 = np.concatenate([np.zeros(5), np.ones(5) * 2])
mu3 = np.concatenate([np.ones(10) * 2])
n_values = [30, 60, 120]
rho_values = [0.3, 0.6]
S = 200


# class-balanced scenarios
np.random.seed(seed=1)

## generate complete data
for n in n_values:
    for rho in rho_values:
        Sigma = generate_covariance(rho)
        for s in range(S):
            n1 = np.int64(np.round(n / 3))
            n2 = np.int64(np.round(n / 3))
            n3 = n - n1 - n2
            cluster1_data = np.random.multivariate_normal(mu1, Sigma, size=n1)
            cluster2_data = np.random.multivariate_normal(mu2, Sigma, size=n2)
            cluster3_data = np.random.multivariate_normal(mu3, Sigma, size=n3)
            label1 = np.ones(n1)
            label2 = np.ones(n2) * 2
            label3 = np.ones(n3) * 3
            data = np.vstack([cluster1_data, cluster2_data, cluster3_data])
            labels = np.array(list(label1) + list(label2) + list(label3))
            pd.DataFrame(data).to_csv(
                "../data/3c_n{0}_rho{1}_{2}".format(n, rho, s) + ".csv"
            )
            pd.DataFrame(labels).to_csv(
                "../data/labels_3c_n{0}_rho{1}_{2}".format(n, rho, s) + ".csv"
            )

## generate missingness
for n in n_values:
    for rho in rho_values:
        for s in range(S):
            file_path = "../data/3c_n{}_rho{}_{}.csv".format(n, rho, s)
            data = pd.read_csv(file_path, index_col=0)
            for tau in tau_values:
                # MCAR mechanism
                missing_data_mcar = generate_missing_data(data, tau, mechanism="MCAR")
                print(
                    "MCAR",
                    missing_data_mcar.shape,
                    np.round(
                        missing_data_mcar.isna().sum().sum()
                        / missing_data_mcar.size
                        * 100
                    )
                    / 100,
                )
                missing_data_mcar.to_csv(
                    "../data/3c_n{}_rho{}_tau{}_MCAR_{}.csv".format(n, rho, tau, s)
                )
                # MAR mechanism
                a_tau = np.mean(
                    norm.ppf(tau) - data.iloc[:, 0].to_numpy()[:, np.newaxis]
                )
                missing_data_mar = generate_missing_data(
                    data, tau, mechanism="MAR", a_tau=a_tau
                )
                print(
                    "MAR",
                    missing_data_mar.shape,
                    np.round(
                        missing_data_mar.isna().sum().sum()
                        / missing_data_mar.size
                        * 100
                    )
                    / 100,
                )
                missing_data_mar.to_csv(
                    "../data/3c_n{}_rho{}_tau{}_MAR_{}.csv".format(n, rho, tau, s)
                )


# class-imbalanced scenarios
np.random.seed(seed=1)

## generate complete data
for n in n_values:
    for rho in rho_values:
        Sigma = generate_covariance(rho)
        for s in range(S):
            n1 = np.int64(np.round(n / 6))
            n2 = np.int64(np.round(n / 3))
            n3 = n - n1 - n2
            cluster1_data = np.random.multivariate_normal(mu1, Sigma, size=n1)
            cluster2_data = np.random.multivariate_normal(mu2, Sigma, size=n2)
            cluster3_data = np.random.multivariate_normal(mu3, Sigma, size=n3)
            label1 = np.ones(n1)
            label2 = np.ones(n2) * 2
            label3 = np.ones(n3) * 3
            data = np.vstack([cluster1_data, cluster2_data, cluster3_data])
            labels = np.array(list(label1) + list(label2) + list(label3))
            pd.DataFrame(data).to_csv(
                "../data/3cib_n{0}_rho{1}_{2}".format(n, rho, s) + ".csv"
            )
            pd.DataFrame(labels).to_csv(
                "../data/labels_3cib_n{0}_rho{1}_{2}".format(n, rho, s) + ".csv"
            )

## generate missingness
for n in n_values:
    for rho in rho_values:
        for s in range(S):
            file_path = "../data/3cib_n{}_rho{}_{}.csv".format(n, rho, s)
            data = pd.read_csv(file_path, index_col=0)
            for tau in tau_values:
                # MCAR mechanism
                missing_data_mcar = generate_missing_data(data, tau, mechanism="MCAR")
                print(
                    "MCAR",
                    missing_data_mcar.shape,
                    np.round(
                        missing_data_mcar.isna().sum().sum()
                        / missing_data_mcar.size
                        * 100
                    )
                    / 100,
                )
                missing_data_mcar.to_csv(
                    "../data/3cib_n{}_rho{}_tau{}_MCAR_{}.csv".format(n, rho, tau, s)
                )
                # MAR mechanism
                a_tau = np.mean(
                    norm.ppf(tau) - data.iloc[:, 0].to_numpy()[:, np.newaxis]
                )
                missing_data_mar = generate_missing_data(
                    data, tau, mechanism="MAR", a_tau=a_tau
                )
                print(
                    "MAR",
                    missing_data_mar.shape,
                    np.round(
                        missing_data_mar.isna().sum().sum()
                        / missing_data_mar.size
                        * 100
                    )
                    / 100,
                )
                missing_data_mar.to_csv(
                    "../data/3cib_n{}_rho{}_tau{}_MAR_{}.csv".format(n, rho, tau, s)
                )
