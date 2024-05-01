import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing, metrics
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix
from scipy.stats import mode
import scipy.spatial.distance as ssd
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.special import comb


# data generation for experiment 1
def gen_data_balanced(N, K, DIM, STD_DEV, random_state=1):
    mu1 = np.zeros(10)
    mu2 = np.concatenate([np.zeros(5), np.ones(5) * 2])
    mu3 = np.concatenate([np.ones(10) * 2])
    centroids = [mu1, mu2, mu3]
    data = []
    labels = []
    for i, center in enumerate(centroids):
        cluster_data = center + STD_DEV * np.random.randn(K, DIM)
        data.append(cluster_data)
        labels.extend([i] * K)
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels


def gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=1):
    mu1 = np.zeros(10)
    mu2 = np.concatenate([np.zeros(5), np.ones(5) * 2])
    mu3 = np.concatenate([np.ones(10) * 2])
    centroids = [mu1, mu2, mu3]
    K1 = np.round(N * K / 6)
    K2 = np.round(N * K / 3)
    K3 = N * K - K1 - K2
    Ks = [int(K1), int(K2), int(K3)]
    data = []
    labels = []
    for i, center in enumerate(centroids):
        cluster_data = center + STD_DEV * np.random.randn(Ks[i], DIM)
        data.append(cluster_data)
        labels.extend([i] * Ks[i])
    data = np.vstack(data)
    labels = np.array(labels)
    return data, labels


def gen_data(N, K, DIM, STD_DEV, imbalance=0, random_state=1):
    if imbalance == 0:
        data, labels = gen_data_balanced(N, K, DIM, STD_DEV, random_state=random_state)
    else:
        data, labels = gen_data_imbalanced(
            N, K, DIM, STD_DEV, random_state=random_state
        )
    return data, labels


# evaluation
def nmi(true_labels, pred_labels):
    return normalized_mutual_info_score(true_labels, pred_labels, average_method="max")


def evaluate_clustering(data, labels_true, labels_pred):
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi_score = nmi(labels_true, labels_pred)
    return {"Adjusted Rand Index": ari, "NMI": nmi_score}


# distance for cluster ensemble
def d(C1, C2, u, v):
    ## Mirkin distance
    condition_1 = C1[u] == C1[v] and C2[u] == C2[v]
    condition_2 = C1[u] != C1[v] and C2[u] != C2[v]
    return 0 if condition_1 or condition_2 else 1


def normalize_labels(labels):
    label_mapping = {}
    new_label = 0
    normalized_labels = np.empty_like(labels)
    for idx, label in enumerate(labels):
        if label not in label_mapping:
            label_mapping[label] = new_label
            new_label += 1
        normalized_labels[idx] = label_mapping[label]
    return normalized_labels


def d_V(Ci, C):
    normalized_Ci = normalize_labels(Ci)
    normalized_C = normalize_labels(C)
    total = 0
    n = len(normalized_Ci)
    for u in range(n):
        for v in range(n):
            total += d(normalized_Ci, normalized_C, u, v)
    max_value = n * n
    normalized_total = total / max_value
    return total


# greedy algorithm for optimizing average NMI
def anmi(clusterings, candidate_clustering):
    r = len(clusterings)
    total_nmi = sum(
        normalized_mutual_info_score(
            candidate_clustering, clustering, average_method="max"
        )
        for clustering in clusterings
    )
    return total_nmi / r


def avdist(clusterings, candidate_clustering):
    r = len(clusterings)
    total_nmi = sum(
        1 - d_V(candidate_clustering, clustering) for clustering in clusterings
    )
    return total_nmi / r


def greedy_algorithm_nmi(initial_clustering, clusterings, k):
    current_clustering = np.copy(initial_clustering)
    n = len(current_clustering)
    label_updated = True
    while label_updated:
        label_updated = False
        for i in range(n):
            current_label = current_clustering[i]
            best_label = current_label
            best_anmi = anmi(clusterings, current_clustering)
            for new_label in set(range(k)) - {current_label}:
                current_clustering[i] = new_label
                new_anmi = anmi(clusterings, current_clustering)
                if new_anmi > best_anmi:
                    best_anmi = new_anmi
                    best_label = new_label
                    label_updated = True
            current_clustering[i] = best_label
    optimized_clustering = current_clustering
    return optimized_clustering


# cluster Ensemble algorithms for experiment 1
def get_km_(k, X):
    km = KMeans(n_clusters=k, init="k-means++")
    km.fit(X)
    return km


def ensemble_clustering_(
    N, K, DIM, STD_DEV, cls_times, n_clst=3, imbalance=0, random_state=1
):
    """AClu ensemble algorithm"""
    np.random.seed(random_state)
    n = N * K
    cM = np.zeros([n, n])
    y_preds = []
    for l in range(cls_times):
        data, labels = gen_data(N, K, DIM, STD_DEV, imbalance=imbalance, random_state=l)
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
    distMatrix = 1 - cM / (cls_times)
    distArray = ssd.squareform(distMatrix)
    clu = linkage(distArray, method="ward")
    distance_threshold = clu[-3, 2]
    hclu = fcluster(clu, t=distance_threshold, criterion="distance")
    return hclu, y_preds


def ensemble_clustering_nmi_(
    N, K, DIM, STD_DEV, cls_times, n_clst=3, imbalance=0, random_state=1
):
    """GNMI ensemble algorithm"""
    base_clusterings = []
    y_preds = []
    for l in range(cls_times):
        data, labels = gen_data(N, K, DIM, STD_DEV, imbalance=imbalance, random_state=l)
        data_tmp = data.copy()
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_tmp)
        data_tmp = pd.DataFrame(scaler.transform(data_tmp))
        km = get_km_(k=n_clst, X=data_tmp)
        y_pred = km.predict(data_tmp)
        y_preds.append(y_pred)
        base_clusterings.append(y_pred)
    base_clusterings = np.array(base_clusterings)
    best_anmi = 1
    best_bc = base_clusterings[0]
    for i in range(len(base_clusterings)):
        # select best base labels as the initial label
        bc = base_clusterings[i]
        anmi_score = 1 - anmi(base_clusterings, bc)
        if anmi_score < best_anmi:
            best_anmi = anmi_score
            best_bc = bc.copy()
    consensus = greedy_algorithm_nmi(base_clusterings[0], base_clusterings, n_clst)
    return np.array(consensus), y_preds


def symmetric_nmf(W, k, max_iters=1000, tol=1e-5, random_state=0):
    """
    Symmetric NMF algorithm

    Args:
    W (numpy array): Input matrix (n x n)
    k (int): Number of dimension
    max_iters (int): Maximum number of iterations
    tol (float): Tolerance for convergence

    Returns:
    Q (numpy array): Output matrix (n x k)
    S (numpy array): Output matrix (k x k)
    """
    n = W.shape[0]
    np.random.seed(random_state)
    Q = np.abs(np.random.randn(n, k))
    S = np.diag(np.diag(np.abs(np.random.randn(k, k))))
    Q /= np.linalg.norm(Q, axis=0)
    for i in range(max_iters):
        Q_prev = Q.copy()
        S_prev = S.copy()
        Q *= np.sqrt((W @ Q @ S) / (Q @ Q.T @ W @ Q @ S + 1e-10))
        S *= np.sqrt((Q.T @ W @ Q) / (Q.T @ Q @ S @ Q.T @ Q + 1e-10))
        if (
            np.linalg.norm(Q - Q_prev, "fro") < tol
            and np.linalg.norm(S - S_prev, "fro") < tol
        ):
            break
    return Q, S


def get_final_partition(Q):
    row_sums = Q.sum(axis=1, keepdims=True)
    Q_normalized = Q / (row_sums + 1e-10)
    clusters = np.argmax(Q_normalized, axis=1)
    return clusters


def ensemble_clustering_NMF_(
    N, K, DIM, STD_DEV, cls_times, n_clst=3, imbalance=0, random_state=1
):
    """NMF ensemble algorithm"""
    np.random.seed(random_state)
    n = N * K
    cM = np.zeros([n, n])
    y_preds = []
    for l in range(cls_times):
        data, labels = gen_data(N, K, DIM, STD_DEV, imbalance=imbalance, random_state=l)
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
    best_anmi = 1
    best_labels = None
    for i in range(5):
        # multiple implementation with different initial values and select the best result
        Q, S = symmetric_nmf(cM / cls_times, n_clst, random_state=random_state + i)
        labels = get_final_partition(Q)
        anmi_score = 1 - anmi(y_preds, labels)
        if anmi_score < best_anmi:
            best_anmi = anmi_score
            best_labels = labels.copy()
    return best_labels, y_preds


# Stability of clustering
def stab(labels_list):
    cls_times = len(labels_list)
    pairwise_d = np.zeros((cls_times, cls_times))
    cnt = 0
    for j in range(cls_times):
        for k in range(cls_times):
            if j > k:
                pairwise_d[j, k] = 1 - nmi(labels_list[j], labels_list[k])
                cnt += 1
    return np.sum(pairwise_d) / cnt


def eval_scores(N, K, DIM, STD_DEV, cls_times, imbalance=0, n_clst=3, random_state=1):
    y_preds = []
    for l in range(cls_times):
        if imbalance == 0:
            data, labels = gen_data(N, K, DIM, STD_DEV, random_state=l)
        else:
            data, labels = gen_data_imbalanced(N, K, DIM, STD_DEV, random_state=l)
        data_tmp = data.copy()
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_tmp)
        data_tmp = pd.DataFrame(scaler.transform(data_tmp))
        km = get_km_(k=n_clst, X=data_tmp)
        y_pred = km.predict(data_tmp)
        y_preds.append(y_pred)
    stability = stab(y_preds)
    return stability


# Cluster Ensemble algorithms for experiment 2
def apply_each_clustering_(df_imp, cls_times, n_clst=3):
    ## apply base clustering for each imputed dataset
    cluster_labels = []
    for l in range(cls_times):
        data_tmp = df_imp.loc[df_imp[".imp"] == l + 1, :].reset_index(drop=True)
        data_tmp = data_tmp.iloc[:, 3:]
        scaler = preprocessing.StandardScaler()
        scaler.fit(data_tmp)
        data_tmp = pd.DataFrame(scaler.transform(data_tmp))
        try:
            km = get_km_(k=n_clst, X=data_tmp)
            y_pred = km.predict(data_tmp)
            cluster_labels.append(y_pred)
        except:
            pass
    return cluster_labels


def apply_mi_ensemble_clustering(res, n_clst=3):
    """AClu ensemble algorithm"""
    n = res.shape[0]
    cls_times = res.shape[1]
    cM = np.zeros([n, n])
    for l in range(cls_times):
        y_pred = res.values[:, l]
        for i in range(n):
            for j in range(n):
                if y_pred[i] == y_pred[j]:
                    cM[i, j] += 1
    distMatrix = 1 - cM / (cls_times)
    distArray = ssd.squareform(distMatrix)
    clu = linkage(distArray, method="ward")
    distance_threshold = clu[-3, 2]
    hclu = fcluster(clu, t=distance_threshold, criterion="distance")
    return hclu


def apply_mi_ensemble_clustering_NMF(res, n_clst=3, random_state=1):
    """NMF ensemble algorithm"""
    n = res.shape[0]
    cls_times = res.shape[1]
    cM = np.zeros([n, n])
    y_preds = []
    for l in range(cls_times):
        y_pred = res.values[:, l]
        y_preds.append(y_pred)
        for i in range(n):
            for j in range(n):
                if y_pred[i] == y_pred[j]:
                    cM[i, j] += 1
    best_anmi = 1
    best_labels = y_preds[0]
    for i in range(5):
        Q, S = symmetric_nmf(cM / cls_times, n_clst, random_state=random_state + i)
        labels = get_final_partition(Q)
        anmi_score = 1 - anmi(y_preds, labels)
        if anmi_score < best_anmi:
            best_anmi = anmi_score
            best_labels = labels.copy()
    return best_labels


def apply_mi_ensemble_clustering_nmi(res, n_clst=3):
    """GNMI ensemble algorithm"""
    n = res.shape[0]
    cls_times = res.shape[1]
    y_preds = []
    for l in range(cls_times):
        y_pred = res.values[:, l]
        y_preds.append(y_pred)
    base_clusterings = np.array(y_preds)
    best_anmi = 1
    best_bc = base_clusterings[0]
    for i in range(len(base_clusterings)):
        bc = base_clusterings[i]
        anmi_score = 1 - anmi(base_clusterings, bc)
        if anmi_score < best_anmi:
            best_anmi = anmi_score
            best_bc = bc.copy()
    labels = greedy_algorithm_nmi(best_bc, y_preds, n_clst)
    return labels
