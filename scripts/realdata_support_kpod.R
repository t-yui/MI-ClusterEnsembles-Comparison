# -*- coding: utf-8 -*-
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", grep("^--file=", script_args, value = TRUE))
if (length(script_path) == 1) {
  setwd(dirname(normalizePath(script_path)))
}

library(kpodclustr)


# parameters
input_dir <- "../data"
output_dir <- "../results"
k <- 5
maxiter <- 100
selected_k_path <- file.path(output_dir, "support_selected_k.txt")


dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (file.exists(selected_k_path)) {
  k_line <- readLines(selected_k_path, warn = FALSE)
  if (length(k_line) >= 1 && nzchar(k_line[1])) {
    k <- as.integer(k_line[1])
  }
}


# load data
X <- read.csv(file.path(input_dir, "support_missing_data.csv"))
ids <- X[, 1]
X <- X[, -1]
X <- scale(X)


# k-pod
res <- kpod(X, k, kmpp_flag = TRUE, maxiter = maxiter)
labels <- as.integer(res$cluster) - 1L


# cluster-wise completion
X_completed <- as.data.frame(X)
col_means <- sapply(X_completed, function(x) mean(x, na.rm = TRUE))
col_means[is.na(col_means)] <- 0

for (clst in sort(unique(labels))) {
  idx <- which(labels == clst)
  if (length(idx) == 0) {
    next
  }
  for (j in 1:ncol(X_completed)) {
    obs_vals <- X_completed[idx, j]
    mu <- mean(obs_vals, na.rm = TRUE)
    if (is.na(mu)) {
      mu <- col_means[j]
    }
    idx_missing <- idx[is.na(X_completed[idx, j])]
    if (length(idx_missing) > 0) {
      X_completed[idx_missing, j] <- mu
    }
  }
}

for (j in 1:ncol(X_completed)) {
  X_completed[is.na(X_completed[, j]), j] <- col_means[j]
}


# output
write.csv(
  data.frame(id = ids, kpod = labels),
  file.path(output_dir, "support_kpod_labels.csv"),
  row.names = FALSE
)
write.csv(
  cbind(id = ids, X_completed),
  file.path(output_dir, "support_kpod_completed_scaled.csv"),
  row.names = FALSE
)

cat("Processed: SUPPORT k-pod, k =", k, "\n")
