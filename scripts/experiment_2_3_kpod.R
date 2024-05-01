# -*- coding: utf-8 -*-
setwd("/your_own_path/MI-ClusterEnsembles-Comparison/scripts/")

library(kpodclustr)


# parameters
n_values <- c(30, 60, 120)
rho_values <- c(0.3, 0.6)
tau_values <- c(0.1, 0.3, 0.5)
scenarios <- c("MCAR", "MAR")
S <- 200


# class-balanced scenarios
for (scenario in scenarios) {
  for (rho in rho_values) {
    for (tau in tau_values) {
      for (n in n_values) {
        results <- list()
        for (s in 1:S) {
          missing_file_path <- sprintf("../data/3c_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s-1)
          X <- read.csv(missing_file_path)
          X <- X[,-1]
          X <- scale(X)
          k <- 3
          res <- kpod(X, k)
          results[[s]] <- res$cluster
        }
        results.df <- as.data.frame(do.call(cbind, results))
        output_path <- sprintf("../results/res_kpod_3c_n%d_rho%.1f_tau%.1f_%s.csv", n, rho, tau, scenario)
        write.csv(results.df, output_path)
      }
    }
  }
}


# class-imbalanced scenarios
for (scenario in scenarios) {
  for (rho in rho_values) {
    for (tau in tau_values) {
      for (n in n_values) {
        results <- list()
        for (s in 1:S) {
          missing_file_path <- sprintf("../data/3cib_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s-1)
          X <- read.csv(missing_file_path)
          X <- X[,-1]
          X <- scale(X)
          k <- 3
          res <- kpod(X, k)
          results[[s]] <- res$cluster
        }
        results.df <- as.data.frame(do.call(cbind, results))
        output_path <- sprintf("../results/res_kpod_3cib_n%d_rho%.1f_tau%.1f_%s.csv", n, rho, tau, scenario)
        write.csv(results.df, output_path)
      }
    }
  }
}

