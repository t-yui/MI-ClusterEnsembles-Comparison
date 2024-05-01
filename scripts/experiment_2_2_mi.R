# -*- coding: utf-8 -*-
setwd("/your_own_path/MI-ClusterEnsembles-Comparison/scripts/")

library(mice)
library(data.table)
library(dplyr)
library(parallel)


# parameters
m <- 30
maxit <- 50
seed.num <- 123
numCores <- 10
n_values <- c(30, 60, 120)
rho_values <- c(0.3, 0.6)
tau_values <- c(0.1, 0.3, 0.5)
scenarios <- c("MCAR", "MAR") 
S <- 200

configs <- expand.grid(
  s = 1:S,
  tau = tau_values,
  rho = rho_values,
  n = n_values,
  scenario = scenarios
)


# class-balanced scenarios
process_data <- function(n, rho, tau, scenario, s) {
  file_path <- sprintf("../data/3c_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s)
  outpath <- sprintf("../data_mi/imp_3c_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s)
  df <- read.csv(file_path)[,-1]
  method_list <- rep("pmm", ncol(df))
  production_mice <- mice(
    df, m = m, maxit = maxit, method = method_list, seed = seed.num, remove.collinear = FALSE)
  imp <- complete(production_mice, "long")
  write.csv(imp, outpath)
  cat("Processed: n =", n, ", rho =", rho, ", tau =", tau, ", scenario =", scenario, ", s =", s, "\n")
}

mclapply(1:nrow(configs), function(i) {
  n <- configs$n[i]
  rho <- configs$rho[i]
  tau <- configs$tau[i]
  scenario <- configs$scenario[i]ons
  s <- configs$s[i]
  process_data(n, rho, tau, scenario, s-1)
}, mc.cores = numCores)


# class-imbalanced scenarios
process_data <- function(n, rho, tau, scenario, s) {
  file_path <- sprintf("../data/3cib_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s)
  outpath <- sprintf("../data_mi/imp_3cib_n%d_rho%.1f_tau%.1f_%s_%d.csv", n, rho, tau, scenario, s)
  df <- read.csv(file_path)[,-1]
  method_list <- rep("pmm", ncol(df))
  production_mice <- mice(df, m = m, maxit = maxit, method = method_list, seed = seed.num, remove.collinear = FALSE)
  imp <- complete(production_mice, "long")
  write.csv(imp, outpath)
  cat("Processed: n =", n, ", rho =", rho, ", tau =", tau, ", scenario =", scenario, ", s =", s, "\n")
}

mclapply(1:nrow(configs), function(i) {
  n <- configs$n[i]
  rho <- configs$rho[i]
  tau <- configs$tau[i]
  scenario <- configs$scenario[i]
  s <- configs$s[i]
  process_data(n, rho, tau, scenario, s-1)
}, mc.cores = numCores)





