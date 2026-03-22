# -*- coding: utf-8 -*-
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", grep("^--file=", script_args, value = TRUE))
if (length(script_path) == 1) {
  setwd(dirname(normalizePath(script_path)))
}

library(Hmisc)
library(mice)
library(data.table)
library(dplyr)


# parameters
output_dir <- "../data"
m <- 30
maxit <- 50
seed.num <- 1
max_missing_prop <- 0.60

candidate_cluster_vars <- c(
  "age", "num.co", "scoma", "meanbp", "hrt", "resp", "temp",
  "wblc", "sod", "crea", "bun", "alb", "bili", "pafi",
  "urine", "adlp", "adls"
)

candidate_meta_vars <- c(
  "sex", "race", "dzgroup", "dzclass", "income", "edu",
  "hospdead", "death", "d.time", "slos", "totcst", "sfdm2"
)

# load data
Hmisc::getHdata(support)
df <- as.data.frame(support)
cluster_vars <- intersect(candidate_cluster_vars, names(df))
meta_vars <- intersect(candidate_meta_vars, names(df))

missing_prop <- sapply(df[, cluster_vars, drop = FALSE], function(x) mean(is.na(x)))
cluster_vars <- cluster_vars[missing_prop <= max_missing_prop]

is_numeric <- sapply(df[, cluster_vars, drop = FALSE], is.numeric)
cluster_vars <- cluster_vars[is_numeric]

is_nonconstant <- sapply(df[, cluster_vars, drop = FALSE], function(x) {
  length(unique(x[!is.na(x)])) >= 2
})
cluster_vars <- cluster_vars[is_nonconstant]

keep_row <- rowSums(!is.na(df[, cluster_vars, drop = FALSE])) > 0
orig_id <- which(keep_row)
df_cluster <- df[keep_row, cluster_vars, drop = FALSE]
row.names(df_cluster) <- orig_id

df_meta <- df[keep_row, meta_vars, drop = FALSE]
df_meta <- cbind(id = orig_id, df_meta)

missing_summary <- data.frame(
  variable = cluster_vars,
  missing_n = colSums(is.na(df_cluster)),
  missing_prop = sapply(df_cluster, function(x) mean(is.na(x))),
  nonmissing_n = nrow(df_cluster) - colSums(is.na(df_cluster))
)

write.csv(cbind(id = orig_id, df_cluster), file.path("../data_mi", "support_missing_data.csv"), row.names = FALSE)
write.csv(df_meta, file.path("../data_mi", "support_meta.csv"), row.names = FALSE)
write.csv(missing_summary, file.path("../results", "support_missingness_summary.csv"), row.names = FALSE)


# multiple imputation
method_list <- rep("pmm", ncol(df_cluster))
production_mice <- mice(
  df_cluster,
  m = m,
  maxit = maxit,
  method = method_list,
  seed = seed.num,
  remove.collinear = FALSE
)
imp <- complete(production_mice, "long", include = FALSE)
imp$.id <- as.integer(imp$.id)
imp <- imp[, c(".imp", ".id", cluster_vars), drop = FALSE]
write.csv(imp, file.path(output_dir, "imp_support.csv"), row.names = FALSE)
saveRDS(production_mice, file.path(output_dir, "support_mids.rds"))

# manifest
manifest <- c(
  sprintf("dataset=SUPPORT"),
  sprintf("n_rows=%d", nrow(df_cluster)),
  sprintf("n_complete_cases=%d", sum(stats::complete.cases(df_cluster))),
  sprintf("n_vars=%d", length(cluster_vars)),
  sprintf("m=%d", m),
  sprintf("maxit=%d", maxit),
  sprintf("seed=%d", seed.num),
  sprintf("max_missing_prop=%.2f", max_missing_prop),
  sprintf("cluster_vars=%s", paste(cluster_vars, collapse = ", "))
)
writeLines(manifest, file.path("../results", "support_manifest.txt"))
