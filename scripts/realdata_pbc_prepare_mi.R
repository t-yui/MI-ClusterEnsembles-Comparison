# -*- coding: utf-8 -*-
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", grep("^--file=", script_args, value = TRUE))
if (length(script_path) == 1) {
  setwd(dirname(normalizePath(script_path)))
}

library(survival)
library(mice)
library(data.table)
library(dplyr)


# parameters
output_dir <- "../realdata/pbc"
m <- 30
maxit <- 50
seed.num <- 1
max_missing_prop <- 0.60

candidate_cluster_vars <- c(
  "age", "edema", "bili", "chol", "albumin", "copper",
  "alk.phos", "ast", "trig", "platelet", "protime"
)

candidate_meta_vars <- c(
  "status", "time", "stage", "trt", "sex", "ascites",
  "hepato", "spiders"
)

# categorical variables more than 3 level
categorical_cluster_vars <- c("edema")


dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


make_dummy_df <- function(dat, vars_to_dummy) {
  out_list <- list()

  for (nm in names(dat)) {
    x <- dat[[nm]]

    if (nm %in% vars_to_dummy && is.factor(x) && nlevels(x) >= 3) {
      levs <- levels(x)

      for (lv in levs[-1]) {
        new_name <- paste0(nm, "_", make.names(lv))
        out_list[[new_name]] <- ifelse(is.na(x), NA_real_, as.numeric(as.character(x) == lv))
      }
    } else {
      out_list[[nm]] <- x
    }
  }

  as.data.frame(out_list, check.names = FALSE)
}


# load data
data(pbc, package = "survival")
df <- as.data.frame(pbc)

cluster_vars_raw <- intersect(candidate_cluster_vars, names(df))
meta_vars <- intersect(candidate_meta_vars, names(df))

categorical_cluster_vars <- intersect(categorical_cluster_vars, cluster_vars_raw)
df[categorical_cluster_vars] <- lapply(df[categorical_cluster_vars], factor)

missing_prop <- sapply(df[, cluster_vars_raw, drop = FALSE], function(x) mean(is.na(x)))
cluster_vars_raw <- cluster_vars_raw[missing_prop <= max_missing_prop]

is_nonconstant <- sapply(df[, cluster_vars_raw, drop = FALSE], function(x) {
  length(unique(x[!is.na(x)])) >= 2
})
cluster_vars_raw <- cluster_vars_raw[is_nonconstant]

keep_row <- rowSums(!is.na(df[, cluster_vars_raw, drop = FALSE])) > 0
orig_id <- df$id[keep_row]

df_cluster_raw <- df[keep_row, cluster_vars_raw, drop = FALSE]
row.names(df_cluster_raw) <- orig_id

dummy_vars <- intersect(categorical_cluster_vars, names(df_cluster_raw))
df_cluster <- make_dummy_df(df_cluster_raw, dummy_vars)
cluster_vars <- names(df_cluster)

df_meta <- df[keep_row, meta_vars, drop = FALSE]
df_meta <- cbind(id = orig_id, df_meta)

missing_summary <- data.frame(
  variable = cluster_vars,
  missing_n = colSums(is.na(df_cluster)),
  missing_prop = sapply(df_cluster, function(x) mean(is.na(x))),
  nonmissing_n = nrow(df_cluster) - colSums(is.na(df_cluster))
)

write.csv(
  cbind(id = orig_id, df_cluster),
  file.path(output_dir, "pbc_missing_data.csv"),
  row.names = FALSE
)
write.csv(df_meta, file.path(output_dir, "pbc_meta.csv"), row.names = FALSE)
write.csv(
  missing_summary,
  file.path(output_dir, "pbc_missingness_summary.csv"),
  row.names = FALSE
)


# multiple imputation
method_list <- sapply(df_cluster_raw, function(x) {
  if (is.factor(x)) {
    if (nlevels(x) >= 3) {
      "polyreg"
    } else {
      "logreg"
    }
  } else {
    "pmm"
  }
})

production_mice <- mice(
  df_cluster_raw,
  m = m,
  maxit = maxit,
  method = method_list,
  seed = seed.num,
  remove.collinear = FALSE
)

imp_list <- lapply(seq_len(m), function(imp_no) {
  tmp <- complete(production_mice, action = imp_no)
  tmp <- make_dummy_df(tmp, dummy_vars)
  data.frame(.imp = imp_no, .id = orig_id, tmp, check.names = FALSE)
})

imp <- as.data.frame(rbindlist(imp_list, use.names = TRUE))
imp$.id <- as.integer(imp$.id)
imp <- imp[, c(".imp", ".id", cluster_vars), drop = FALSE]

write.csv(imp, file.path(output_dir, "imp_pbc.csv"), row.names = FALSE)
saveRDS(production_mice, file.path(output_dir, "pbc_mids.rds"))


# manifest
manifest <- c(
  sprintf("dataset=PBC"),
  sprintf("n_rows=%d", nrow(df_cluster)),
  sprintf("n_complete_cases=%d", sum(stats::complete.cases(df_cluster))),
  sprintf("n_vars=%d", length(cluster_vars)),
  sprintf("m=%d", m),
  sprintf("maxit=%d", maxit),
  sprintf("seed=%d", seed.num),
  sprintf("max_missing_prop=%.2f", max_missing_prop),
  sprintf("cluster_vars=%s", paste(cluster_vars, collapse = ", "))
)
writeLines(manifest, file.path(output_dir, "pbc_manifest.txt"))

cat("Processed: PBC\n")
cat("n =", nrow(df_cluster), ", p =", length(cluster_vars), ", m =", m, "\n")
