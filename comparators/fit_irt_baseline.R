#!/usr/bin/env Rscript
#
# Marginal maximum-likelihood IRT baselines, unidimensional and multidimensional.
#
# Distinct from validate_grm.R, which is the *independent reference* for
# acceptance test T3 and must stay untouched. This script is a *baseline*: it is
# allowed to be called inside the evaluation loop and its output is compared
# against other methods rather than against the generator.
#
# Keeping them separate matters. If the reference and the baseline were the same
# script, improving the baseline would silently move the reference that
# validates the generator.
#
# Input : headerless CSV of integer response categories, rows = persons.
#         De-identified by construction: no identifiers, timestamps or metadata.
# Output: JSON with EAP person scores and item parameters.
#
# Usage:
#   Rscript fit_irt_baseline.R --input r.csv --output o.json --dimensions 1

suppressWarnings(suppressMessages({
  ok <- requireNamespace("mirt", quietly = TRUE)
}))
if (!ok) {
  cat("ERROR: the 'mirt' package is not installed\n", file = stderr())
  quit(status = 2)
}

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  hit <- which(args == flag)
  if (length(hit) == 0L || hit[1L] == length(args)) {
    if (is.null(default)) {
      cat(sprintf("ERROR: missing required argument %s\n", flag), file = stderr())
      quit(status = 2)
    }
    return(default)
  }
  args[[hit[1L] + 1L]]
}

input_path <- get_arg("--input")
output_path <- get_arg("--output")
n_dimensions <- as.integer(get_arg("--dimensions", "1"))
# Milestone 8, ADDITIVE: --model-spec <file> switches to the CONFIRMATORY
# path - a fixed item-to-factor mapping via mirt.model syntax, factor
# covariances freed by the spec's COV line, variances fixed at 1, no rotation
# (the confirmatory solution is rotationally identified). Absent the flag,
# behaviour is byte-identical to before this argument existed.
model_spec_path <- get_arg("--model-spec", "")
confirmatory <- nzchar(model_spec_path)

responses <- utils::read.csv(input_path, header = FALSE)
responses[] <- lapply(responses, as.integer)

started <- Sys.time()
if (confirmatory) {
  spec_text <- paste(readLines(model_spec_path), collapse = "\n")
  model_object <- mirt::mirt.model(spec_text)
  # Stage 1 section 3, fields 2 and 7: graded itemtype for binary and ordinal
  # alike; EM for K <= 2, QMCEM for K >= 3; NCYCLES 2000, TOL 1e-4.
  chosen_method <- if (n_dimensions >= 3) "QMCEM" else "EM"
  fit <- try(
    mirt::mirt(
      responses,
      model = model_object,
      itemtype = "graded",
      method = chosen_method,
      verbose = FALSE,
      TOL = 1e-4,
      technical = list(NCYCLES = 2000)
    ),
    silent = TRUE
  )
  # Field 8: one retry with MHRM before a world is recorded nonconverged.
  retried <- FALSE
  if (!inherits(fit, "try-error") && !mirt::extract.mirt(fit, "converged")) {
    retried <- TRUE
    chosen_method <- "MHRM"
    fit <- try(
      mirt::mirt(
        responses,
        model = model_object,
        itemtype = "graded",
        method = "MHRM",
        verbose = FALSE,
        technical = list(NCYCLES = 2000)
      ),
      silent = TRUE
    )
  }
} else {
  # Multidimensional GRM needs a rotation to be interpretable. oblimin is the
  # conventional choice for correlated factors, which is what the corpus
  # generates; a varimax solution would force orthogonality the data does not
  # have. This EXPLORATORY path is retained unchanged as a sensitivity
  # analysis; the confirmatory path above is the Milestone 8 primary.
  fit <- try(
    mirt::mirt(
      responses,
      model = n_dimensions,
      itemtype = "graded",
      method = if (n_dimensions > 1) "QMCEM" else "EM",
      verbose = FALSE,
      technical = list(NCYCLES = 1500)
    ),
    silent = TRUE
  )
}

if (inherits(fit, "try-error")) {
  writeLines(
    paste0('{"converged": false, "reason": "fit failed", "n_dimensions": ', n_dimensions, "}"),
    output_path
  )
  quit(status = 0)
}

# full.scores.SE gives the EAP posterior standard deviation per person, which
# the dropout and missingness benchmarks need for uncertainty coverage. Only in
# the unidimensional path: rotated multidimensional SEs are not comparable
# across rotations and would invite over-interpretation.
#
# Confirmatory path (field 6): EAP, QMC integration for K >= 3, NO rotation -
# the fixed-mapping solution is rotationally identified.
scores <- try(
  if (confirmatory) {
    mirt::fscores(fit, method = "EAP", QMC = (n_dimensions >= 3))
  } else if (n_dimensions > 1) {
    mirt::fscores(fit, method = "EAP", rotate = "oblimin")
  } else {
    mirt::fscores(fit, method = "EAP", full.scores.SE = TRUE)
  },
  silent = TRUE
)
if (inherits(scores, "try-error")) {
  writeLines(
    paste0('{"converged": false, "reason": "scoring failed", "n_dimensions": ', n_dimensions, "}"),
    output_path
  )
  quit(status = 0)
}

escape <- function(x) {
  ifelse(is.na(x) | is.nan(x) | is.infinite(x), "null", format(x, scientific = FALSE, trim = TRUE))
}
# JSON string literal without a jsonlite dependency: escape backslash, quote
# and newline - the only characters a mirt.model spec can contain that need it.
jsonlite_free_string <- function(x) {
  x <- gsub("\\\\", "\\\\\\\\", x)
  x <- gsub('"', '\\\\"', x)
  x <- gsub("\n", "\\\\n", x)
  paste0('"', x, '"')
}
as_array <- function(x) paste0("[", paste(escape(x), collapse = ","), "]")
as_matrix <- function(m) {
  rows <- vapply(seq_len(nrow(m)), function(i) as_array(as.numeric(m[i, ])), character(1))
  paste0("[", paste(rows, collapse = ","), "]")
}

elapsed_seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))

parameters <- mirt::coef(fit, IRTpars = (n_dimensions == 1), simplify = TRUE)$items
columns <- colnames(parameters)
discrimination_columns <- columns[grepl("^a", columns)]
threshold_columns <- columns[grepl("^b|^d", columns)]

score_matrix <- as.matrix(scores)
se_columns <- grep("^SE", colnames(score_matrix))
theta_matrix <- score_matrix[, setdiff(seq_len(ncol(score_matrix)), se_columns), drop = FALSE]
se_json <- if (length(se_columns) > 0) {
  paste0('  "theta_se": ', as_matrix(score_matrix[, se_columns, drop = FALSE]), ",\n")
} else {
  ""
}

# Confirmatory extras (fields 3, 7, 8, 10): the freed covariance block IS the
# factor correlation because variances are fixed at 1; the exact model string,
# the method actually used, the retry flag and wall-clock all travel with the
# result.
confirmatory_json <- if (confirmatory) {
  covariance <- mirt::coef(fit, simplify = TRUE)$cov
  paste0(
    '  "confirmatory": true,\n',
    '  "method_used": "', chosen_method, '",\n',
    '  "retried_with_mhrm": ', if (retried) "true" else "false", ",\n",
    '  "elapsed_seconds": ', escape(elapsed_seconds), ",\n",
    '  "model_spec": ', jsonlite_free_string(spec_text), ",\n",
    '  "factor_correlation": ', as_matrix(as.matrix(covariance)), ",\n"
  )
} else {
  paste0('  "elapsed_seconds": ', escape(elapsed_seconds), ",\n")
}

json <- paste0(
  "{\n",
  '  "converged": ', if (mirt::extract.mirt(fit, "converged")) "true" else "false", ",\n",
  '  "n_dimensions": ', n_dimensions, ",\n",
  '  "n_persons": ', nrow(responses), ",\n",
  '  "n_items": ', nrow(parameters), ",\n",
  '  "log_likelihood": ', escape(mirt::extract.mirt(fit, "logLik")), ",\n",
  confirmatory_json,
  se_json,
  '  "theta": ', as_matrix(theta_matrix), ",\n",
  '  "discrimination": ',
  as_matrix(matrix(as.numeric(as.matrix(parameters[, discrimination_columns, drop = FALSE])),
                   nrow = nrow(parameters))), ",\n",
  '  "thresholds": ',
  as_matrix(matrix(as.numeric(as.matrix(parameters[, threshold_columns, drop = FALSE])),
                   nrow = nrow(parameters))), "\n",
  "}\n"
)
writeLines(json, output_path)
