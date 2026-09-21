#!/usr/bin/env Rscript
# Weighted-estimation and selection references from R VineCopula (audit FR-11,
# wave 3).
#
# Reads the fixed samples of pmcprg/tests/data/parity/weighted_data.csv and
# selection_data.csv (scripts/parity/gen_weighted_data.py; every value an
# integer, so R and Python read the same doubles) and records
#
# * for each weighted sample and each weighting -- unit (no weights), gen
#   (w = w_gen / 2^20), w01 (the {0, 1} weights), subset (no weights, on the
#   points of weight 1) -- VineCopula's weighted Kendall tau (fasttau, i.e.
#   TauMatrix with weights, the C ktau without) and BiCopEst with method
#   "itau" (one-parameter families and Student) and "mle": the native and the
#   key's parameters, the logLik the fitted object reports, BiCopPar2Tau, and
#   the raw weighted sum of log BiCopPDF at the optimum;
# * for each selection sample, unweighted and with the deterministic weights
#   w_i = ((7919 i) mod 1024 + 1) / 1024 of the point of u-rank i, the family
#   BiCopSelect picks among {independence, Gauss, true family} under each of
#   logLik, AIC and BIC (rotations = FALSE, presel = FALSE: exactly those
#   families, as pmcprg's candidates), and each candidate's own MLE fit and
#   weighted log-likelihood (as BiCopSelect computes them; the script stops
#   if its own arg max differs from BiCopSelect's choice).
#
# Written, with the provenance and the md5 of the data files, to
# pmcprg/tests/data/parity/weighted_vinecopula.json, read by
# pmcprg/tests/test_parity_weighted.py. R is not needed at test time.
#
# Usage, from the repository root:
#
#     Rscript scripts/parity/gen_r_weighted.R
#
# Needs VineCopula only (no JSON package: the file is written by hand,
# doubles as "%.17g").

suppressPackageStartupMessages(library(VineCopula))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
root <- normalizePath(file.path(here, "..", ".."))
data_dir <- file.path(root, "pmcprg", "tests", "data", "parity")
W_SCALE <- 2^20

# ---------------------------------------------------------------------------
# JSON writing (as gen_r.R)
# ---------------------------------------------------------------------------
jnum <- function(x) {
  if (is.null(x) || length(x) == 0 || !is.finite(x)) "null" else sprintf("%.17g", x)
}
jvec <- function(x) {
  if (is.null(x)) return("null")
  paste0("[", paste(vapply(as.numeric(x), jnum, ""), collapse = ", "), "]")
}
jstr <- function(s) paste0('"', gsub('"', '\\\\"', s), '"')
jobj <- function(fields, indent = "") {
  body <- paste0(indent, " ", jstr(names(fields)), ": ", unlist(fields), collapse = ",\n")
  paste0("{\n", body, "\n", indent, "}")
}
jline <- function(fields) paste0("  {", paste0(jstr(names(fields)), ": ", fields, collapse = ", "), "}")

read_rows <- function(name) read.csv(file.path(data_dir, name), comment.char = "#",
                                     stringsAsFactors = FALSE, colClasses = "character")
ints <- function(s) as.numeric(strsplit(s, " ", fixed = TRUE)[[1]])

# VineCopula family codes of the key (README): 90-degree rotations take -theta.
VC_CODE <- c(gaussian = 1, student = 2, clayton = 3, gumbel = 4, frank = 5, joe = 6, bb1 = 7)
vc_code <- function(family, rotation) {
  VC_CODE[[family]] + c("0" = 0, "180" = 10, "90" = 20, "270" = 30)[[as.character(rotation)]]
}
key_pars <- function(family, rotation, par, par2) {
  s <- if (rotation %in% c(90, 270)) -1 else 1
  if (family %in% c("student", "bb1")) s * c(par, par2) else s * par
}
ITAU <- c("gaussian", "student", "clayton", "gumbel", "frank")   # BiCopEst itau: 1-par and t

# ---------------------------------------------------------------------------
# Weighted estimation
# ---------------------------------------------------------------------------
wrows <- read_rows("weighted_data.csv")
kendall_lines <- character(0)
fit_lines <- character(0)
for (i in seq_len(nrow(wrows))) {
  r <- wrows[i, ]
  den <- as.numeric(r$denom)
  u <- ints(r$u_int) / den; v <- ints(r$v_int) / den
  w_gen <- ints(r$w_gen) / W_SCALE; w01 <- ints(r$w01)
  keep <- w01 > 0
  schemes <- list(unit = list(u = u, v = v, w = NA), gen = list(u = u, v = v, w = w_gen),
                  w01 = list(u = u, v = v, w = w01),
                  subset = list(u = u[keep], v = v[keep], w = NA))
  kt <- character(0)
  for (sn in names(schemes)) {
    s <- schemes[[sn]]
    ft <- VineCopula:::fasttau(s$u, s$v, s$w)
    tm <- if (any(is.na(s$w))) NA else TauMatrix(cbind(s$u, s$v), s$w)[2, 1]
    kt <- c(kt, paste0(jstr(sn), ": {\"fasttau\": ", jnum(ft), ", \"TauMatrix\": ", jnum(tm), "}"))
    if (r$dataset == "gaussian_ties") next
    code <- vc_code(r$family, as.numeric(r$rotation))
    for (method in c("itau", "mle")) {
      if (method == "itau" && !(r$family %in% ITAU)) next
      ww <- if (any(is.na(s$w))) rep(1, length(s$u)) else s$w
      fit <- suppressWarnings(BiCopEst(s$u, s$v, family = code, method = method, weights = s$w))
      logc <- log(BiCopPDF(s$u, s$v, code, fit$par, fit$par2, check.pars = FALSE))
      native <- sprintf("BiCopEst(family = %d, method = \"%s\"): par = %s, par2 = %s",
                        code, method, jnum(fit$par), jnum(fit$par2))
      fit_lines <- c(fit_lines, jline(c(
        dataset = jstr(r$dataset), scheme = jstr(sn), method = jstr(method),
        native = jstr(native), code = as.character(code),
        pars = jvec(key_pars(r$family, as.numeric(r$rotation), fit$par, fit$par2)),
        tau = jnum(BiCopPar2Tau(code, fit$par, fit$par2)),
        emptau = jnum(fit$emptau),
        loglik_reported = jnum(fit$logLik),
        loglik_weighted_own = jnum(sum(ww * logc)),
        loglik_unweighted_own = jnum(sum(logc)),
        sum_w = jnum(sum(ww)), nobs = jnum(fit$nobs))))
    }
  }
  kendall_lines <- c(kendall_lines, paste0("  ", jstr(r$dataset), ": {", paste(kt, collapse = ", "), "}"))
}

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
srows <- read_rows("selection_data.csv")
CRITS <- c("logLik", "AIC", "BIC")
sel_lines <- character(0)
for (i in seq_len(nrow(srows))) {
  r <- srows[i, ]
  n <- as.numeric(r$n)
  u <- seq_len(n) / (n + 1)
  v <- ints(r$v_by_u) / (n + 1)
  code <- vc_code(r$family, as.numeric(r$rotation))
  fams <- unique(c(0, 1, code))
  wts <- list(unit = NA, formula = ((7919 * seq_len(n)) %% 1024 + 1) / 1024)
  for (wn in names(wts)) {
    w <- wts[[wn]]
    ww <- if (any(is.na(w))) rep(1, n) else w
    fits <- character(0)
    lls <- numeric(0); npars <- numeric(0)
    for (f in fams) {
      if (f == 0) {
        par <- 0; par2 <- 0; ll <- 0; np <- 0
      } else {
        fit <- suppressWarnings(BiCopEst(u, v, family = f, method = "mle", weights = w))
        par <- fit$par; par2 <- fit$par2
        ll <- sum(log(BiCopPDF(u, v, f, par, par2, check.pars = FALSE)) * ww)
        np <- if (f %in% c(2, 7)) 2 else 1
      }
      lls <- c(lls, ll); npars <- c(npars, np)
      fits <- c(fits, paste0("{\"code\": ", f, ", \"par\": ", jnum(par), ", \"par2\": ", jnum(par2),
                             ", \"loglik\": ", jnum(ll), ", \"npars\": ", np, "}"))
    }
    crit_val <- list(logLik = -lls, AIC = -2 * lls + 2 * npars, BIC = -2 * lls + log(n) * npars)
    picked <- character(0)
    for (cr in CRITS) {
      sel <- suppressWarnings(BiCopSelect(u, v, familyset = fams, selectioncrit = cr, weights = w,
                                          rotations = FALSE, presel = FALSE, method = "mle"))
      mine <- fams[which.min(crit_val[[cr]])]
      if (sel$family != mine)
        stop(r$dataset, " seed ", r$seed, " ", wn, " ", cr, ": BiCopSelect picked ", sel$family,
             ", the recomputed criterion ", mine)
      picked <- c(picked, paste0(jstr(cr), ": ", sel$family))
    }
    sel_lines <- c(sel_lines, jline(c(
      dataset = jstr(r$dataset), seed = r$seed, weighting = jstr(wn), familyset = jvec(fams),
      selected = paste0("{", paste(picked, collapse = ", "), "}"),
      fits = paste0("[", paste(fits, collapse = ", "), "]"))))
  }
}

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
repo_version <- sub('.*__version__ = "([^"]+)".*', "\\1",
                    grep("__version__ =", readLines(file.path(root, "pmcprg", "__init__.py")),
                         value = TRUE)[1])
generated <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC")
md5 <- function(name) unname(tools::md5sum(file.path(data_dir, name)))
defs <- list(
  scheme = paste("unit: unweighted; gen: w = w_gen/2^20; w01: the {0,1} weights; subset:",
                 "unweighted on the points with w01 = 1"),
  kendall = paste("fasttau(u, v, weights) (VineCopula:::fasttau, BiCopEst's emptau): TauMatrix",
                  "with weights, the C ktau without; TauMatrix(cbind(u, v), weights)[2, 1]"),
  fit = "BiCopEst(u, v, family = code, method, weights) (weights = NA unweighted)",
  pars = paste("reference-agnostic key (README): [rho], [rho, nu], [theta] of the unrotated family",
               "(theta > 0; VineCopula's 90/270 codes take -theta), [theta, delta]"),
  tau = "BiCopPar2Tau(code, par, par2)",
  emptau = "the fitted object's emptau (the weighted Kendall tau of the data)",
  loglik_reported = "the fitted object's logLik",
  loglik_weighted_own = "sum(w * log(BiCopPDF(u, v, code, par, par2))) (w = 1 unweighted)",
  loglik_unweighted_own = "sum(log(BiCopPDF(u, v, code, par, par2)))",
  sum_w = "sum(w) (n unweighted)",
  selection = paste("BiCopSelect(u, v, familyset, selectioncrit, weights, rotations = FALSE,",
                    "presel = FALSE, method = 'mle'); weighting unit: weights = NA; formula:",
                    "w_i = ((7919 i) mod 1024 + 1)/1024 for the point of u-rank i;",
                    "fits: each candidate's BiCopEst(method = 'mle', weights) and",
                    "sum(log(BiCopPDF) * w), npars as BiCopSelect counts them;",
                    "AIC = -2 loglik + 2 npars, BIC = -2 loglik + log(n) npars, n = length(u)"))
header <- c(
  format = "1",
  about = jstr(paste0("FR-11 weighted-estimation and selection references, generated by ",
                      "scripts/parity/gen_r_weighted.R - do not edit by hand. Floats are %.17g ",
                      "decimal strings of IEEE-754 doubles; null = not recorded.")),
  reference = jobj(list(package = jstr("VineCopula"),
                        version = jstr(as.character(packageVersion("VineCopula")))), " "),
  environment = jobj(list(R = jstr(R.version.string), platform = jstr(R.version$platform),
                          VineCopula = jstr(as.character(packageVersion("VineCopula")))), " "),
  generated = jstr(generated),
  repository_version = jstr(repo_version),
  command = jstr("Rscript scripts/parity/gen_r_weighted.R"),
  data = jobj(list(weighted = jstr(md5("weighted_data.csv")),
                   selection = jstr(md5("selection_data.csv"))), " "),
  definitions = jobj(lapply(defs, jstr), " "),
  kendall = paste0("{\n", paste(kendall_lines, collapse = ",\n"), "\n }"))
txt <- paste0("{\n", paste0(" ", jstr(names(header)), ": ", header, collapse = ",\n"),
              ",\n \"fits\": [\n", paste(fit_lines, collapse = ",\n"), "\n ]",
              ",\n \"selection\": [\n", paste(sel_lines, collapse = ",\n"), "\n ]\n}\n")
fname <- file.path(data_dir, "weighted_vinecopula.json")
writeLines(txt, fname, sep = "")
cat(sprintf("wrote %s: %d fits, %d selection records\n", fname, length(fit_lines), length(sel_lines)))
