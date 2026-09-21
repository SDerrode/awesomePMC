#!/usr/bin/env Rscript
# Interior parity references from R VineCopula for Joe's BB1, BB6, BB7 and
# BB8 (audit FR-11, wave 2).
#
# Evaluates pdf, cdf, both h-functions, both h-inverses, Kendall's tau and
# the two diagonal tail-dependence coefficients of the cases of
# scripts/parity/cases_bb.csv -- BB1 in its four rotations, BB6, BB7 and BB8
# unrotated (pmcprg registers no rotation of the last three) -- on the
# interior points of scripts/parity/points.csv, and writes them, with their
# provenance, to pmcprg/tests/data/parity/bb_vinecopula.json, read by
# pmcprg/tests/test_parity_bb.py. R is not needed at test time.
#
# R's copula, copBasic and fCopulae have none of the four families (copBasic
# has Joe's BB4 only, JOcopBB4): VineCopula is the one R reference, and the
# tests' mpmath oracle the independent one.
#
# Which native parameter is which is *measured*, not assumed: VineCopula's
# CDF is compared with the textbook formula of the reference-agnostic key
# [theta, delta] (Joe 1997, ch. 5: the tb_* functions below, the formulas of
# scripts/parity/README.md) under both orders of the two parameters and, for
# the 90/270 rotations, both signs; the one that matches is recorded under
# "convention_checks". The rotations and the h-function conventions are
# measured as in gen_r.R. The script stops if a check fails.
#
# Usage, from the repository root:
#
#     Rscript scripts/parity/gen_r_bb.R
#
# Needs VineCopula only (no JSON package: the file is written by hand,
# doubles as "%.17g").

suppressPackageStartupMessages(library(VineCopula))

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
root <- normalizePath(file.path(here, "..", ".."))
out_dir <- file.path(root, "pmcprg", "tests", "data", "parity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

points <- read.csv(file.path(here, "points.csv"))
U <- cbind(points$u, points$v)
cases <- read.csv(file.path(here, "cases_bb.csv"), stringsAsFactors = FALSE)

# Same thresholds as gen_r.R, gen_r_extra.R and the Python generators.
FD_STEP <- 1e-5
FD_TOL <- 1e-4
ROUNDTRIP_TOL <- 1e-8
# The swapped order or the other sign, where VineCopula accepts it at all, is
# off by O(0.01) to O(1) at most points.
PARAM_TOL <- 1e-9

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

# ---------------------------------------------------------------------------
# Textbook CDFs of the key (Joe 1997, ch. 5; README), [theta, delta]
# ---------------------------------------------------------------------------
tb_base <- function(family, th, de) {
  switch(family,
    bb1 = function(u, v) (1 + ((u^-th - 1)^de + (v^-th - 1)^de)^(1 / de))^(-1 / th),
    bb6 = function(u, v) {
      p <- -log(1 - (1 - u)^th); q <- -log(1 - (1 - v)^th)
      1 - (1 - exp(-(p^de + q^de)^(1 / de)))^(1 / th)
    },
    bb7 = function(u, v) {
      a <- 1 - (1 - u)^th; b <- 1 - (1 - v)^th
      1 - (1 - (a^-de + b^-de - 1)^(-1 / de))^(1 / th)
    },
    bb8 = function(u, v) {
      eta <- 1 - (1 - de)^th
      A <- (1 - (1 - de * u)^th) * (1 - (1 - de * v)^th) / eta
      (1 - (1 - A)^(1 / th)) / de
    })
}
tb_cdf <- function(family, rotation, th, de) {
  C <- tb_base(family, th, de)
  switch(as.character(rotation),
    "0" = C,
    "90" = function(u, v) v - C(1 - u, v),
    "180" = function(u, v) u + v - 1 + C(1 - u, 1 - v),
    "270" = function(u, v) u - C(u, 1 - v))
}

# ---------------------------------------------------------------------------
# VineCopula: family codes 7-10, +10 for 180, +20 for 90, +30 for 270
# ---------------------------------------------------------------------------
VC_BASE <- c(bb1 = 7, bb6 = 8, bb7 = 9, bb8 = 10)
VC_ROT <- c("0" = 0, "180" = 10, "90" = 20, "270" = 30)

# The (par, par2) whose CDF is the textbook one: each order, and for 90/270
# each sign (the pilot's one-parameter rotations take a negative parameter).
vc_spec <- function(family, rotation, th, de) {
  code <- VC_BASE[[family]] + VC_ROT[[as.character(rotation)]]
  tb <- tb_cdf(family, rotation, th, de)(U[, 1], U[, 2])
  signs <- if (rotation %in% c(90, 270)) c(1, -1) else 1
  hits <- list()
  for (s in signs) for (ord in list(c(th, de), c(de, th))) {
    cdf <- tryCatch(suppressWarnings(BiCopCDF(U[, 1], U[, 2], code, s * ord[1], s * ord[2])),
                    error = function(e) NULL)
    if (is.null(cdf) || any(!is.finite(cdf))) next
    err <- median(abs(cdf - tb))
    if (err < PARAM_TOL)
      hits[[length(hits) + 1]] <- list(code = code, par = s * ord[1], par2 = s * ord[2],
        order = if (identical(ord, c(th, de))) "par = theta, par2 = delta" else "par = delta, par2 = theta",
        sign = if (s > 0) "positive" else "negative")
  }
  if (length(hits) != 1) stop(family, "/", rotation, " ", th, ", ", de, ": no unique native parameters match")
  hits[[1]]
}

vc_f <- function(s) function(fun, M) fun(M[, 1], M[, 2], s$code, s$par, s$par2)

vc_eval <- function(s) {
  f <- vc_f(s)
  td <- BiCopPar2TailDep(s$code, s$par, s$par2)
  list(pdf = f(BiCopPDF, U), cdf = f(BiCopCDF, U),
       h1 = f(BiCopHfunc1, U), h2 = f(BiCopHfunc2, U),
       hinv1 = f(BiCopHinv1, U), hinv2 = f(BiCopHinv2, U),
       tau = BiCopPar2Tau(s$code, s$par, s$par2),
       lambda_L = td$lower, lambda_U = td$upper)
}

# The h-function conventions, as gen_r.R (finite differences and round trips).
vc_checks <- function(s, family, rotation, th, de) {
  f <- vc_f(s)
  e1 <- cbind(FD_STEP, 0); e2 <- cbind(0, FD_STEP)
  sh <- function(M, e) M + e[rep(1, nrow(M)), ]
  out <- list()
  out$param_cdf_abs_error <- abs(f(BiCopCDF, U) - tb_cdf(family, rotation, th, de)(U[, 1], U[, 2]))
  out$h1_minus_dC_du1 <- abs(f(BiCopHfunc1, U) -
    (f(BiCopCDF, sh(U, e1)) - f(BiCopCDF, sh(U, -e1))) / (2 * FD_STEP))
  out$h2_minus_dC_du2 <- abs(f(BiCopHfunc2, U) -
    (f(BiCopCDF, sh(U, e2)) - f(BiCopCDF, sh(U, -e2))) / (2 * FD_STEP))
  pdf <- f(BiCopPDF, U)
  scale <- pmax(pdf, 1)
  out$pdf_minus_dh1_du2 <- abs(pdf -
    (f(BiCopHfunc1, sh(U, e2)) - f(BiCopHfunc1, sh(U, -e2))) / (2 * FD_STEP)) / scale
  out$pdf_minus_dh2_du1 <- abs(pdf -
    (f(BiCopHfunc2, sh(U, e1)) - f(BiCopHfunc2, sh(U, -e1))) / (2 * FD_STEP)) / scale
  y <- f(BiCopHinv1, U)
  out$h1_of_hinv1_minus_level <- abs(f(BiCopHfunc1, cbind(U[, 1], y)) - U[, 2])
  x <- f(BiCopHinv2, U)
  out$h2_of_hinv2_minus_level <- abs(f(BiCopHfunc2, cbind(x, U[, 2])) - U[, 1])
  out
}

REFLECTIONS <- list(
  "c0(1-u, v)" = function(M) cbind(1 - M[, 1], M[, 2]),
  "c0(u, 1-v)" = function(M) cbind(M[, 1], 1 - M[, 2]),
  "c0(1-u, 1-v)" = function(M) cbind(1 - M[, 1], 1 - M[, 2]))

which_reflection <- function(pdf_rot, pdf_base_fun, label) {
  rel <- lapply(REFLECTIONS, function(r) abs(pdf_base_fun(r(U)) - pdf_rot) / pdf_rot)
  med <- vapply(rel, median, 0)
  hit <- names(med)[med < 1e-9]
  if (length(hit) != 1) stop(label, ": no unique reflection matches (",
                             paste(names(med), signif(med, 3), collapse = "; "), ")")
  list(density_equals = hit, rel_error = max(rel[[hit]]), rel_error_median = med[[hit]])
}

check_tol <- function(checks, label) {
  for (k in names(checks)) {
    tol <- if (grepl("hinv", k)) ROUNDTRIP_TOL else if (grepl("^param", k)) PARAM_TOL else FD_TOL
    med <- median(checks[[k]])
    if (!(med < tol)) stop("convention check failed for ", label, ": median ", k, " = ",
                           signif(med, 3))
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
repo_version <- sub('.*__version__ = "([^"]+)".*', "\\1",
                    grep("__version__ =", readLines(file.path(root, "pmcprg", "__init__.py")),
                         value = TRUE)[1])
generated <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC")

entries <- character(0)
per_family <- list()
rotations <- list()
orders <- list()
signs <- list()
for (i in seq_len(nrow(cases))) {
  fam <- cases$family[i]; rot <- cases$rotation[i]
  th <- cases$par1[i]; de <- cases$par2[i]
  key <- paste0(fam, "/", rot)
  s <- vc_spec(fam, rot, th, de)
  if (!is.null(orders[[fam]]) && orders[[fam]] != s$order) stop(fam, ": the order depends on the case")
  orders[[fam]] <- s$order
  if (rot != 0) signs[[key]] <- s$sign
  val <- vc_eval(s)
  chk <- vc_checks(s, fam, rot, th, de)
  check_tol(chk, key)
  for (k in names(chk)) {
    kmax <- paste0(k, "_max"); kmed <- paste0(k, "_median")
    per_family[[key]][[kmax]] <- max(c(per_family[[key]][[kmax]], chk[[k]]))
    per_family[[key]][[kmed]] <- max(c(per_family[[key]][[kmed]], median(chk[[k]])))
  }
  if (rot != 0) {
    b <- vc_spec(fam, 0, th, de)
    rc <- which_reflection(val$pdf, function(M) BiCopPDF(M[, 1], M[, 2], b$code, b$par, b$par2), key)
    rc$tau_rotated_over_tau_base <- val$tau / BiCopPar2Tau(b$code, b$par, b$par2)
    prev <- rotations[[key]]
    if (!is.null(prev) && prev$density_equals != rc$density_equals)
      stop(key, ": the reflection depends on the parameter")
    if (is.null(prev) || rc$rel_error > prev$rel_error) rotations[[key]] <- rc
  }
  native <- sprintf("family = %d, par = %s, par2 = %s", s$code, jnum(s$par), jnum(s$par2))
  fields <- c(
    family = jstr(fam), rotation = as.character(rot), pars = jvec(c(th, de)),
    native = jstr(native),
    pdf = jvec(val$pdf), cdf = jvec(val$cdf), h1 = jvec(val$h1), h2 = jvec(val$h2),
    hinv1 = jvec(val$hinv1), hinv2 = jvec(val$hinv2),
    tau = jnum(val$tau), lambda_L = jnum(val$lambda_L), lambda_U = jnum(val$lambda_U))
  entries <- c(entries, paste0("  {", paste0(jstr(names(fields)), ": ", fields, collapse = ", "), "}"))
}

pf <- vapply(names(per_family), function(k)
  paste0("   ", jstr(k), ": ", jobj(lapply(per_family[[k]], jnum), "   ")), "")
ro <- vapply(names(rotations), function(k) {
  r <- rotations[[k]]
  paste0("   ", jstr(k), ": ", jobj(list(density_equals = jstr(r$density_equals),
                                         rel_error_max = jnum(r$rel_error),
                                         rel_error_median = jnum(r$rel_error_median),
                                         tau_rotated_over_tau_base = jnum(r$tau_rotated_over_tau_base),
                                         parameter_sign = jstr(signs[[k]])),
                                    "   "))
}, "")
defs <- list(
  pdf = "BiCopPDF", cdf = "BiCopCDF",
  h1 = "BiCopHfunc1 = P(U2 <= v | U1 = u) = dC/du",
  h2 = "BiCopHfunc2 = P(U1 <= u | U2 = v) = dC/dv",
  hinv1 = "BiCopHinv1: the y with h1(u, y) = v (inverse in the second argument)",
  hinv2 = "BiCopHinv2: the x with h2(x, v) = u (inverse in the first argument)",
  tau = "BiCopPar2Tau", lambda = "BiCopPar2TailDep: lower, upper",
  pars = paste("reference-agnostic key (README): [theta, delta] of the unrotated family (Joe 1997,",
               "ch. 5), positive for every rotation; native = the VineCopula call (codes 7-10,",
               "+10 for 180, +20 for 90, +30 for 270; both parameters negated for 90/270, measured)"))
header <- c(
  format = "1",
  about = jstr(paste0("FR-11 interior parity references (BB1, BB6, BB7, BB8), generated by ",
                      "scripts/parity/gen_r_bb.R - do not edit by hand. Floats are %.17g decimal ",
                      "strings of IEEE-754 doubles; null = not recorded.")),
  reference = jobj(list(package = jstr("VineCopula"),
                        version = jstr(as.character(packageVersion("VineCopula")))), " "),
  environment = jobj(list(R = jstr(R.version.string), platform = jstr(R.version$platform),
                          VineCopula = jstr(as.character(packageVersion("VineCopula")))), " "),
  generated = jstr(generated),
  repository_version = jstr(repo_version),
  command = jstr("Rscript scripts/parity/gen_r_bb.R"),
  definitions = jobj(lapply(defs, jstr), " "),
  convention_checks = paste0("{\n", "  \"fd_step\": ", jnum(FD_STEP),
                             ",\n  \"fd_tolerance\": ", jnum(FD_TOL),
                             ",\n  \"roundtrip_tolerance\": ", jnum(ROUNDTRIP_TOL),
                             ",\n  \"param_tolerance\": ", jnum(PARAM_TOL),
                             ",\n  \"judged_on\": \"median over the points (max recorded); ",
                             "param_* absolute against the textbook CDF; ",
                             "pdf errors relative above pdf = 1, absolute below\"",
                             ",\n  \"per_family\": {\n", paste(pf, collapse = ",\n"), "\n  }",
                             ",\n  \"rotations\": {\n", paste(ro, collapse = ",\n"), "\n  }",
                             ",\n  \"parameter_order\": ", jobj(lapply(orders, jstr), "  "), "\n }"),
  points = paste0("[", paste(apply(U, 1, jvec), collapse = ", "), "]"),
  grid = paste0("[", paste(vapply(seq_len(nrow(cases)), function(i)
    paste0("[", jstr(cases$family[i]), ", ", cases$rotation[i], ", ",
           jvec(c(cases$par1[i], cases$par2[i])), "]"), ""), collapse = ", "), "]"))
txt <- paste0("{\n", paste0(" ", jstr(names(header)), ": ", header, collapse = ",\n"),
              ",\n \"cases\": [\n", paste(entries, collapse = ",\n"), "\n ]\n}\n")
fname <- file.path(out_dir, "bb_vinecopula.json")
writeLines(txt, fname, sep = "")
cat(sprintf("wrote %s: %d cases x %d points\n", fname, length(entries), nrow(U)))
for (k in names(orders)) cat(sprintf("  %s: %s\n", k, orders[[k]]))
for (k in names(rotations)) cat(sprintf("  %s: density = %s, tau ratio %+.1f, %s parameters\n", k,
                                        rotations[[k]]$density_equals,
                                        rotations[[k]]$tau_rotated_over_tau_base, signs[[k]]))
