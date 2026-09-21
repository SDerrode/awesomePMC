#!/usr/bin/env Rscript
# Interior parity references from R: VineCopula and copula (audit FR-11).
#
# Evaluates pdf, cdf, both h-functions, both h-inverses, Kendall's tau and
# the two diagonal tail-dependence coefficients of the families of
# scripts/parity/cases.csv on the interior points of scripts/parity/points.csv,
# once with VineCopula and once with copula, and writes them, with their
# provenance, to pmcprg/tests/data/parity/vinecopula.json and
# pmcprg/tests/data/parity/rcopula.json. The tests
# (pmcprg/tests/test_parity_interior.py) read those files only: R is not
# needed at test time.
#
# Every convention the tests rely on -- which argument each h-function
# conditions on, which argument each inverse solves for, which argument each
# rotation reflects -- is measured here on each package and written to the
# file under "convention_checks"; the script stops if a check does not come
# out as documented in scripts/parity/README.md.
#
# Usage, from the repository root:
#
#     Rscript scripts/parity/gen_r.R
#
# Needs VineCopula and copula only (no JSON package: the files are written
# by hand, doubles as "%.17g", which identifies every IEEE-754 double).

suppressPackageStartupMessages({
  library(VineCopula)
  library(copula)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
root <- normalizePath(file.path(here, "..", ".."))
out_dir <- file.path(root, "pmcprg", "tests", "data", "parity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

points <- read.csv(file.path(here, "points.csv"))
U <- cbind(points$u, points$v)
cases <- read.csv(file.path(here, "cases.csv"), stringsAsFactors = FALSE)

# Families whose reference CDF is not recorded: pmcprg's Student copula has no
# CDF, and copula's pCopula refuses a non-integer df anyway.
NO_CDF <- c("student")

# Same thresholds as scripts/parity/gen_pyvinecopulib.py (see there).
FD_STEP <- 1e-5
FD_TOL <- 1e-4
ROUNDTRIP_TOL <- 1e-8

# ---------------------------------------------------------------------------
# JSON writing
# ---------------------------------------------------------------------------
# %.17g: 17 significant digits identify every IEEE-754 double for a
# correctly rounded reader (Python's float()). A shorter string cannot be
# chosen by reading it back in R: R's own strtod is not correctly rounded
# (as.numeric("4.724642376160763e-08") equals the double that Python reads
# from "4.7246423761607634e-08", Python reads it as its neighbour).
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
# VineCopula
# ---------------------------------------------------------------------------
VC_BASE <- c(gaussian = 1, student = 2, clayton = 3, gumbel = 4, frank = 5, joe = 6)
VC_ROT <- c("0" = 0, "180" = 10, "90" = 20, "270" = 30)

vc_spec <- function(family, rotation, par1, par2) {
  code <- VC_BASE[[family]] + VC_ROT[[as.character(rotation)]]
  # VineCopula parametrises its 90/270 rotations by a negative parameter.
  par <- if (rotation %in% c(90, 270)) -par1 else par1
  list(code = code, par = par, par2 = if (is.na(par2)) 0 else par2)
}

vc_eval <- function(s, family) {
  f <- function(fun, M) fun(M[, 1], M[, 2], s$code, s$par, s$par2)
  td <- BiCopPar2TailDep(s$code, s$par, s$par2)
  list(
    pdf = f(BiCopPDF, U),
    cdf = if (family %in% NO_CDF) NULL else f(BiCopCDF, U),
    h1 = f(BiCopHfunc1, U), h2 = f(BiCopHfunc2, U),
    hinv1 = f(BiCopHinv1, U), hinv2 = f(BiCopHinv2, U),
    tau = BiCopPar2Tau(s$code, s$par, s$par2),
    lambda_L = td$lower, lambda_U = td$upper
  )
}

vc_checks <- function(s, family) {
  f <- function(fun, M) fun(M[, 1], M[, 2], s$code, s$par, s$par2)
  e1 <- cbind(FD_STEP, 0); e2 <- cbind(0, FD_STEP)
  sh <- function(M, e) M + e[rep(1, nrow(M)), ]
  out <- list()
  if (!(family %in% NO_CDF)) {
    out$h1_minus_dC_du1 <- abs(f(BiCopHfunc1, U) -
      (f(BiCopCDF, sh(U, e1)) - f(BiCopCDF, sh(U, -e1))) / (2 * FD_STEP))
    out$h2_minus_dC_du2 <- abs(f(BiCopHfunc2, U) -
      (f(BiCopCDF, sh(U, e2)) - f(BiCopCDF, sh(U, -e2))) / (2 * FD_STEP))
  }
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

# ---------------------------------------------------------------------------
# copula
# ---------------------------------------------------------------------------
cop_base <- function(family, par1, par2) {
  switch(family,
    gaussian = normalCopula(par1),
    student = tCopula(par1, df = par2, df.fixed = TRUE),
    clayton = claytonCopula(par1),
    gumbel = gumbelCopula(par1),
    frank = frankCopula(par1),
    joe = joeCopula(par1))
}
COP_FLIP <- list("180" = c(TRUE, TRUE), "90" = c(TRUE, FALSE), "270" = c(FALSE, TRUE))

cop_spec <- function(family, rotation, par1, par2) {
  base <- cop_base(family, par1, par2)
  base_txt <- switch(family,
    gaussian = sprintf("normalCopula(%s)", jnum(par1)),
    student = sprintf("tCopula(%s, df = %s, df.fixed = TRUE)", jnum(par1), jnum(par2)),
    sprintf("%sCopula(%s)", family, jnum(par1)))
  if (rotation == 0) return(list(cop = base, base = base, txt = base_txt))
  flip <- COP_FLIP[[as.character(rotation)]]
  list(cop = rotCopula(base, flip = flip), base = base,
       txt = sprintf("rotCopula(%s, flip = c(%s))", base_txt,
                     paste(flip, collapse = ", ")))
}

# h-functions: cCopula(u, cop, indices = 2) is C(u2 | u1) = dC/du1, and its
# inverse = TRUE solves for u2. The rotated families get no h-values from
# copula 1.1-7: cCopula(inverse = TRUE) is not implemented for rotCopula
# ("Not yet implemented for copula class rotExplicitCopula"), and the forward
# cCopula of a rotCopula returns the base's value at the flipped point
# without the "1 -" a flipped second margin needs (its rosenblatt() source)
# -- a valid Rosenblatt transform of the flipped vector, not dC/du1 (Clayton
# theta = 2, flip = c(FALSE, TRUE), (0.3, 0.8): 0.178 against dC/du = 0.822).
# h2 and hinv2 of the unrotated families -- all exchangeable -- are h1 and
# hinv1 at the swapped point: h2(u, v) = h1(v, u), and h2(x, v) = u <=>
# h1(v, x) = u.
#
# The inverse is closed-form for Gauss, t and Clayton; for every other
# Archimedean family copula's iRosenblatt() solves h1 = w by uniroot() on
# [0, 1] with uniroot's default tol = .Machine$double.eps^0.25 = 1.2e-4 (its
# source, copula 1.1-7). That default leaves a median round-trip error
# h1(u, hinv1) - w of 3.3e-5 on this grid (Frank), which would say nothing
# about pmcprg; iRosenblatt() passes '...' on to uniroot(), so HINV_TOL is
# handed over instead. It is ignored by the closed forms.
HINV_TOL <- 1e-15
cop_hinv <- function(M, cop) as.numeric(cCopula(M, cop, indices = 2, inverse = TRUE,
                                                tol = HINV_TOL))

cop_eval <- function(s, family, rotation) {
  cop <- s$cop
  has_h <- rotation == 0
  lam <- if (rotation == 0) lambda(cop) else c(lower = NA, upper = NA)
  Us <- U[, 2:1]
  list(
    pdf = dCopula(U, cop),
    cdf = if (family %in% NO_CDF) NULL else pCopula(U, cop),
    h1 = if (has_h) as.numeric(cCopula(U, cop, indices = 2)) else NULL,
    h2 = if (has_h) as.numeric(cCopula(Us, cop, indices = 2)) else NULL,
    hinv1 = if (has_h) cop_hinv(U, cop) else NULL,
    hinv2 = if (has_h) cop_hinv(Us, cop) else NULL,
    tau = tau(cop),
    lambda_L = unname(lam[1]), lambda_U = unname(lam[2])
  )
}

cop_checks <- function(s, family, rotation) {
  cop <- s$cop
  e1 <- cbind(FD_STEP, 0); e2 <- cbind(0, FD_STEP)
  sh <- function(M, e) M + e[rep(1, nrow(M)), ]
  out <- list()
  if (rotation != 0) return(out)
  h1 <- function(M) as.numeric(cCopula(M, cop, indices = 2))
  if (!(family %in% NO_CDF)) {
    out$h1_minus_dC_du1 <- abs(h1(U) -
      (pCopula(sh(U, e1), cop) - pCopula(sh(U, -e1), cop)) / (2 * FD_STEP))
  }
  pdf <- dCopula(U, cop)
  out$pdf_minus_dh1_du2 <- abs(pdf - (h1(sh(U, e2)) - h1(sh(U, -e2))) / (2 * FD_STEP)) /
                           pmax(pdf, 1)
  y <- cop_hinv(U, cop)
  out$h1_of_hinv1_minus_level <- abs(h1(cbind(U[, 1], y)) - U[, 2])
  out
}

# ---------------------------------------------------------------------------
# Rotation conventions, measured on each package's densities
# ---------------------------------------------------------------------------
REFLECTIONS <- list(
  "c0(1-u, v)" = function(M) cbind(1 - M[, 1], M[, 2]),
  "c0(u, 1-v)" = function(M) cbind(M[, 1], 1 - M[, 2]),
  "c0(1-u, 1-v)" = function(M) cbind(1 - M[, 1], 1 - M[, 2]))

# Judged on the median relative error over the points, like the other checks:
# copula evaluates a rotated Clayton/Gumbel/Joe (class rotExplicitCopula) from
# its symbolic density, not by calling the base density at the reflected
# point, and that expression is off by up to 1.5e-3 at one point (Joe,
# theta = 7, (0.01, 0.01)) -- a reference inaccuracy the parity tests
# measure. The wrong reflections are off by O(1) at most points.
which_reflection <- function(pdf_rot, pdf_base_fun, label) {
  rel <- lapply(REFLECTIONS, function(r) abs(pdf_base_fun(r(U)) - pdf_rot) / pdf_rot)
  med <- vapply(rel, median, 0)
  hit <- names(med)[med < 1e-9]
  if (length(hit) != 1) stop(label, ": no unique reflection matches (",
                             paste(names(med), signif(med, 3), collapse = "; "), ")")
  list(density_equals = hit, rel_error = max(rel[[hit]]), rel_error_median = med[[hit]])
}

# A convention is judged on the median over the points: a wrong convention
# (h1 for h2, a flipped argument) is off by O(0.1) almost everywhere, while a
# reference's local inaccuracy -- measured by the parity tests themselves --
# shows at one or two points. The maximum is recorded as well.
check_tol <- function(checks, label) {
  for (k in names(checks)) {
    tol <- if (grepl("hinv", k)) ROUNDTRIP_TOL else FD_TOL
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

run <- function(pkg) {
  entries <- character(0)
  per_family <- list()
  rotations <- list()
  for (i in seq_len(nrow(cases))) {
    fam <- cases$family[i]; rot <- cases$rotation[i]
    p1 <- cases$par1[i]; p2 <- cases$par2[i]
    pars <- if (is.na(p2)) p1 else c(p1, p2)
    key <- paste0(fam, "/", rot)
    if (pkg == "VineCopula") {
      s <- vc_spec(fam, rot, p1, p2)
      val <- vc_eval(s, fam)
      native <- sprintf("family = %d, par = %s, par2 = %s", s$code, jnum(s$par), jnum(s$par2))
      chk <- vc_checks(s, fam)
      if (rot != 0) {
        b <- vc_spec(fam, 0, p1, p2)
        rc <- which_reflection(val$pdf, function(M) BiCopPDF(M[, 1], M[, 2], b$code, b$par, b$par2),
                               key)
        rc$tau_rotated_over_tau_base <- val$tau / BiCopPar2Tau(b$code, b$par, b$par2)
      }
    } else {
      s <- cop_spec(fam, rot, p1, p2)
      val <- cop_eval(s, fam, rot)
      native <- s$txt
      chk <- cop_checks(s, fam, rot)
      if (rot != 0) {
        rc <- which_reflection(val$pdf, function(M) dCopula(M, s$base), key)
        rc$tau_rotated_over_tau_base <- val$tau / tau(s$base)
      }
    }
    check_tol(chk, paste(pkg, key))
    for (k in names(chk)) {
      kmax <- paste0(k, "_max"); kmed <- paste0(k, "_median")
      per_family[[key]][[kmax]] <- max(c(per_family[[key]][[kmax]], chk[[k]]))
      per_family[[key]][[kmed]] <- max(c(per_family[[key]][[kmed]], median(chk[[k]])))
    }
    if (rot != 0) {
      prev <- rotations[[key]]
      if (!is.null(prev) && prev$density_equals != rc$density_equals)
        stop(key, ": the reflection depends on the parameter")
      if (is.null(prev) || rc$rel_error > prev$rel_error) rotations[[key]] <- rc
    }
    fields <- c(
      family = jstr(fam), rotation = as.character(rot), pars = jvec(pars),
      native = jstr(native),
      pdf = jvec(val$pdf), cdf = jvec(val$cdf), h1 = jvec(val$h1), h2 = jvec(val$h2),
      hinv1 = jvec(val$hinv1), hinv2 = jvec(val$hinv2),
      tau = jnum(val$tau), lambda_L = jnum(val$lambda_L), lambda_U = jnum(val$lambda_U))
    entries <- c(entries, paste0("  {", paste0(jstr(names(fields)), ": ", fields, collapse = ", "),
                                 "}"))
  }

  pf <- vapply(names(per_family), function(k)
    paste0("   ", jstr(k), ": ", jobj(lapply(per_family[[k]], jnum), "   ")), "")
  ro <- vapply(names(rotations), function(k) {
    r <- rotations[[k]]
    paste0("   ", jstr(k), ": ", jobj(list(density_equals = jstr(r$density_equals),
                                           rel_error_max = jnum(r$rel_error),
                                           rel_error_median = jnum(r$rel_error_median),
                                           tau_rotated_over_tau_base = jnum(r$tau_rotated_over_tau_base)),
                                      "   "))
  }, "")
  defs <- if (pkg == "VineCopula") list(
    h1 = "BiCopHfunc1 = P(U2 <= v | U1 = u) = dC/du",
    h2 = "BiCopHfunc2 = P(U1 <= u | U2 = v) = dC/dv",
    hinv1 = "BiCopHinv1: the y with h1(u, y) = v (inverse in the second argument)",
    hinv2 = "BiCopHinv2: the x with h2(x, v) = u (inverse in the first argument)",
    tau = "BiCopPar2Tau", lambda = "BiCopPar2TailDep: lower, upper",
    pars = paste("reference-agnostic key: gaussian [rho]; student [rho, nu]; others [theta] of",
                 "the unrotated family, theta > 0 for every rotation; native = the VineCopula call",
                 "(family code 1-6, +10 for 180, +20 for 90, +30 for 270; negative par for 90/270)")
  ) else list(
    h1 = "cCopula(cbind(u, v), cop, indices = 2) = P(U2 <= v | U1 = u) = dC/du; unrotated only",
    h2 = "cCopula(cbind(v, u), cop, indices = 2) = h1(v, u) = dC/dv for these exchangeable families; unrotated only",
    hinv1 = "cCopula(cbind(u, v), cop, indices = 2, inverse = TRUE, tol = 1e-15): the y with h1(u, y) = v; unrotated only; tol reaches uniroot() for gumbel/frank/joe",
    hinv2 = "cCopula(cbind(v, u), cop, indices = 2, inverse = TRUE, tol = 1e-15): the x with h2(x, v) = u; unrotated only",
    tau = "tau(cop)", lambda = "lambda(cop): lower, upper; unrotated only (not implemented for rotCopula)",
    pars = paste("reference-agnostic key: gaussian [rho]; student [rho, nu]; others [theta] of",
                 "the unrotated family, theta > 0 for every rotation; native = the copula object")
  )
  header <- c(
    format = "1",
    about = jstr(paste0("FR-11 interior parity references, generated by scripts/parity/gen_r.R - ",
                        "do not edit by hand. Floats are %.17g decimal strings of IEEE-754 doubles; ",
                        "null = not recorded.")),
    reference = jobj(list(package = jstr(pkg), version = jstr(as.character(packageVersion(pkg)))), " "),
    environment = jobj(list(R = jstr(R.version.string), platform = jstr(R.version$platform),
                            VineCopula = jstr(as.character(packageVersion("VineCopula"))),
                            copula = jstr(as.character(packageVersion("copula")))), " "),
    generated = jstr(generated),
    repository_version = jstr(repo_version),
    command = jstr("Rscript scripts/parity/gen_r.R"),
    definitions = jobj(lapply(defs, jstr), " "),
    convention_checks = paste0("{\n", "  \"fd_step\": ", jnum(FD_STEP),
                               ",\n  \"fd_tolerance\": ", jnum(FD_TOL),
                               ",\n  \"roundtrip_tolerance\": ", jnum(ROUNDTRIP_TOL),
                               ",\n  \"judged_on\": \"median over the points (max recorded); ",
                               "pdf errors relative above pdf = 1, absolute below\"",
                               ",\n  \"per_family\": {\n", paste(pf, collapse = ",\n"), "\n  }",
                               ",\n  \"rotations\": {\n", paste(ro, collapse = ",\n"), "\n  }\n }"),
    points = paste0("[", paste(apply(U, 1, jvec), collapse = ", "), "]"))
  txt <- paste0("{\n", paste0(" ", jstr(names(header)), ": ", header, collapse = ",\n"),
                ",\n \"cases\": [\n", paste(entries, collapse = ",\n"), "\n ]\n}\n")
  fname <- file.path(out_dir, if (pkg == "VineCopula") "vinecopula.json" else "rcopula.json")
  writeLines(txt, fname, sep = "")
  cat(sprintf("wrote %s: %d cases x %d points\n", fname, nrow(cases), nrow(U)))
  for (k in names(rotations)) cat(sprintf("  %s: density = %s, tau ratio %+.1f\n", k,
                                          rotations[[k]]$density_equals,
                                          rotations[[k]]$tau_rotated_over_tau_base))
}

run("VineCopula")
run("copula")
