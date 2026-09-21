#!/usr/bin/env Rscript
# Interior parity references from R for the families beyond the pilot
# (audit FR-11, wave 1): the extreme-value families (Galambos, Hüsler-Reiss,
# t-EV, the asymmetric logistic Tawn model), Plackett, AMH, FGM, and Nelsen's
# Archimedean families 4.2.12 and 4.2.14 with their 90/270 rotations.
#
# Evaluates, wherever a package provides them, pdf, cdf, both h-functions,
# both h-inverses, Kendall's tau, the two diagonal tail-dependence
# coefficients and the Pickands dependence function A(t) of the cases of
# scripts/parity/cases_extra.csv on the interior points of
# scripts/parity/points.csv (A on scripts/parity/pickands_points.csv), with
# four packages, and writes one file per package to pmcprg/tests/data/parity/:
#
#     extra_vinecopula.json  VineCopula  -- Tawn types 1 and 2 (families 104, 204)
#     extra_rcopula.json     copula      -- EV families, Tawn (Khoudraji device),
#                                           Plackett, AMH, FGM
#     extra_copbasic.json    copBasic    -- the CDFs (closed forms)
#     extra_fcopulae.json    fCopulae    -- EV families, AMH, A12, A14
#
# The tests (pmcprg/tests/test_parity_extra.py) read those files only: R is
# not needed at test time.
#
# Every parametrisation the tests rely on is *measured* here on each package,
# not assumed: each package's CDF is compared with the textbook formula of
# the reference-agnostic key (the tb_* functions below, the formulas of
# scripts/parity/README.md), which fixes which native parameter is which --
# in particular which Tawn weight VineCopula's family 104 frees -- and which
# argument copBasic's "acute"/"grave" reflections reflect. The results are
# written under "convention_checks"; the script stops if a check fails.
#
# Usage, from the repository root:
#
#     Rscript scripts/parity/gen_r_extra.R
#
# Needs VineCopula, copula, copBasic and fCopulae (no JSON package: the files
# are written by hand, doubles as "%.17g").

suppressPackageStartupMessages({
  library(VineCopula)
  library(copula)
  library(copBasic)
  library(fCopulae)
})

file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", file_arg)))
root <- normalizePath(file.path(here, "..", ".."))
out_dir <- file.path(root, "pmcprg", "tests", "data", "parity")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

points <- read.csv(file.path(here, "points.csv"))
U <- cbind(points$u, points$v)
TGRID <- read.csv(file.path(here, "pickands_points.csv"))$t
cases <- read.csv(file.path(here, "cases_extra.csv"), stringsAsFactors = FALSE)
case_pars <- function(i) {
  p <- c(cases$par1[i], cases$par2[i], cases$par3[i])
  p[!is.na(p)]
}

# Same thresholds as gen_r.R and gen_pyvinecopulib.py (see there).
FD_STEP <- 1e-5
FD_TOL <- 1e-4
ROUNDTRIP_TOL <- 1e-8
# A parametrisation is accepted when the package's CDF equals the textbook
# formula to this median absolute error over the points (relative for A: the
# rotated CDFs v - C(1 - u, v) cancel to 0 in both where they are tiny); a
# wrong mapping (a swapped weight, a reciprocal parameter, the other
# reflection) is off by O(0.001) to O(1) at most points.
PARAM_TOL <- 1e-9
# The second mixed difference of the CDF (pdf check of the packages that
# have no h-function): step and median tolerance (relative above pdf = 1).
FD2_STEP <- 1e-4
# copula inverts h by uniroot(); see gen_r.R (tol = 1e-15 handed over).
HINV_TOL <- 1e-15

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
# Textbook CDFs of the reference-agnostic key (README), w = -log u, z = -log v
# ---------------------------------------------------------------------------
tb_base <- function(family, p) {
  switch(family,
    galambos = function(u, v) {
      w <- -log(u); z <- -log(v)
      u * v * exp((w^(-p[1]) + z^(-p[1]))^(-1 / p[1]))
    },
    husler_reiss = function(u, v) {
      w <- -log(u); z <- -log(v); l <- p[1]
      exp(-w * pnorm(1 / l + l / 2 * log(w / z)) - z * pnorm(1 / l + l / 2 * log(z / w)))
    },
    tev = function(u, v) {
      w <- -log(u); z <- -log(v); rho <- p[1]; nu <- p[2]
      k <- sqrt((nu + 1) / (1 - rho^2))
      exp(-(w * pt(k * ((w / z)^(1 / nu) - rho), nu + 1) +
            z * pt(k * ((z / w)^(1 / nu) - rho), nu + 1)))
    },
    tawn = function(u, v) {      # p = (theta, psi_u, psi_v)
      w <- -log(u); z <- -log(v); th <- p[1]; pu <- p[2]; pv <- p[3]
      exp(-((1 - pu) * w + (1 - pv) * z + ((pu * w)^th + (pv * z)^th)^(1 / th)))
    },
    plackett = function(u, v) {
      th <- p[1]; S <- 1 + (th - 1) * (u + v)
      (S - sqrt(S^2 - 4 * th * (th - 1) * u * v)) / (2 * (th - 1))
    },
    amh = function(u, v) u * v / (1 - p[1] * (1 - u) * (1 - v)),
    fgm = function(u, v) u * v * (1 + p[1] * (1 - u) * (1 - v)),
    a12 = function(u, v) 1 / (1 + ((1 / u - 1)^p[1] + (1 / v - 1)^p[1])^(1 / p[1])),
    a14 = function(u, v) {
      th <- p[1]
      (1 + ((u^(-1 / th) - 1)^th + (v^(-1 / th) - 1)^th)^(1 / th))^(-th)
    })
}
tb_cdf <- function(family, rotation, p) {
  C <- tb_base(family, p)
  switch(as.character(rotation),
    "0" = C,
    "90" = function(u, v) v - C(1 - u, v),
    "270" = function(u, v) u - C(u, 1 - v))
}
# Textbook Pickands functions, in the u-share convention t = w/(w + z).
tb_A <- function(family, p) {
  switch(family,
    galambos = function(t) 1 - (t^(-p[1]) + (1 - t)^(-p[1]))^(-1 / p[1]),
    husler_reiss = function(t) {
      l <- p[1]
      t * pnorm(1 / l + l / 2 * log(t / (1 - t))) + (1 - t) * pnorm(1 / l - l / 2 * log(t / (1 - t)))
    },
    tev = function(t) {
      rho <- p[1]; nu <- p[2]; k <- sqrt((nu + 1) / (1 - rho^2))
      t * pt(k * ((t / (1 - t))^(1 / nu) - rho), nu + 1) +
        (1 - t) * pt(k * (((1 - t) / t)^(1 / nu) - rho), nu + 1)
    },
    tawn = function(t) {
      th <- p[1]; pu <- p[2]; pv <- p[3]
      (1 - pu) * t + (1 - pv) * (1 - t) + ((pu * t)^th + (pv * (1 - t))^th)^(1 / th)
    },
    NULL)
}

rel_err <- function(x, ref) abs(x - ref) / abs(ref)

# ---------------------------------------------------------------------------
# VineCopula: Tawn types 1 and 2 only (families 104 and 204, par = theta,
# par2 = the free weight). Which weight each code frees is measured: the
# code whose CDF equals the textbook CDF of the case.
# ---------------------------------------------------------------------------
vc_tawn_code <- function(p) {
  if (!xor(p[2] == 1, p[3] == 1)) return(NULL)
  psi <- if (p[2] == 1) p[3] else p[2]
  tb <- tb_cdf("tawn", 0, p)(U[, 1], U[, 2])
  med <- sapply(c(104, 204), function(code) median(abs(BiCopCDF(U[, 1], U[, 2], code, p[1], psi) - tb)))
  hit <- c(104, 204)[med < PARAM_TOL]
  if (length(hit) != 1) stop("VineCopula Tawn ", paste(p, collapse = ", "), ": no unique code matches")
  list(code = hit, par = p[1], par2 = psi,
       frees = if (p[2] == 1) "psi_v" else "psi_u", median_abs_error = min(med))
}

vc_handler <- function(family, rotation, p) {
  if (family != "tawn") return(NULL)
  s <- vc_tawn_code(p)
  if (is.null(s)) return(NULL)
  f <- function(fun, M) fun(M[, 1], M[, 2], s$code, s$par, s$par2)
  td <- BiCopPar2TailDep(s$code, s$par, s$par2)
  list(native = sprintf("family = %d, par = %s, par2 = %s", s$code, jnum(s$par), jnum(s$par2)),
       pdf = f(BiCopPDF, U), cdf = f(BiCopCDF, U),
       h1 = f(BiCopHfunc1, U), h2 = f(BiCopHfunc2, U),
       hinv1 = f(BiCopHinv1, U), hinv2 = f(BiCopHinv2, U),
       tau = BiCopPar2Tau(s$code, s$par, s$par2),
       lambda_L = td$lower, lambda_U = td$upper, A = NULL,
       cdf_fun = function(M) f(BiCopCDF, M),
       h_funs = list(h1 = function(M) f(BiCopHfunc1, M), h2 = function(M) f(BiCopHfunc2, M),
                     hinv1 = function(M) f(BiCopHinv1, M), hinv2 = function(M) f(BiCopHinv2, M)),
       pdf_fun = function(M) f(BiCopPDF, M),
       tawn_code = s)
}

# ---------------------------------------------------------------------------
# copula
# ---------------------------------------------------------------------------
# The asymmetric logistic Tawn model is Khoudraji's device applied to Gumbel:
# C(u, v) = u^(1 - a) v^(1 - b) C_Gumbel(u^a, v^b), ln C = -l(w, z) with
# (psi_u, psi_v) = (a, b) (the shapes). copula's own tawnCopula is another
# model -- Tawn's (1988) one-parameter *mixed* model A(t) = 1 - theta t +
# theta t^2 -- which pmcprg does not have; it is not used.
cop_object <- function(family, p) {
  switch(family,
    galambos = list(galambosCopula(p[1]), sprintf("galambosCopula(%s)", jnum(p[1]))),
    husler_reiss = list(huslerReissCopula(p[1]), sprintf("huslerReissCopula(%s)", jnum(p[1]))),
    tev = list(tevCopula(p[1], df = p[2], df.fixed = TRUE),
               sprintf("tevCopula(%s, df = %s, df.fixed = TRUE)", jnum(p[1]), jnum(p[2]))),
    tawn = list(khoudrajiCopula(indepCopula(), gumbelCopula(p[1]), shapes = c(p[2], p[3])),
                sprintf("khoudrajiCopula(indepCopula(), gumbelCopula(%s), shapes = c(%s, %s))",
                        jnum(p[1]), jnum(p[2]), jnum(p[3]))),
    plackett = list(plackettCopula(p[1]), sprintf("plackettCopula(%s)", jnum(p[1]))),
    amh = list(amhCopula(p[1]), sprintf("amhCopula(%s)", jnum(p[1]))),
    fgm = list(fgmCopula(p[1]), sprintf("fgmCopula(%s)", jnum(p[1]))),
    NULL)
}
COP_EV <- c("galambos", "husler_reiss", "tev")

cop_handler <- function(family, rotation, p) {
  if (rotation != 0) return(NULL)
  o <- cop_object(family, p)
  if (is.null(o)) return(NULL)
  cop <- o[[1]]
  has_h <- family == "amh"       # cCopula is "not yet implemented" for the others
  Us <- U[, 2:1]
  lam <- tryCatch(lambda(cop), error = function(e) c(lower = NA, upper = NA))
  h1 <- function(M) as.numeric(cCopula(M, cop, indices = 2))
  hinv1 <- function(M) as.numeric(cCopula(M, cop, indices = 2, inverse = TRUE, tol = HINV_TOL))
  list(native = o[[2]],
       pdf = dCopula(U, cop), cdf = pCopula(U, cop),
       # AMH is exchangeable: h2(u, v) = h1(v, u), hinv2(w, v) = hinv1(v, w).
       h1 = if (has_h) h1(U) else NULL, h2 = if (has_h) h1(Us) else NULL,
       hinv1 = if (has_h) hinv1(U) else NULL, hinv2 = if (has_h) hinv1(Us) else NULL,
       tau = if (family == "tawn") NA else tau(cop),
       lambda_L = if (family == "tawn") NA else unname(lam[1]),
       lambda_U = if (family == "tawn") NA else unname(lam[2]),
       # copula's A is in the v-share convention, C = exp(log(uv) A(log v / log uv)):
       # recorded at 1 - t, i.e. in the u-share convention of the tables
       # (these three families are exchangeable, A(t) = A(1 - t), anyway).
       A = if (family %in% COP_EV) A(cop, 1 - TGRID) else NULL,
       cdf_fun = function(M) pCopula(M, cop), pdf_fun = function(M) dCopula(M, cop),
       h_funs = if (has_h) list(h1 = h1, hinv1 = hinv1) else NULL)
}

# ---------------------------------------------------------------------------
# copBasic: closed-form CDFs only. Its derivatives (derCOP, densityCOP) and
# tau (tauCOP) are numerical -- finite differences of step 1.5e-8 and
# integrate() -- and are not recorded (tauCOP: Plackett theta = 3 is 1.8e-5
# from copula's value; N4212cop theta = 2 falls back twice and ends 9e-7 off
# the closed form 1 - 2/(3 theta)).
# ---------------------------------------------------------------------------
CB_REFLECT <- c("90" = "acute", "270" = "grave")
cb_handler <- function(family, rotation, p) {
  base <- switch(family,
    galambos = list(GLcop, p[1], sprintf("GLcop(u, v, para = %s)", jnum(p[1]))),
    husler_reiss = list(HRcop, p[1], sprintf("HRcop(u, v, para = %s)", jnum(p[1]))),
    tev = list(tEVcop, c(p[1], p[2]), sprintf("tEVcop(u, v, para = c(%s, %s))", jnum(p[1]), jnum(p[2]))),
    tawn = list(khoudrajiPCOP, list(cop = GHcop, para = p[1], alpha = 1 - p[2], beta = 1 - p[3]),
                sprintf("khoudrajiPCOP(u, v, para = list(cop = GHcop, para = %s, alpha = %s, beta = %s))",
                        jnum(p[1]), jnum(1 - p[2]), jnum(1 - p[3]))),
    plackett = list(PLcop, p[1], sprintf("PLcop(u, v, para = %s)", jnum(p[1]))),
    amh = list(AMHcop, p[1], sprintf("AMHcop(u, v, para = %s)", jnum(p[1]))),
    fgm = list(FGMcop, p[1], sprintf("FGMcop(u, v, para = %s)", jnum(p[1]))),
    a12 = list(N4212cop, p[1], sprintf("N4212cop(u, v, para = %s)", jnum(p[1]))),
    NULL)
  if (is.null(base)) return(NULL)
  if (rotation == 0) {
    cdf_fun <- function(M) base[[1]](M[, 1], M[, 2], para = base[[2]])
    native <- base[[3]]
  } else {
    refl <- CB_REFLECT[[as.character(rotation)]]
    cdf_fun <- function(M) COP(M[, 1], M[, 2], cop = base[[1]], para = base[[2]], reflect = refl)
    native <- sprintf("COP(u, v, cop = %s, para = %s, reflect = \"%s\")",
                      sub("\\(.*", "", base[[3]]), jnum(p[1]), refl)
  }
  list(native = native, pdf = NULL, cdf = cdf_fun(U), h1 = NULL, h2 = NULL,
       hinv1 = NULL, hinv2 = NULL, tau = NA, lambda_L = NA, lambda_U = NA, A = NULL,
       cdf_fun = cdf_fun)
}

# ---------------------------------------------------------------------------
# fCopulae: EV (galambos, husler.reiss, tawn with param = c(alpha, beta, r))
# and Archimedean (Nelsen's numbering: 3 = AMH, 12, 14).
#
# Its generic Archimedean density (.darchm1Copula, the default) is wrong for
# type 3: .invPhiFirstDer/.invPhiSecondDer write (e^y - 1) where the AMH
# generator inverse psi(y) = (1 - a)/(e^y - a) needs (e^y - a) -- measured
# below (fcopulae_amh_default_density) against its own per-type formula
# (alternative = TRUE), which is the one recorded for type 3.
# ---------------------------------------------------------------------------
FC_EV <- c(galambos = "galambos", husler_reiss = "husler.reiss", tawn = "tawn")
FC_ARCHM <- c(amh = "3", a12 = "12", a14 = "14")
fc_handler <- function(family, rotation, p) {
  if (rotation != 0) return(NULL)
  if (family %in% names(FC_EV)) {
    type <- FC_EV[[family]]
    param <- if (family == "tawn") c(p[2], p[3], p[1]) else p[1]
    pdf_fun <- function(M) as.numeric(devCopula(M[, 1], M[, 2], param = param, type = type))
    cdf_fun <- function(M) as.numeric(pevCopula(M[, 1], M[, 2], param = param, type = type))
    td <- evTailCoeff(param, type)
    return(list(native = sprintf("type = \"%s\", param = %s", type, jvec(param)),
                pdf = pdf_fun(U), cdf = cdf_fun(U), h1 = NULL, h2 = NULL, hinv1 = NULL, hinv2 = NULL,
                tau = as.numeric(evTau(param, type)),
                lambda_L = unname(td[["lower"]]), lambda_U = unname(td[["upper"]]),
                A = as.numeric(Afunc(TGRID, param, type)),
                cdf_fun = cdf_fun, pdf_fun = pdf_fun))
  }
  if (family %in% names(FC_ARCHM)) {
    type <- FC_ARCHM[[family]]
    alt <- family == "amh"
    pdf_fun <- function(M) as.numeric(darchmCopula(M[, 1], M[, 2], alpha = p[1], type = type,
                                                   alternative = alt))
    cdf_fun <- function(M) as.numeric(parchmCopula(M[, 1], M[, 2], alpha = p[1], type = type))
    out <- list(native = sprintf("type = \"%s\", alpha = %s%s", type, jnum(p[1]),
                                 if (alt) ", alternative = TRUE (density)" else ""),
                pdf = pdf_fun(U), cdf = cdf_fun(U), h1 = NULL, h2 = NULL, hinv1 = NULL, hinv2 = NULL,
                tau = as.numeric(archmTau(p[1], type)), lambda_L = NA, lambda_U = NA, A = NULL,
                cdf_fun = cdf_fun, pdf_fun = pdf_fun)
    if (alt) {
      default <- as.numeric(darchmCopula(U[, 1], U[, 2], alpha = p[1], type = type))
      out$amh_default_density_rel_error <- rel_err(default, out$pdf)
    }
    return(out)
  }
  NULL
}

# ---------------------------------------------------------------------------
# Convention checks
# ---------------------------------------------------------------------------
shift <- function(M, e) M + e[rep(1, nrow(M)), ]

# pdf against the second mixed difference of the package's own CDF, for the
# packages that give no h-function (relative above pdf = 1, absolute below).
fd2_pdf <- function(val) {
  d <- FD2_STEP
  C <- val$cdf_fun
  e1 <- cbind(d, 0); e2 <- cbind(0, d)
  dd <- (C(shift(shift(U, e1), e2)) - C(shift(shift(U, e1), -e2)) -
         C(shift(shift(U, -e1), e2)) + C(shift(shift(U, -e1), -e2))) / (4 * d * d)
  abs(val$pdf - dd) / pmax(val$pdf, 1)
}

# The h-function conventions, as gen_r.R (finite differences and round trips).
h_checks <- function(val) {
  hf <- val$h_funs
  e1 <- cbind(FD_STEP, 0); e2 <- cbind(0, FD_STEP)
  C <- val$cdf_fun
  out <- list()
  out$h1_minus_dC_du1 <- abs(hf$h1(U) - (C(shift(U, e1)) - C(shift(U, -e1))) / (2 * FD_STEP))
  pdf <- val$pdf
  out$pdf_minus_dh1_du2 <- abs(pdf - (hf$h1(shift(U, e2)) - hf$h1(shift(U, -e2))) / (2 * FD_STEP)) /
                           pmax(pdf, 1)
  y <- hf$hinv1(U)
  out$h1_of_hinv1_minus_level <- abs(hf$h1(cbind(U[, 1], y)) - U[, 2])
  if (!is.null(hf$h2)) {
    out$h2_minus_dC_du2 <- abs(hf$h2(U) - (C(shift(U, e2)) - C(shift(U, -e2))) / (2 * FD_STEP))
    out$pdf_minus_dh2_du1 <- abs(pdf - (hf$h2(shift(U, e1)) - hf$h2(shift(U, -e1))) / (2 * FD_STEP)) /
                             pmax(pdf, 1)
    x <- hf$hinv2(U)
    out$h2_of_hinv2_minus_level <- abs(hf$h2(cbind(x, U[, 2])) - U[, 1])
  }
  out
}

check_median <- function(checks, label) {
  for (k in names(checks)) {
    tol <- if (grepl("hinv", k)) ROUNDTRIP_TOL else if (grepl("^param|^A_", k)) PARAM_TOL else FD_TOL
    med <- median(checks[[k]])
    if (is.na(med) || !(med < tol))
      stop("convention check failed for ", label, ": median ", k, " = ", signif(med, 3))
  }
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
repo_version <- sub('.*__version__ = "([^"]+)".*', "\\1",
                    grep("__version__ =", readLines(file.path(root, "pmcprg", "__init__.py")),
                         value = TRUE)[1])
generated <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S+00:00", tz = "UTC")

PACKAGES <- list(
  VineCopula = list(handler = vc_handler, file = "extra_vinecopula.json"),
  copula = list(handler = cop_handler, file = "extra_rcopula.json"),
  copBasic = list(handler = cb_handler, file = "extra_copbasic.json"),
  fCopulae = list(handler = fc_handler, file = "extra_fcopulae.json"))

DEFINITIONS <- list(
  VineCopula = list(
    pdf = "BiCopPDF", cdf = "BiCopCDF",
    h1 = "BiCopHfunc1 = P(U2 <= v | U1 = u) = dC/du",
    h2 = "BiCopHfunc2 = P(U1 <= u | U2 = v) = dC/dv",
    hinv1 = "BiCopHinv1: the y with h1(u, y) = v", hinv2 = "BiCopHinv2: the x with h2(x, v) = u",
    tau = "BiCopPar2Tau", lambda = "BiCopPar2TailDep: lower, upper",
    tawn_codes = "104 frees psi_u (psi_v = 1), 204 frees psi_v (psi_u = 1): measured, convention_checks.tawn_codes"),
  copula = list(
    pdf = "dCopula", cdf = "pCopula",
    h1 = "cCopula(cbind(u, v), cop, indices = 2) = dC/du; AMH only (not implemented for the others)",
    h2 = "cCopula(cbind(v, u), cop, indices = 2) = h1(v, u) = dC/dv (AMH is exchangeable)",
    hinv1 = "cCopula(cbind(u, v), cop, indices = 2, inverse = TRUE, tol = 1e-15); AMH only",
    hinv2 = "cCopula(cbind(v, u), cop, indices = 2, inverse = TRUE, tol = 1e-15); AMH only",
    tau = "tau(cop); not recorded for the Khoudraji-Gumbel Tawn (no method)",
    lambda = "lambda(cop): lower, upper; not for fgmCopula (no method) nor the Tawn",
    A = "A(cop, 1 - t): copula's A is in the v-share convention; recorded in the u-share one",
    tawn = "khoudrajiCopula(indepCopula(), gumbelCopula(theta), shapes = c(psi_u, psi_v)): the asymmetric logistic model; copula's tawnCopula is Tawn's mixed model, not recorded"),
  copBasic = list(
    cdf = "the family's closed-form CDF; COP(reflect = \"acute\") for 90, \"grave\" for 270 (measured)",
    others = "not recorded: derCOP/densityCOP are finite differences, tauCOP numerical integration"),
  fCopulae = list(
    pdf = "devCopula / darchmCopula (alternative = TRUE for type 3, see convention_checks)",
    cdf = "pevCopula / parchmCopula", tau = "evTau (integrate) / archmTau",
    lambda = "evTailCoeff = 2 - 2 A(1/2); Archimedean types not recorded (archmTailCoeff is a numerical sequence)",
    A = "Afunc(t): u-share convention t = log u / log(uv) (measured, convention_checks)",
    tawn = "type tawn, param = c(alpha, beta, r) = c(psi_u, psi_v, theta) (measured)"))

run <- function(pkg) {
  spec <- PACKAGES[[pkg]]
  entries <- character(0)
  checks <- list()
  tawn_codes <- list()
  amh_default <- NULL
  for (i in seq_len(nrow(cases))) {
    fam <- cases$family[i]; rot <- cases$rotation[i]; p <- case_pars(i)
    val <- spec$handler(fam, rot, p)
    if (is.null(val)) next
    key <- paste0(fam, "/", rot)
    chk <- list()
    tb <- tb_cdf(fam, rot, p)(U[, 1], U[, 2])
    chk$param_cdf_abs_error <- abs(val$cdf - tb)
    if (!is.null(val$A)) chk$A_rel_error <- rel_err(val$A, tb_A(fam, p)(TGRID))
    if (!is.null(val$pdf)) {
      if (!is.null(val$h_funs)) chk <- c(chk, h_checks(val)) else chk$pdf_minus_d2C_dudv <- fd2_pdf(val)
    }
    check_median(chk, paste(pkg, key, paste(p, collapse = ", ")))
    for (k in names(chk)) {
      kmax <- paste0(k, "_max"); kmed <- paste0(k, "_median")
      checks[[key]][[kmax]] <- max(c(checks[[key]][[kmax]], chk[[k]]))
      checks[[key]][[kmed]] <- max(c(checks[[key]][[kmed]], median(chk[[k]])))
    }
    if (!is.null(val$tawn_code)) {
      tc <- val$tawn_code
      prev <- tawn_codes[[as.character(tc$code)]]
      if (!is.null(prev) && prev != tc$frees) stop("VineCopula Tawn: the freed weight depends on the case")
      tawn_codes[[as.character(tc$code)]] <- tc$frees
    }
    if (!is.null(val$amh_default_density_rel_error))
      amh_default <- max(c(amh_default, val$amh_default_density_rel_error))
    fields <- c(
      family = jstr(fam), rotation = as.character(rot), pars = jvec(p), native = jstr(val$native),
      pdf = jvec(val$pdf), cdf = jvec(val$cdf), h1 = jvec(val$h1), h2 = jvec(val$h2),
      hinv1 = jvec(val$hinv1), hinv2 = jvec(val$hinv2),
      tau = jnum(val$tau), lambda_L = jnum(val$lambda_L), lambda_U = jnum(val$lambda_U),
      A = jvec(val$A))
    entries <- c(entries, paste0("  {", paste0(jstr(names(fields)), ": ", fields, collapse = ", "), "}"))
  }
  pf <- vapply(names(checks), function(k)
    paste0("   ", jstr(k), ": ", jobj(lapply(checks[[k]], jnum), "   ")), "")
  extra <- ""
  if (length(tawn_codes))
    extra <- paste0(extra, ",\n  \"tawn_codes\": ", jobj(lapply(tawn_codes, jstr), "  "))
  if (!is.null(amh_default))
    extra <- paste0(extra, ",\n  \"fcopulae_amh_default_density_rel_error_max\": ", jnum(amh_default))
  header <- c(
    format = "1",
    about = jstr(paste0("FR-11 interior parity references (families beyond the pilot), generated by ",
                        "scripts/parity/gen_r_extra.R - do not edit by hand. Floats are %.17g decimal ",
                        "strings of IEEE-754 doubles; null = not recorded.")),
    reference = jobj(list(package = jstr(pkg), version = jstr(as.character(packageVersion(pkg)))), " "),
    environment = jobj(list(R = jstr(R.version.string), platform = jstr(R.version$platform),
                            VineCopula = jstr(as.character(packageVersion("VineCopula"))),
                            copula = jstr(as.character(packageVersion("copula"))),
                            copBasic = jstr(as.character(packageVersion("copBasic"))),
                            fCopulae = jstr(as.character(packageVersion("fCopulae")))), " "),
    generated = jstr(generated),
    repository_version = jstr(repo_version),
    command = jstr("Rscript scripts/parity/gen_r_extra.R"),
    definitions = jobj(lapply(c(DEFINITIONS[[pkg]], list(
      pars = paste("reference-agnostic key (README): galambos [theta]; husler_reiss [lambda];",
                   "tev [rho, nu]; tawn [theta, psi_u, psi_v], psi_u the weight of -log u;",
                   "plackett/amh/fgm/a12/a14 [theta] of the unrotated family"))), jstr), " "),
    convention_checks = paste0("{\n  \"fd_step\": ", jnum(FD_STEP),
                               ",\n  \"fd_tolerance\": ", jnum(FD_TOL),
                               ",\n  \"fd2_step\": ", jnum(FD2_STEP),
                               ",\n  \"roundtrip_tolerance\": ", jnum(ROUNDTRIP_TOL),
                               ",\n  \"param_tolerance\": ", jnum(PARAM_TOL),
                               ",\n  \"judged_on\": \"median over the points (max recorded); ",
                               "param_* absolute, A_* relative, to the textbook formula; ",
                               "pdf errors relative above pdf = 1, absolute below\"",
                               ",\n  \"per_family\": {\n", paste(pf, collapse = ",\n"), "\n  }",
                               extra, "\n }"),
    points = paste0("[", paste(apply(U, 1, jvec), collapse = ", "), "]"),
    pickands_points = jvec(TGRID),
    # The whole grid of cases_extra.csv, whether this package covers a case or
    # not: the tests read it from here (the mpmath oracle covers every case).
    grid = paste0("[", paste(vapply(seq_len(nrow(cases)), function(i)
      paste0("[", jstr(cases$family[i]), ", ", cases$rotation[i], ", ", jvec(case_pars(i)), "]"), ""),
      collapse = ", "), "]"))
  txt <- paste0("{\n", paste0(" ", jstr(names(header)), ": ", header, collapse = ",\n"),
                ",\n \"cases\": [\n", paste(entries, collapse = ",\n"), "\n ]\n}\n")
  fname <- file.path(out_dir, spec$file)
  writeLines(txt, fname, sep = "")
  cat(sprintf("wrote %s: %d cases x %d points\n", fname, length(entries), nrow(U)))
  for (k in names(tawn_codes)) cat(sprintf("  VineCopula family %s frees %s\n", k, tawn_codes[[k]]))
  if (!is.null(amh_default))
    cat(sprintf("  fCopulae default AMH density: max relative error %.3g\n", amh_default))
}

for (pkg in names(PACKAGES)) run(pkg)
