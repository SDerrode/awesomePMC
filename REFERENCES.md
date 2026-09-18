# References

Every reference the code, the reports and the methodological note rely on, with
the identifier that resolves to it and the place in the package that uses it.
Each DOI was resolved against Crossref (or DataCite for the dataset) on
2026-09-13 (audit B-8 / K-16); entries marked *no DOI* were confirmed to have
none. The two papers the package implements are **DerrodePieczynski_CSDA2013**
and **DerrodePieczynski_SP2016**; please cite both — see
[`CITATION.cff`](CITATION.cff).

## The model and its estimators

- **DerrodePieczynski_CSDA2013** — Derrode, S. & Pieczynski, W. (2013). Unsupervised data classification using pairwise Markov chains with automatic copulas selection. *Computational Statistics & Data Analysis* 63, 81–98. doi:10.1016/j.csda.2013.01.027 — the PMC family (§2), forward–backward/MPM (§3), ICE with copula selection and the Huard criterion (§4, Eq. 20); `pmcprg/pmc/model.py`, `inference.py`, `ice.py`, `report/`.
- **DerrodePieczynski_SP2016** — Derrode, S. & Pieczynski, W. (2016). Unsupervised classification using hidden Markov chain with unknown noise copulas and margins. *Signal Processing* 128, 8–17. doi:10.1016/j.sigpro.2016.03.008 — GICE, margin-family selection (§3, Eqs. 10–11); `pmcprg/pmc/ice.py` (margin rules).
- Derrode, S. & Pieczynski, W. (2004). Signal and image segmentation using pairwise Markov chains. *IEEE Transactions on Signal Processing* 52(9), 2477–2489. doi:10.1109/TSP.2004.832015 — PMC segmentation, Hilbert–Peano scan; `pmcprg/pmc/peano.py`.
- Pieczynski, W. (1992). Statistical image segmentation. *Machine Graphics and Vision* 1(1/2), 261–268. *No DOI.* — ICE; `pmcprg/pmc/ice.py`.
- Delignon, Y., Marzouki, A. & Pieczynski, W. (1997). Estimation of generalized mixtures and its application in image segmentation. *IEEE Transactions on Image Processing* 6(10), 1364–1375. doi:10.1109/83.624951 — generalised mixtures / ICE; `pmcprg/pmc/ice.py`.
- Giordana, N. & Pieczynski, W. (1997). Estimation of generalized multisensor hidden Markov chains and unsupervised image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 19(5), 465–475. doi:10.1109/34.589206 — ICE, Hilbert–Peano scan; `pmcprg/pmc/ice.py`, `peano.py`.
- Celeux, G. & Diebolt, J. (1985). The SEM algorithm: a probabilistic teacher algorithm derived from the EM algorithm for the mixture problem. *Computational Statistics Quarterly* 2, 73–82. *No DOI.* — `pmcprg/pmc/sem.py`.
- Delyon, B., Lavielle, M. & Moulines, E. (1999). Convergence of a stochastic approximation version of the EM algorithm. *The Annals of Statistics* 27(1), 94–128. doi:10.1214/aos/1018031103 — SAEM, the smoothed alternative to SEM (note).
- Nielsen, S. F. (2000). The stochastic EM algorithm: estimation and asymptotic results. *Bernoulli* 6(3), 457–489. doi:10.2307/3318671 — SEM asymptotics (note).
- Devijver, P. A. (1985). Baum's forward-backward algorithm revisited. *Pattern Recognition Letters* 3(6), 369–373. doi:10.1016/0167-8655(85)90023-6 — normalised forward pass; `pmcprg/pmc/inference.py`.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE* 77(2), 257–286. doi:10.1109/5.18626 — per-step scaling of the backward pass; `pmcprg/pmc/inference.py`.
- Carter, C. K. & Kohn, R. (1994). On Gibbs sampling for state space models. *Biometrika* 81(3), 541–553. doi:10.1093/biomet/81.3.541 — forward-filtering backward-sampling; `pmcprg/pmc/inference.py` (`sample_posterior`).
- Frühwirth-Schnatter, S. (1994). Data augmentation and dynamic linear models. *Journal of Time Series Analysis* 15(2), 183–202. doi:10.1111/j.1467-9892.1994.tb00184.x — FFBS.
- Chib, S. (1996). Calculating posterior distributions and modal estimates in Markov mixture models. *Journal of Econometrics* 75(1), 79–97. doi:10.1016/0304-4076(95)01770-4 — FFBS for discrete states.
- Skarbek, W. (1992). Generalized Hilbert scan in image printing. In R. Klette & W. Kropatsch (eds), *Theoretical Foundations of Computer Vision*, Akademie Verlag, Berlin. *No DOI (no Crossref record).* — Hilbert–Peano scan; `pmcprg/pmc/peano.py`.
- Červený, J. (2018). *gilbert* — generalized Hilbert curves. https://github.com/jakubcerveny/gilbert (BSD-2-Clause) — the space-filling-curve code ported in `pmcprg/pmc/_gilbert.py`.

## Copulas

- Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de l'Institut de Statistique de l'Université de Paris* 8, 229–231. *No DOI.* — `pmcprg/copulas/bivariate.py`.
- Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed. Springer, New York. doi:10.1007/0-387-28678-0 — Table 4.1 (Archimedean families (4.2.n)); ch. 2 (survival copulas, random variate generation), ch. 3 (quadratic and cubic sections, Plackett distributions), ch. 5 (τ and ρ_S); every module of `pmcprg/copulas/`.
- Joe, H. (1993). Parametric families of multivariate distributions with given margins. *Journal of Multivariate Analysis* 46(2), 262–282. doi:10.1006/jmva.1993.1061 — the BB construction the two-parameter families come from; `archimedean/bb7.py`, `archimedean/bb8.py`. *The DOI often copied for this article, `10.1006/jmva.93.1061`, does not resolve: the one above is what Crossref returns for it (verified 2026-09-18).*
- Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall/CRC, Monographs on Statistics and Applied Probability 73. doi:10.1201/b13150 — ch. 5 (Joe family, BB1, BB6, BB7, BB8), ch. 2 (tail dependence). *Crossref titles this edition "Multivariate Models and Multivariate Dependence Concepts" (verified 2026-09-18); the spine reads as above.*
- Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman & Hall/CRC. doi:10.1201/b17116.
- Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate distributions with uniform marginals. *The American Statistician* 40(4), 280–283. doi:10.1080/00031305.1986.10475414 — τ = 1 + 4∫φ/φ′ for Archimedean generators.
- Clayton, D. G. (1978). A model for association in bivariate life tables and its application in epidemiological studies of familial tendency in chronic disease incidence. *Biometrika* 65(1), 141–151. doi:10.1093/biomet/65.1.141 — `pmcprg/copulas/archimedean/clayton.py`.
- Gumbel, E. J. (1960). Bivariate exponential distributions. *Journal of the American Statistical Association* 55(292), 698–707. doi:10.1080/01621459.1960.10483368 — `gumbel.py`.
- Frank, M. J. (1979). On the simultaneous associativity of F(x, y) and x + y − F(x, y). *Aequationes Mathematicae* 19(1), 194–226. doi:10.1007/BF02189866 — `frank.py`.
- Genest, C. (1987). Frank's family of bivariate distributions. *Biometrika* 74(3), 549–555. doi:10.1093/biomet/74.3.549 — `frank.py`.
- Ali, M. M., Mikhail, N. N. & Haq, M. S. (1978). A class of bivariate distributions including the bivariate logistic. *Journal of Multivariate Analysis* 8(3), 405–412. doi:10.1016/0047-259X(78)90063-5 — `amh.py`.
- Plackett, R. L. (1965). A class of bivariate distributions. *Journal of the American Statistical Association* 60(310), 516–522. doi:10.1080/01621459.1965.10480807 — `explicit/plackett.py`.
- Farlie, D. J. G. (1960). The performance of some correlation coefficients for a general bivariate distribution. *Biometrika* 47(3/4), 307–323. doi:10.1093/biomet/47.3-4.307 — FGM; `explicit/fgm.py`.
- Nelsen, R. B., Quesada-Molina, J. J. & Rodríguez-Lallena, J. A. (1997). Bivariate copulas with cubic sections. *Journal of Nonparametric Statistics* 7(3), 205–220. doi:10.1080/10485259708832700 — `explicit/cubic_section.py`.
- Demarta, S. & McNeil, A. J. (2005). The t copula and related copulas. *International Statistical Review* 73(1), 111–129. doi:10.1111/j.1751-5823.2005.tb00254.x — Student tail dependence; `elliptical/student.py`.
- Aas, K., Czado, C., Frigessi, A. & Bakken, H. (2009). Pair-copula constructions of multiple dependence. *Insurance: Mathematics and Economics* 44(2), 182–198. doi:10.1016/j.insmatheco.2007.02.001 — h-functions of the elliptical copulas; `elliptical/`.
- Rosenblatt, M. (1952). Remarks on a multivariate transformation. *The Annals of Mathematical Statistics* 23(3), 470–472. doi:10.1214/aoms/1177729394 — conditional-inverse sampling; `pmcprg/pmc/simulate.py`, `pmcprg/copulas/_base.py`.
- Patton, A. J. (2006). Modelling asymmetric exchange rate dependence. *International Economic Review* 47(2), 527–556. doi:10.1111/j.1468-2354.2006.00387.x — the "symmetrised Joe-Clayton" reparametrisation of BB7 by its two tail-dependence coefficients (λ_U = 2 − 2^{1/θ}, λ_L = 2^{−1/δ}); `archimedean/bb7.py`.
- Caillault, C. & Guégan, D. (2005). Empirical estimation of tail dependence using copulas: application to Asian markets. *Quantitative Finance* 5(5), 489–501. doi:10.1080/14697680500147853 — empirical λ̂_L, λ̂_U; `pmcprg/copulas/_fit.py`.
- Segers, J., Sibuya, M. & Tsukahara, H. (2017). The empirical beta copula. *Journal of Multivariate Analysis* 155, 35–51. doi:10.1016/j.jmva.2016.11.010 — verified against Crossref by bibliographic query. `pmcprg/copulas/_nonparametric.py` (audit FR-9, nonparametric comparison tool, not a registered family).

### Numerical evaluation

- Hofert, M., Mächler, M. & McNeil, A. J. (2012). Likelihood inference for Archimedean copulas in high dimensions under known margins. *Journal of Multivariate Analysis* 110, 133–150. doi:10.1016/j.jmva.2012.02.019 — log-space evaluation of Archimedean densities; the kernels of `clayton.py`, `gumbel.py`, `joe.py`, `frank.py`, `a12.py`, `a14.py`, `bb1.py` (audit K-1…K-3, FR-1).
- Mächler, M. (2012). *Accurately computing log(1 − exp(−|a|)) assessed by the Rmpfr package*. Vignette of the R package Rmpfr, CRAN. *No DOI.* — `log1mexp` with its switch at log 2.
- Drezner, Z. & Wesolowsky, G. O. (1990). On the computation of the bivariate normal integral. *Journal of Statistical Computation and Simulation* 35, 101–107. doi:10.1080/00949659008811236 — deterministic bivariate normal CDF in `elliptical/gaussian.py`.
- Genz, A. (2004). Numerical computation of rectangular bivariate and trivariate normal and t probabilities. *Statistics and Computing* 14, 251–260. doi:10.1023/B:STCO.0000035304.20635.31 — the Gauss–Legendre form of that integral.

## Estimating and selecting copulas

- Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika* 30(1/2), 81–93. doi:10.1093/biomet/30.1-2.81 — τ; weighted τ plug-in in `pmcprg/pmc/ice.py`.
- Genest, C. & Rivest, L.-P. (1993). Statistical inference procedures for bivariate Archimedean copulas. *Journal of the American Statistical Association* 88(423), 1034–1043. doi:10.1080/01621459.1993.10476372 — inversion of Kendall's τ (`fit(method='tau')`), the Kendall distribution K_θ.
- Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric estimation procedure of dependence parameters in multivariate families of distributions. *Biometrika* 82(3), 543–552. doi:10.1093/biomet/82.3.543 — pseudo-maximum likelihood (`fit(method='mle')`); its sandwich variance with the estimated-rank corrections (`standard_errors`, `pmcprg/copulas/_stderr.py`, audit FR-4).
- Genest, C. & Favre, A.-C. (2007). Everything you always wanted to know about copula modeling but were afraid to ask. *Journal of Hydrologic Engineering* 12(4), 347–368. doi:10.1061/(ASCE)1084-0699(2007)12:4(347) — asymptotic variance 16·Var{2C(U,V) − U − V} of Kendall's τ̂ (`standard_errors(method='tau')`).
- Kojadinovic, I. & Yan, J. (2010). Comparison of three semiparametric methods for estimating dependence parameters in copula models. *Insurance: Mathematics and Economics* 47(1), 52–63. doi:10.1016/j.insmatheco.2010.03.008 — the same variance estimated with the empirical copula; τ inversion against pseudo-likelihood.
- Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators and likelihood ratio tests under nonstandard conditions. *Journal of the American Statistical Association* 82(398), 605–610. doi:10.1080/01621459.1987.10478472 — ½χ²₀ + ½χ²₁ at the boundary (`independence_lr_test`, `StandardErrors.at_boundary`).
- Klaassen, C. A. J. & Wellner, J. A. (1997). Efficient estimation in the bivariate normal copula model: normal margins are least favourable. *Bernoulli* 3(1), 55–77. doi:10.2307/3318652 — n·Var(ρ̂) → (1 − ρ²)², the closed form `test_fr4_standard_errors.py` checks.
- Huard, D., Évin, G. & Favre, A.-C. (2006). Bayesian copula selection. *Computational Statistics & Data Analysis* 51(2), 809–822. doi:10.1016/j.csda.2005.08.010 — the `huard` criterion (DerrodePieczynski_CSDA2013 Eq. 20); `pmcprg/pmc/ice.py`.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control* 19(6), 716–723. doi:10.1109/TAC.1974.1100705 — `aic`.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics* 6(2), 461–464. doi:10.1214/aos/1176344136 — `bic`.
- Grønneberg, S. & Hjort, N. L. (2014). The copula information criteria. *Scandinavian Journal of Statistics* 41(2), 436–459. doi:10.1111/sjos.12042 — why AIC on pseudo-observations needs correction; context for `xvcic`.
- Ko, V. & Hjort, N. L. (2019). Copula information criterion for model selection with two-stage maximum likelihood estimation. *Econometrics and Statistics* 12, 167–180. doi:10.1016/j.ecosta.2019.01.001 — the two-stage (parametric-margin) criterion `xvcic` approximates.
- Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for copulas: A review and a power study. *Insurance: Mathematics and Economics* 44(2), 199–213. doi:10.1016/j.insmatheco.2007.10.005 — the Cramér–von Mises statistic S_n (`cvm` criterion, `gof_test`).

## Diagnostics

- Genest, C. & Rémillard, B. (2008). Validity of the parametric bootstrap for goodness-of-fit testing in semiparametric models. *Annales de l'Institut Henri Poincaré, Probabilités et Statistiques* 44(6), 1096–1127. doi:10.1214/07-AIHP148 — the composite-null bootstrap (`FitResult.gof_test` refits; `parametric_bootstrap` deliberately does not).
- Durbin, J. (1973). Weak convergence of the sample distribution function when parameters are estimated. *The Annals of Statistics* 1(2), 279–290. doi:10.1214/aos/1176342365 — the "Durbin effect"; `pmcprg/diagnostics/bootstrap.py`.
- Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press. doi:10.1017/CBO9780511802843 — the Monte-Carlo p-value (1 + #)/(B + 1), ch. 4.
- Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1), Article 39. doi:10.2202/1544-6115.1585.
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics* 7(1), 1–26. doi:10.1214/aos/1176344552 — percentile intervals for τ (`_do_tau_ci`).
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall, New York. doi:10.1007/978-1-4899-4541-9 — ch. 13.
- Neyman, J. (1937). "Smooth test" for goodness of fit. *Skandinavisk Aktuarietidskrift* 20, 149–199. doi:10.1080/03461238.1937.10404821 — `pmcprg/diagnostics/smooth.py`.
- Rayner, J. C. W. & Best, D. J. (1989). *Smooth Tests of Goodness of Fit*. Oxford University Press, New York. ISBN 0-19-505610-8. *No DOI.* — reading V₁…V₄ as location/dispersion/asymmetry/tail weight.
- Ledwina, T. (1994). Data-driven version of Neyman's smooth test of fit. *Journal of the American Statistical Association* 89(427), 1000–1005. doi:10.1080/01621459.1994.10476834.
- Kallenberg, W. C. M. & Ledwina, T. (1997). Data-driven smooth tests when the hypothesis is composite. *Journal of the American Statistical Association* 92(439), 1094–1104. doi:10.1080/01621459.1997.10474065 — why the components are bootstrap-calibrated here.
- Henze, N. (1997). Do components of smooth tests of fit have diagnostic properties? *Metrika* 45(1), 121–130. doi:10.1007/BF02717098 — the caveat on reading a single component.
- Genest, C., Quessy, J.-F. & Rémillard, B. (2006). Goodness-of-fit procedures for copula models based on the probability integral transformation. *Scandinavian Journal of Statistics* 33(2), 337–366. doi:10.1111/j.1467-9469.2006.00470.x — the Kendall-process statistic (dK_θ form).
- Wang, W. & Wells, M. T. (2000). Model selection and semiparametric inference for bivariate failure-time data. *Journal of the American Statistical Association* 95(449), 62–72. doi:10.1080/01621459.2000.10473899 — the Lebesgue-weighted form the GUI's `kendall` statistic uses.
- Anderson, T. W. & Darling, D. A. (1952). Asymptotic theory of certain "goodness of fit" criteria based on stochastic processes. *The Annals of Mathematical Statistics* 23(2), 193–212. doi:10.1214/aoms/1177729437; and (1954). A test of goodness of fit. *Journal of the American Statistical Association* 49(268), 765–769. doi:10.1080/01621459.1954.10501232 — `report/margin_statistic_panel.py`.
- Székely, G. J. & Rizzo, M. L. (2013). Energy statistics: A class of statistics based on distances. *Journal of Statistical Planning and Inference* 143(8), 1249–1272. doi:10.1016/j.jspi.2013.03.018 — `report/margin_statistic_panel.py`.
- Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B. & Smola, A. (2012). A kernel two-sample test. *Journal of Machine Learning Research* 13, 723–773. *No DOI* — https://jmlr.org/papers/v13/gretton12a.html — MMD; `report/margin_statistic_panel.py`.
- Naaman, M. (2021). On the tight constant in the multivariate Dvoretzky–Kiefer–Wolfowitz inequality. *Statistics & Probability Letters* 173, 109088. doi:10.1016/j.spl.2021.109088 — the multivariate KS test; `pmcprg/diagnostics/mks.py`.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158), 209–212. doi:10.1080/01621459.1927.10502953 — score intervals in `report/`.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2), 153–157. doi:10.1007/BF02295996 — paired tests in `report/reproduce_csda2013.py`.
- Kish, L. (1965). *Survey Sampling*. Wiley, New York. ISBN 0-471-48900-X. *No DOI.* — the effective sample size (Σw)²/Σw² the package does **not** use (it uses Σw; CHANGELOG, note).

## Data and benchmarks

- Anguita, D., Ghio, A., Oneto, L., Parra, X. & Reyes-Ortiz, J. L. (2013). A public domain dataset for human activity recognition using smartphones. *ESANN 2013*, pp. 437–442. *No DOI.* The dataset itself: Reyes-Ortiz, J. L. & Anguita, D. (2013). *Human Activity Recognition Using Smartphones* [dataset]. UCI Machine Learning Repository. doi:10.24432/C54S4K — `examples/`, `data/uci_har/`.
- Gorynin, I., Gangloff, H., Monfrini, E. & Pieczynski, W. (2018). Assessing the segmentation performance of pairwise and triplet Markov models. *Signal Processing* 145, 183–192. doi:10.1016/j.sigpro.2017.12.006 — the PMM/TMM-vs-HMM benchmark.
