Forecasting study — 2026-09-23 19:45, 1228446 score rows, wall 92.6 min, sum of pmcprg task times 367.3 min, pmmforecast jobs 126.5 min (3.14.7, numpy 2.5.3, arm64).
Quantile CRPS (200 levels) against the exact CRPS: {"gaussian": {"n": 468135, "mean_rel_diff": 3.7409298167066247e-05, "max_rel_diff": 0.0003036628052002386, "rel_diff_of_mean": 3.450984695462589e-05}, "hmc_in_mixture": {"n": 342438, "mean_rel_diff": 8.298970350852841e-05, "max_rel_diff": 0.0021164674819841387, "rel_diff_of_mean": 3.21674236069811e-05}, "pmc_node_law_vs_200_quantiles": {"n": 2832, "mean_rel_diff": 0.0007392849962199574, "max_rel_diff": 0.023199298231480862, "rel_diff_of_mean": 0.00013558889564567003}}
Rows with quantiles inconsistent with the node law (pmcprg P3; q_check > 0.25 on each horizon's own grid): {"pmc_state_K2|raw": 6, "pmc_state_K2|raw+gate": 6, "pmc_state_K2|robust": 28, "pmc_state_K2|robust+gate": 26}; on the reference nodes: {"pmc_state_K2|raw": 6, "pmc_state_K2|raw+gate": 6, "pmc_state_K2|robust": 34, "pmc_state_K2|robust+gate": 26, "pmc_state_K2|true": 31}; statistics {"n_grid_rows": 276726, "q_check_max": 0.3756619844067825, "q_check_p99": 0.10378811547984909, "q_check_ref_max": 3.210945933677482, "q_check_tail_max": 0.3545698129250693, "q_check_tail_p99": 0.016677140985876544, "q_check_tail_rows_above_0.05": 463, "q_check_tail_rows_above_0.05_by_method": {"pmc_pair_K3|hampel+gate": 2, "pmc_pair_K3|raw": 58, "pmc_pair_K3|raw+gate": 59, "pmc_pair_K3|robust": 5, "pmc_pair_K3|robust+gate": 5, "pmc_state_K2|raw": 68, "pmc_state_K2|raw+gate": 68, "pmc_state_K2|robust": 97, "pmc_state_K2|robust+gate": 92, "pmc_state_K2|true": 9}}
pmcprg WARNINGs: {"other": {"records": 289, "task_steps": 32, "max_value": null, "steps": ["fit", "forecast"]}, "quad_error": {"records": 3710, "task_steps": 131, "max_value": 160.0, "steps": ["choose_nodes", "classify", "fit", "forecast", "quad_error gated", "quad_error observed"]}, "quantile_fallback": {"records": 489, "task_steps": 28, "max_value": null, "steps": ["choose_nodes", "forecast"]}}

## Part 1 — forecasts on real series


### Beijing Aotizhongxin (ln PM2.5, 1 h)

RMSE by horizon (n = 90 origins at h = 1):

| h | PMC pair K=3 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| 1 | 0.396 | 0.591 | 0.396 | 0.398 | 0.397 | 0.410 |
| 2 | 0.606 | 0.710 | 0.595 | 0.595 | 0.595 | 0.639 |
| 3 | 0.723 | 0.800 | 0.711 | 0.712 | 0.711 | 0.777 |
| 6 | 1.040 | 1.087 | 1.011 | 1.012 | 1.011 | 1.109 |
| 12 | 1.084 | 1.115 | 1.070 | 1.070 | 1.070 | 1.201 |
| 24 | 1.214 | 1.207 | 1.187 | 1.187 | 1.187 | 1.489 |
| mean 1–24 | 1.009 | 1.050 | 0.996 | 0.996 | 0.996 | 1.158 |

CRPS by horizon (n = 90 origins at h = 1):

| h | PMC pair K=3 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| 1 | 0.199 | 0.334 | 0.206 | 0.207 | 0.206 | 0.210 |
| 2 | 0.320 | 0.403 | 0.318 | 0.319 | 0.318 | 0.339 |
| 3 | 0.386 | 0.449 | 0.389 | 0.390 | 0.389 | 0.414 |
| 6 | 0.572 | 0.624 | 0.577 | 0.577 | 0.576 | 0.614 |
| 12 | 0.594 | 0.632 | 0.608 | 0.608 | 0.609 | 0.650 |
| 24 | 0.709 | 0.707 | 0.689 | 0.689 | 0.689 | 0.859 |
| mean 1–24 | 0.557 | 0.600 | 0.561 | 0.562 | 0.561 | 0.632 |

Coverage of the central 50 / 80 / 95 % intervals (all horizons), width of the 95 % interval, and CRPS against the Gaussian CRPS of the predictive mean and sd:

| model | cov50 | cov80 | cov95 | cov95 h=1 | cov95 h=24 | width95 | CRPS (Gaussian approx.) | CRPS |
|---|---|---|---|---|---|---|---|---|
| PMC pair K=3 | 0.35 | 0.67 | 0.88 | 0.90 | 0.90 | 3.082 | 0.562 | 0.557 |
| HMC-IN K=3 | 0.35 | 0.68 | 0.89 | 0.92 | 0.89 | 3.434 | 0.605 | 0.600 |
| PMM (pmmforecast) | 0.38 | 0.74 | 0.91 | 0.90 | 0.91 | 3.280 | 0.561 | 0.561 |
| PMM, default starts | 0.38 | 0.74 | 0.91 | 0.89 | 0.91 | 3.279 | 0.562 | 0.562 |
| AR(1) | 0.38 | 0.75 | 0.91 | 0.90 | 0.92 | 3.303 | 0.561 | 0.561 |
| persistence | 0.42 | 0.69 | 0.89 | 0.89 | 0.92 | 4.168 | – | 0.632 |


### tsNH4 (ln NH4, 10 min)

RMSE by horizon (n = 93 origins at h = 1):

| h | PMC state K=2 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| 1 | 0.089 | 0.304 | 0.087 | 0.087 | 0.088 | 0.088 |
| 2 | 0.114 | 0.298 | 0.110 | 0.111 | 0.112 | 0.113 |
| 3 | 0.130 | 0.316 | 0.120 | 0.127 | 0.128 | 0.130 |
| 6 | 0.186 | 0.341 | 0.158 | 0.182 | 0.183 | 0.188 |
| 12 | 0.374 | 0.472 | 0.333 | 0.361 | 0.367 | 0.380 |
| 24 | 0.586 | 0.608 | 0.547 | 0.546 | 0.570 | 0.612 |
| mean 1–24 | 0.388 | 0.471 | 0.352 | 0.368 | 0.379 | 0.400 |

CRPS by horizon (n = 93 origins at h = 1):

| h | PMC state K=2 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| 1 | 0.050 | 0.185 | 0.049 | 0.050 | 0.050 | 0.050 |
| 2 | 0.065 | 0.181 | 0.063 | 0.064 | 0.064 | 0.065 |
| 3 | 0.073 | 0.190 | 0.071 | 0.074 | 0.074 | 0.074 |
| 6 | 0.104 | 0.207 | 0.097 | 0.104 | 0.105 | 0.106 |
| 12 | 0.200 | 0.265 | 0.186 | 0.194 | 0.199 | 0.204 |
| 24 | 0.327 | 0.327 | 0.305 | 0.301 | 0.316 | 0.336 |
| mean 1–24 | 0.194 | 0.260 | 0.181 | 0.185 | 0.191 | 0.198 |

Coverage of the central 50 / 80 / 95 % intervals (all horizons), width of the 95 % interval, and CRPS against the Gaussian CRPS of the predictive mean and sd:

| model | cov50 | cov80 | cov95 | cov95 h=1 | cov95 h=24 | width95 | CRPS (Gaussian approx.) | CRPS |
|---|---|---|---|---|---|---|---|---|
| PMC state K=2 | 0.51 | 0.74 | 0.89 | 0.96 | 0.88 | 1.100 | 0.193 | 0.194 |
| HMC-IN K=3 | 0.52 | 0.84 | 0.99 | 1.00 | 0.96 | 2.167 | 0.250 | 0.260 |
| PMM (pmmforecast) | 0.67 | 0.88 | 0.97 | 0.98 | 0.92 | 1.541 | 0.181 | 0.181 |
| PMM, default starts | 0.61 | 0.81 | 0.91 | 0.99 | 0.88 | 1.211 | 0.185 | 0.185 |
| AR(1) | 0.60 | 0.81 | 0.91 | 0.99 | 0.88 | 1.257 | 0.191 | 0.191 |
| persistence | 0.54 | 0.83 | 0.99 | 0.99 | 0.99 | 1.825 | – | 0.198 |


### Intel Lab mote 20 (°C, 30 s)

RMSE by horizon (n = 106 origins at h = 1):

| h | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|
| 1 | 1.317 | 0.019 | 0.025 | 0.021 | 0.021 |
| 2 | 1.296 | 0.027 | 0.031 | 0.030 | 0.030 |
| 3 | 1.342 | 0.035 | 0.047 | 0.044 | 0.044 |
| 6 | 1.337 | 0.060 | 0.080 | 0.080 | 0.081 |
| 12 | 1.345 | 0.299 | 0.311 | 0.311 | 0.312 |
| 24 | 1.417 | 0.401 | 0.427 | 0.426 | 0.427 |
| mean 1–24 | 1.359 | 0.272 | 0.289 | 0.289 | 0.289 |

CRPS by horizon (n = 106 origins at h = 1):

| h | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|
| 1 | 0.738 | 0.013 | 0.017 | 0.015 | 0.011 |
| 2 | 0.724 | 0.019 | 0.020 | 0.021 | 0.015 |
| 3 | 0.747 | 0.025 | 0.028 | 0.028 | 0.021 |
| 6 | 0.742 | 0.043 | 0.043 | 0.045 | 0.036 |
| 12 | 0.748 | 0.105 | 0.097 | 0.099 | 0.089 |
| 24 | 0.795 | 0.174 | 0.153 | 0.155 | 0.142 |
| mean 1–24 | 0.759 | 0.098 | 0.091 | 0.092 | 0.083 |

Coverage of the central 50 / 80 / 95 % intervals (all horizons), width of the 95 % interval, and CRPS against the Gaussian CRPS of the predictive mean and sd:

| model | cov50 | cov80 | cov95 | cov95 h=1 | cov95 h=24 | width95 | CRPS (Gaussian approx.) | CRPS |
|---|---|---|---|---|---|---|---|---|
| HMC-IN K=3 | 0.44 | 0.78 | 0.93 | 0.94 | 0.94 | 5.161 | 0.756 | 0.759 |
| PMM (pmmforecast) | 0.90 | 0.97 | 0.98 | 1.00 | 0.97 | 1.053 | 0.098 | 0.098 |
| PMM, default starts | 0.78 | 0.90 | 0.94 | 1.00 | 0.92 | 0.580 | 0.091 | 0.091 |
| AR(1) | 0.81 | 0.91 | 0.95 | 1.00 | 0.93 | 0.640 | 0.092 | 0.092 |
| persistence | 0.57 | 0.84 | 0.95 | 0.96 | 0.96 | 0.765 | – | 0.083 |


### Reproduction of real_series §4 (Aotizhongxin)

| model | RMSE h=1 (§4) | RMSE h=1 (here) | RMSE h=6 (§4) | RMSE h=6 (here) | RMSE h=24 (§4) | RMSE h=24 (here) |
|---|---|---|---|---|---|---|
| PMC pair K=3 | 0.396 | 0.396 | 1.040 | 1.040 | 1.214 | 1.214 |
| HMC-IN K=3 | 0.591 | 0.591 | 1.087 | 1.087 | 1.207 | 1.207 |
| AR(1) | 0.397 | 0.397 | 1.011 | 1.011 | 1.187 | 1.187 |
| persistence | 0.410 | 0.410 | 1.109 | 1.109 | 1.489 | 1.489 |


### PMM: empirical MSE against the theoretical MSE (paper Eqs. 18–24)

| series | h | theoretical MSE | empirical MSE | ratio | mean predictive variance |
|---|---|---|---|---|---|
| aotizhongxin | 1 | 0.1161 | 0.1571 | 1.35 | 0.1161 |
| aotizhongxin | 2 | 0.2162 | 0.3536 | 1.64 | 0.2162 |
| aotizhongxin | 3 | 0.3079 | 0.5050 | 1.64 | 0.3079 |
| aotizhongxin | 6 | 0.5283 | 1.0222 | 1.93 | 0.5306 |
| aotizhongxin | 12 | 0.8026 | 1.1444 | 1.43 | 0.8026 |
| aotizhongxin | 24 | 1.0199 | 1.4096 | 1.38 | 1.0202 |
| tsnh4 | 1 | 0.0098 | 0.0076 | 0.77 | 0.0098 |
| tsnh4 | 2 | 0.0206 | 0.0121 | 0.59 | 0.0206 |
| tsnh4 | 3 | 0.0323 | 0.0145 | 0.45 | 0.0323 |
| tsnh4 | 6 | 0.0720 | 0.0251 | 0.35 | 0.0720 |
| tsnh4 | 12 | 0.1645 | 0.1110 | 0.67 | 0.1645 |
| tsnh4 | 24 | 0.3520 | 0.2990 | 0.85 | 0.3520 |
| mote20 | 1 | 0.0015 | 0.0004 | 0.23 | 0.0020 |
| mote20 | 2 | 0.0037 | 0.0007 | 0.20 | 0.0043 |
| mote20 | 3 | 0.0066 | 0.0012 | 0.18 | 0.0074 |
| mote20 | 6 | 0.0199 | 0.0035 | 0.18 | 0.0210 |
| mote20 | 12 | 0.0674 | 0.0892 | 1.32 | 0.0690 |
| mote20 | 24 | 0.2285 | 0.1607 | 0.70 | 0.2317 |


### Paired CRPS differences against the PMM (per-origin mean over h; 95 % bootstrap CI)

| study | model | n_origins | mean_diff | CI | frac_a_better |
|---|---|---|---|---|---|
| aotizhongxin | AR(1) | 91 | +0.0001 | [-0.0009, +0.0011] | 0.47 |
| aotizhongxin | AR(1), raw + gate | 91 | +0.0275 | [-0.0015, +0.0667] | 0.45 |
| aotizhongxin | AR(1), robust + gate | 91 | +0.0723 | [+0.0139, +0.1445] | 0.51 |
| aotizhongxin | HMC-IN K=3 | 91 | +0.0396 | [+0.0144, +0.0672] | 0.33 |
| aotizhongxin | HMC-IN K=3, raw + gate | 91 | +0.0396 | [+0.0142, +0.0667] | 0.33 |
| aotizhongxin | HMC-IN K=3, robust | 91 | +0.0401 | [+0.0128, +0.0658] | 0.33 |
| aotizhongxin | HMC-IN K=3, robust + gate | 91 | +0.0401 | [+0.0151, +0.0664] | 0.33 |
| aotizhongxin | persistence | 91 | +0.0669 | [+0.0125, +0.1220] | 0.49 |
| aotizhongxin | PMC pair K=3 | 91 | -0.0067 | [-0.0333, +0.0182] | 0.52 |
| aotizhongxin | PMC pair K=3, raw + gate | 91 | -0.0069 | [-0.0312, +0.0195] | 0.52 |
| aotizhongxin | PMC pair K=3, robust | 91 | +0.0013 | [-0.0254, +0.0302] | 0.52 |
| aotizhongxin | PMC pair K=3, robust + gate | 91 | +0.0013 | [-0.0259, +0.0295] | 0.52 |
| aotizhongxin | PMM, default starts | 91 | +0.0002 | [-0.0001, +0.0004] | 0.46 |
| mote20 | AR(1) | 120 | -0.0067 | [-0.0147, +0.0036] | 0.84 |
| mote20 | AR(1), raw + gate | 120 | +1.5426 | [+1.0939, +2.0331] | 0.68 |
| mote20 | AR(1), robust + gate | 120 | +4.8902 | [+4.4027, +5.3903] | 0.08 |
| mote20 | HMC-IN K=3 | 120 | +0.6605 | [+0.5620, +0.7646] | 0.01 |
| mote20 | HMC-IN K=3, raw + gate | 120 | +0.6605 | [+0.5613, +0.7668] | 0.01 |
| mote20 | HMC-IN K=3, robust | 120 | +0.6779 | [+0.5714, +0.7907] | 0.01 |
| mote20 | HMC-IN K=3, robust + gate | 120 | +0.6791 | [+0.5693, +0.7950] | 0.01 |
| mote20 | persistence | 120 | -0.0163 | [-0.0263, -0.0052] | 0.84 |
| mote20 | PMM, default starts | 120 | -0.0078 | [-0.0163, +0.0033] | 0.84 |
| tsnh4 | AR(1) | 93 | +0.0101 | [+0.0002, +0.0207] | 0.57 |
| tsnh4 | AR(1), raw + gate | 93 | +0.0131 | [+0.0012, +0.0281] | 0.57 |
| tsnh4 | AR(1), robust + gate | 93 | +0.0177 | [+0.0033, +0.0351] | 0.58 |
| tsnh4 | HMC-IN K=3 | 93 | +0.0785 | [+0.0576, +0.1014] | 0.18 |
| tsnh4 | HMC-IN K=3, raw + gate | 93 | +0.0785 | [+0.0580, +0.0997] | 0.18 |
| tsnh4 | HMC-IN K=3, robust | 93 | +0.0789 | [+0.0574, +0.1002] | 0.19 |
| tsnh4 | HMC-IN K=3, robust + gate | 93 | +0.0789 | [+0.0558, +0.1026] | 0.19 |
| tsnh4 | persistence | 93 | +0.0170 | [+0.0073, +0.0271] | 0.52 |
| tsnh4 | PMC state K=2 | 93 | +0.0126 | [-0.0000, +0.0258] | 0.59 |
| tsnh4 | PMC state K=2, raw + gate | 93 | +0.0126 | [-0.0004, +0.0260] | 0.59 |
| tsnh4 | PMC state K=2, robust | 93 | +0.0143 | [+0.0012, +0.0276] | 0.59 |
| tsnh4 | PMC state K=2, robust + gate | 93 | +0.0148 | [+0.0021, +0.0288] | 0.58 |
| tsnh4 | PMM, default starts | 93 | +0.0040 | [-0.0055, +0.0157] | 0.67 |


### pmmforecast fits: default starts against the AR(1)-embedding start

| name | n_fit | default loglik | ar1-start loglik | AR(1) loglik | chosen_fit | default_c | c | trA | detA | spectral_radius |
|---|---|---|---|---|---|---|---|---|---|---|
| aotizhongxin__clean | 6570 | -2253.0 | -2252.1 | -2258.4 | ar1_start | 0.9458 | 0.9458 | 0.4197 | -0.4995 | 0.9471 |
| tsnh4__clean | 3414 | 3009.4 | 3051.1 | 3017.9 | ar1_start | 0.0773 | 0.9935 | 1.9069 | 0.9084 | 0.9788 |
| mote20__clean | 21599 | 31962.4 | 39308.4 | 28252.4 | ar1_start | 0.4805 | 0.9999 | 1.9294 | 0.9295 | 0.9989 |
| mote20__raw | 21599 | -27216.5 | -25585.3 | -27070.2 | ar1_start | 0.9382 | 0.9594 | 1.2714 | 0.2719 | 0.9993 |
| aotizhongxin__spikes_r0.01_k6_s0 | 6570 | -8244.3 | -8244.3 | -8849.3 | ar1_start | 0.6712 | 0.6712 | 0.9190 | -0.0258 | 0.9463 |
| aotizhongxin__spikes_r0.01_k6_s1 | 6570 | -7991.7 | -7991.7 | -8563.6 | ar1_start | 0.6821 | 0.6821 | 0.9383 | -0.0070 | 0.9457 |
| aotizhongxin__spikes_r0.01_k6_s2 | 6570 | -7959.5 | -7959.5 | -8507.2 | ar1_start | 0.6912 | 0.6912 | 0.9355 | -0.0070 | 0.9429 |
| aotizhongxin__spikes_r0.05_k6_s0 | 6570 | -12268.9 | -12268.9 | -12703.6 | default | 0.3077 | 0.3077 | 0.9062 | -0.0399 | 0.9483 |
| aotizhongxin__spikes_r0.05_k6_s1 | 6570 | -12410.9 | -12410.9 | -12822.5 | default | 0.2969 | 0.2969 | 0.9065 | -0.0382 | 0.9468 |
| aotizhongxin__spikes_r0.05_k6_s2 | 6570 | -12464.2 | -12464.2 | -12884.4 | default | 0.2925 | 0.2925 | 0.8830 | -0.0548 | 0.9412 |

Fixture series (60 fits): the AR(1)-embedding start reaches a higher likelihood in 2, the default starts in 22 (median gain of the better fit over the default 0.00 nats).


### Gating on the clean real series (flags at α = 1e-3 in the test part) and its cost

| series | model | fit | test rows flagged | per 1000 | CRPS | CRPS gated | masked train rows | change % |
|---|---|---|---|---|---|---|---|---|
| aotizhongxin | HMC-IN K=3 | raw | 14 | 6.47 | 0.6002 | 0.6002 | – | +0.0 |
| aotizhongxin | HMC-IN K=3 | robust | 19 | 8.78 | 0.6007 | 0.6007 | 8 | +0.0 |
| mote20 | HMC-IN K=3 | raw | 2 | 0.32 | 0.7592 | 0.7592 | – | +0.0 |
| mote20 | HMC-IN K=3 | robust | 92 | 14.75 | 0.7766 | 0.7778 | 164 | +0.2 |
| tsnh4 | HMC-IN K=3 | raw | 0 | 0.00 | 0.2596 | 0.2596 | – | +0.0 |
| tsnh4 | HMC-IN K=3 | robust | 0 | 0.00 | 0.2600 | 0.2600 | 1 | +0.0 |
| tsnh4 | PMC state K=2 | raw | 2 | 1.76 | 0.1938 | 0.1938 | – | +0.0 |
| tsnh4 | PMC state K=2 | robust | 3 | 2.64 | 0.1954 | 0.1960 | 44 | +0.3 |
| aotizhongxin | PMC pair K=3 | raw | 2 | 0.92 | 0.5568 | 0.5566 | – | -0.0 |
| aotizhongxin | PMC pair K=3 | robust | 0 | 0.00 | 0.5644 | 0.5644 | 18 | +0.0 |
| mote20 | AR(1) | raw | 1658 | 265.88 | 0.0923 | 1.6165 | – | +1651.5 |
| mote20 | AR(1) | robust | 5640 | 904.43 | – | 4.9402 | 13147 | – |
| aotizhongxin | AR(1) | raw | 53 | 24.50 | 0.5614 | 0.5891 | – | +4.9 |
| aotizhongxin | AR(1) | robust | 258 | 119.28 | – | 0.6363 | 548 | – |
| tsnh4 | AR(1) | raw | 4 | 3.51 | 0.1912 | 0.1942 | – | +1.6 |
| tsnh4 | AR(1) | robust | 15 | 13.18 | – | 0.1988 | 140 | – |


### Quadrature: nodes chosen per fitted copula model and quad_error (see Library problems, P1)

G: first of 64 / 128 / 256 nodes whose predictive means and sds (first, middle and last origin) change by less than 1 % of the predictive sd when G is doubled (`check`; `unconverged`: > 1 % at G = 256). `tail_G`: change of the node-law 2.5 / 97.5 % quantiles at the chosen G. `ratio`: pmcprg's `GapPosterior.quad_error` / WARNING limit of the same forecast chains (recorded, not required). Worst values over the fits; `cond_ratio_G`: quad_error / limit of the whole conditioning series, observed and gated (– : no missing row).

| study | family | fits | G64 | G128 | G256 | unconverged | check_64 | check_G | tail_G | ratio_64 | ratio_G | cond_ratio_G |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aotizhongxin | pmc_pair_K3 | 8 | 1 | 2 | 5 | 5 | 4.7e-02 | 2.8e-02 | 9.0e-02 | 5e+02 | 2.3e+02 | 69 |
| fix_pmc_gauss_k2 | pmc_state_K2 | 150 | 150 | 0 | 0 | 0 | 4.4e-04 | 4.4e-04 | 4.5e-02 | 5.6 | 5.6 | 0.00023 |
| tsnh4 | pmc_state_K2 | 2 | 0 | 2 | 0 | 0 | 2.2e-02 | 1.4e-04 | 4.5e-02 | 4.1e+02 | 2.3 | 0.28 |


### pmcprg WARNINGs collected during the campaign

| kind | study | step | template | records | max_value |
|---|---|---|---|---|---|
| other | aotizhongxin | fit | %s iter %d: log-lik regressed by %.4e (streak=%d/%d). | 70 | – |
| other | aotizhongxin | fit | %s: %d consecutive regressions — stopping early at iter %d. | 23 | – |
| other | aotizhongxin | fit | %s: degenerate fitted model — %s. Its likelihood may be unbounded (a collapsing variance o | 4 | – |
| other | fix_pmc_gauss_k2 | fit | %s iter %d: log-lik regressed by %.4e (streak=%d/%d). | 7 | – |
| other | mote20 | fit | %s iter %d: log-lik regressed by %.4e (streak=%d/%d). | 38 | – |
| other | mote20 | fit | %s: %d consecutive regressions — stopping early at iter %d. | 1 | – |
| other | mote20 | fit | Missing-data pass: %s in linear space; recomputing in log space. | 1 | – |
| other | mote20 | forecast | Forward: C=%.3e at step n=%d — the transition weights underflow; recomputing the pass in l | 120 | – |
| other | tsnh4 | fit | %s iter %d: log-lik regressed by %.4e (streak=%d/%d). | 15 | – |
| other | tsnh4 | fit | %s: degenerate fitted model — %s. Its likelihood may be unbounded (a collapsing variance o | 9 | – |
| other | tsnh4 | fit | robust_estimate: the flagged rows still change after %d refit(s) (%d masked, %d flagged by | 1 | – |
| quad_error | aotizhongxin | choose_nodes | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 132 | 2.50e+00 |
| quad_error | aotizhongxin | fit | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 747 | 1.80e+00 |
| quad_error | aotizhongxin | forecast | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 1274 | 1.50e+00 |
| quad_error | aotizhongxin | quad_error gated | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 8 | 3.70e-01 |
| quad_error | aotizhongxin | quad_error observed | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 6 | 3.30e-01 |
| quad_error | fix_pmc_gauss_k2 | choose_nodes | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 32 | 5.60e-03 |
| quad_error | fix_pmc_gauss_k2 | forecast | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 1200 | 2.00e-02 |
| quad_error | mote20 | choose_nodes | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 72 | 8.80e+01 |
| quad_error | mote20 | classify | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 4 | 2.20e+01 |
| quad_error | mote20 | fit | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 103 | 1.60e+02 |
| quad_error | tsnh4 | choose_nodes | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 18 | 4.10e-01 |
| quad_error | tsnh4 | fit | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 43 | 3.40e-01 |
| quad_error | tsnh4 | forecast | Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals the gri | 71 | 8.40e-03 |
| quantile_fallback | aotizhongxin | choose_nodes | Quantiles of the missing values: the interpolated CDF of %d law(s) is not monotone or leav | 50 | – |
| quantile_fallback | aotizhongxin | forecast | Quantiles of the missing values: the interpolated CDF of %d law(s) is not monotone or leav | 420 | – |
| quantile_fallback | fix_pmc_gauss_k2 | forecast | Quantiles of the missing values: the interpolated CDF of %d law(s) is not monotone or leav | 2 | – |
| quantile_fallback | mote20 | choose_nodes | Quantiles of the missing values: the interpolated CDF of %d law(s) is not monotone or leav | 14 | – |
| quantile_fallback | tsnh4 | choose_nodes | Quantiles of the missing values: the interpolated CDF of %d law(s) is not monotone or leav | 3 | – |


## Part 2 — Gaussian cross-check

| tuple | abcde | d_minus_bc | e_minus_bc | Y_is_AR1 | c | gamma2 | c2 | ll_pmm | ll_pmc_ar1map | ll_ar1_closed | ll_pmc_minus_pmm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d_eq_bc | 0.8 0.5 0.7 0.35 0.6 | +0.0000 | +0.2500 | True | 0.700 | 0.4900 | 0.4900 | -755.0942 | -755.0942 | -755.0942 | +2.96e-12 |
| e_eq_bc | 0.8 0.5 0.7 0.6 0.35 | +0.2500 | +0.0000 | True | 0.700 | 0.4900 | 0.4900 | -773.5301 | -773.5301 | -773.5301 | +1.59e-12 |
| generic | 0.8 0.5 0.7 0.5 0.65 | +0.1500 | +0.3000 | False | 0.700 | 0.5500 | 0.4900 | -728.3943 | -757.1987 | -757.1987 | -2.88e+01 |
| hmm | 0.9 0.8 0.576 0.72 0.72 | +0.2592 | +0.2592 | False | 0.576 | 0.5184 | 0.3318 | -963.8185 | -1057.5029 | -1057.5029 | -9.37e+01 |

Forecasts at origins 50, 500, 1000, 2000, h = 1…20 (data scale y = 2 + 0.5 Y): largest absolute differences.

| tuple | max_abs_mean_pmc_vs_ar1 | max_abs_sd_pmc_vs_ar1 | max_abs_quantile_pmc_vs_gauss | max_abs_mean_pmm_vs_ar1 | max_abs_sd_pmm_vs_ar1 | max_abs_theoryMSE_vs_pmm_var |
|---|---|---|---|---|---|---|
| d_eq_bc | 3.1e-08 | 1.7e-07 | 1.0e-06 | 2.2e-16 | 1.7e-16 | 0.0e+00 |
| e_eq_bc | 1.1e-08 | 8.5e-08 | 3.8e-07 | 2.2e-16 | 5.6e-17 | 0.0e+00 |
| generic | 6.2e-09 | 8.5e-08 | 2.5e-07 | 5.4e-02 | 3.0e-02 | 0.0e+00 |
| hmm | 1.4e-08 | 8.1e-08 | 4.4e-07 | 3.5e-01 | 6.2e-02 | 0.0e+00 |

Monte-Carlo MSE (40 000-step path, 975 origins; model scale):

| tuple | h | theory_pmm | one_minus_c2h | theory_ar1map | emp_pmm | se_pmm | emp_pmc_ar1map | se_pmc_ar1map |
|---|---|---|---|---|---|---|---|---|
| d_eq_bc | 1 | 0.5100 | 0.5100 | 0.5100 | 0.4999 | 0.0229 | 0.4999 | 0.0229 |
| d_eq_bc | 2 | 0.7599 | 0.7599 | 0.7599 | 0.7532 | 0.0350 | 0.7532 | 0.0350 |
| d_eq_bc | 3 | 0.8824 | 0.8824 | 0.8824 | 0.9184 | 0.0418 | 0.9184 | 0.0418 |
| d_eq_bc | 5 | 0.9718 | 0.9718 | 0.9718 | 0.9576 | 0.0416 | 0.9576 | 0.0416 |
| d_eq_bc | 10 | 0.9992 | 0.9992 | 0.9992 | 1.0251 | 0.0454 | 1.0251 | 0.0454 |
| e_eq_bc | 1 | 0.5100 | 0.5100 | 0.5100 | 0.4908 | 0.0214 | 0.4908 | 0.0214 |
| e_eq_bc | 2 | 0.7599 | 0.7599 | 0.7599 | 0.7619 | 0.0336 | 0.7619 | 0.0336 |
| e_eq_bc | 3 | 0.8824 | 0.8824 | 0.8824 | 0.9246 | 0.0433 | 0.9246 | 0.0433 |
| e_eq_bc | 5 | 0.9718 | 0.9718 | 0.9718 | 0.9860 | 0.0434 | 0.9860 | 0.0434 |
| e_eq_bc | 10 | 0.9992 | 0.9992 | 0.9992 | 0.9610 | 0.0466 | 0.9610 | 0.0466 |
| generic | 1 | 0.4985 | 0.5100 | 0.5100 | 0.4759 | 0.0220 | 0.4925 | 0.0224 |
| generic | 2 | 0.6800 | 0.7599 | 0.7011 | 0.6643 | 0.0274 | 0.6853 | 0.0280 |
| generic | 3 | 0.7725 | 0.8824 | 0.8028 | 0.7323 | 0.0325 | 0.7676 | 0.0336 |
| generic | 5 | 0.8720 | 0.9718 | 0.9133 | 0.9241 | 0.0415 | 0.9865 | 0.0439 |
| generic | 10 | 0.9665 | 0.9992 | 0.9909 | 0.9310 | 0.0404 | 0.9554 | 0.0409 |
| hmm | 1 | 0.5975 | 0.6682 | 0.6682 | 0.5769 | 0.0262 | 0.6344 | 0.0290 |
| hmm | 2 | 0.6740 | 0.8899 | 0.7661 | 0.6567 | 0.0292 | 0.7210 | 0.0313 |
| hmm | 3 | 0.7359 | 0.9635 | 0.8582 | 0.7216 | 0.0350 | 0.8197 | 0.0379 |
| hmm | 5 | 0.8267 | 0.9960 | 0.9561 | 0.8356 | 0.0371 | 0.9594 | 0.0417 |
| hmm | 10 | 0.9396 | 1.0000 | 0.9982 | 0.9934 | 0.0470 | 1.0483 | 0.0503 |

Identifiability: Y-only negative log-likelihood of pmmforecast on the 'generic' path (`neg_log_likelihood_y_only(tuple, y)`) for tuples with the same (c, tr A, det A), and the tuple returned by `estimate_pmm_mle_y_only(y, n_restarts=4, seed=0)`:

| tuple | abcde | c | trA | detA | nll_y_only |
|---|---|---|---|---|---|
| generic | 0.8000 0.5000 0.7000 0.5000 0.6500 | 0.7000 | 1.2333 | 0.3133 | 2114.688681 |
| sign_flip(b,d,e) | 0.8000 -0.5000 0.7000 -0.5000 -0.6500 | 0.7000 | 1.2333 | 0.3133 | 2114.688681 |
| swap(d,e) | 0.8000 0.5000 0.7000 0.6500 0.5000 | 0.7000 | 1.2333 | 0.3133 | 2114.688681 |
| same_law_b=0.2 | 0.6458 0.2000 0.7000 0.2934 0.5156 | 0.7000 | 1.2333 | 0.3133 | 2114.688681 |
| same_law_b=0.7 | 0.8825 0.7000 0.7000 0.7579 0.6042 | 0.7000 | 1.2333 | 0.3133 | 2114.688681 |
| Y-only MLE (n_restarts=4) | 0.3532 0.6019 0.6799 0.0144 0.3060 | 0.6799 | 1.3177 | 0.3697 | 2114.326530 |


## Part 3 — robust forecasting


### Simulated fixtures (report/erroneous_data), spikes of 6 sd

**HMC-IN fixture** — CRPS averaged over h = 1…10, 99 origins and the replicates; and at h = 1:

| method | CRPS 0% | CRPS h=1 0% | CRPS 1% | CRPS h=1 1% | CRPS 5% | CRPS h=1 5% |
|---|---|---|---|---|---|---|
| HMC-IN K=2, true model | 0.768 | 0.670 | 0.767 | 0.669 | 0.777 | 0.698 |
| HMC-IN K=2, true model + gate | 0.768 | 0.670 | 0.768 | 0.669 | 0.770 | 0.675 |
| HMC-IN K=2 | 0.768 | 0.670 | 0.778 | 0.689 | 0.818 | 0.805 |
| HMC-IN K=2, raw + gate | 0.768 | 0.670 | 0.778 | 0.689 | 0.818 | 0.805 |
| HMC-IN K=2, robust | 0.768 | 0.670 | 0.768 | 0.670 | 0.817 | 0.805 |
| HMC-IN K=2, robust + gate | 0.768 | 0.670 | 0.768 | 0.670 | 0.817 | 0.805 |
| HMC-IN K=2, Hampel robust + gate | 0.768 | 0.670 | 0.768 | 0.670 | 0.770 | 0.675 |
| HMC-IN K=2, clean fit | 0.768 | 0.670 | 0.768 | 0.670 | 0.768 | 0.670 |
| PMM (pmmforecast) | 0.774 | 0.689 | 0.782 | 0.709 | 0.869 | 0.827 |
| AR(1) | 0.790 | 0.708 | 0.797 | 0.731 | 0.878 | 0.856 |
| AR(1), raw + gate | 0.790 | 0.708 | 0.797 | 0.727 | 0.877 | 0.851 |
| AR(1), robust + gate | 0.790 | 0.707 | 0.790 | 0.708 | 0.791 | 0.714 |
| AR(1), clean fit | 0.790 | 0.708 | 0.790 | 0.708 | 0.790 | 0.708 |
| persistence | 0.983 | 0.828 | 1.010 | 0.861 | 1.310 | 1.183 |

**PMC fixture** — CRPS averaged over h = 1…10, 99 origins and the replicates; and at h = 1:

| method | CRPS 0% | CRPS h=1 0% | CRPS 1% | CRPS h=1 1% | CRPS 5% | CRPS h=1 5% |
|---|---|---|---|---|---|---|
| PMC state K=2, true model | 0.740 | 0.482 | 0.755 | 0.542 | 0.800 | 0.702 |
| PMC state K=2, true model + gate | 0.740 | 0.482 | 0.741 | 0.485 | 0.743 | 0.492 |
| PMC state K=2 | 0.742 | 0.482 | 0.751 | 0.511 | 0.769 | 0.530 |
| PMC state K=2, raw + gate | 0.742 | 0.483 | 0.749 | 0.496 | 0.769 | 0.530 |
| PMC state K=2, robust | 0.742 | 0.482 | 0.757 | 0.541 | 0.769 | 0.530 |
| PMC state K=2, robust + gate | 0.743 | 0.486 | 0.744 | 0.489 | 0.769 | 0.530 |
| PMC state K=2, Hampel robust + gate | 0.743 | 0.486 | 0.744 | 0.489 | 0.746 | 0.496 |
| PMC state K=2, clean fit | 0.742 | 0.482 | 0.742 | 0.482 | 0.742 | 0.482 |
| PMM (pmmforecast) | 0.750 | 0.495 | 0.771 | 0.598 | 0.870 | 0.781 |
| AR(1) | 0.746 | 0.494 | 0.779 | 0.609 | 0.883 | 0.816 |
| AR(1), raw + gate | 0.758 | 0.540 | 0.773 | 0.567 | 0.881 | 0.797 |
| AR(1), robust + gate | 0.770 | 0.552 | 0.770 | 0.553 | 0.772 | 0.557 |
| AR(1), clean fit | 0.746 | 0.494 | 0.746 | 0.494 | 0.746 | 0.494 |
| persistence | 0.924 | 0.504 | 1.003 | 0.595 | 1.219 | 0.838 |

One-step forecasts whose last conditioning row is a spike, against the others (h = 1):

| fixture | rate | method | n spike last | CRPS h=1, spike last | CRPS h=1, otherwise |
|---|---|---|---|---|---|
| HMC-IN fixture | 1% | HMC-IN K=2 | 5 | 0.644 | 0.689 |
| HMC-IN fixture | 5% | HMC-IN K=2 | 50 | 0.846 | 0.803 |
| HMC-IN fixture | 1% | HMC-IN K=2, raw + gate | 5 | 0.748 | 0.689 |
| HMC-IN fixture | 5% | HMC-IN K=2, raw + gate | 50 | 0.846 | 0.803 |
| HMC-IN fixture | 1% | HMC-IN K=2, robust + gate | 5 | 0.484 | 0.670 |
| HMC-IN fixture | 5% | HMC-IN K=2, robust + gate | 50 | 0.846 | 0.803 |
| HMC-IN fixture | 1% | HMC-IN K=2, Hampel robust + gate | 5 | 0.484 | 0.670 |
| HMC-IN fixture | 5% | HMC-IN K=2, Hampel robust + gate | 50 | 0.754 | 0.671 |
| HMC-IN fixture | 1% | HMC-IN K=2, true model | 5 | 0.595 | 0.670 |
| HMC-IN fixture | 5% | HMC-IN K=2, true model | 50 | 1.150 | 0.674 |
| HMC-IN fixture | 1% | HMC-IN K=2, true model + gate | 5 | 0.480 | 0.670 |
| HMC-IN fixture | 5% | HMC-IN K=2, true model + gate | 50 | 0.759 | 0.671 |
| HMC-IN fixture | 1% | PMM (pmmforecast) | 5 | 1.013 | 0.708 |
| HMC-IN fixture | 5% | PMM (pmmforecast) | 50 | 1.094 | 0.812 |
| HMC-IN fixture | 1% | AR(1) | 5 | 1.385 | 0.728 |
| HMC-IN fixture | 5% | AR(1) | 50 | 1.229 | 0.836 |
| HMC-IN fixture | 1% | AR(1), raw + gate | 5 | 0.631 | 0.728 |
| HMC-IN fixture | 5% | AR(1), raw + gate | 50 | 1.136 | 0.836 |
| HMC-IN fixture | 1% | AR(1), robust + gate | 5 | 0.598 | 0.709 |
| HMC-IN fixture | 5% | AR(1), robust + gate | 50 | 0.820 | 0.708 |
| HMC-IN fixture | 1% | persistence | 5 | 6.841 | 0.831 |
| HMC-IN fixture | 5% | persistence | 50 | 7.122 | 0.867 |
| PMC fixture | 1% | PMC state K=2 | 12 | 2.379 | 0.488 |
| PMC fixture | 5% | PMC state K=2 | 43 | 0.942 | 0.512 |
| PMC fixture | 1% | PMC state K=2, raw + gate | 12 | 1.131 | 0.488 |
| PMC fixture | 5% | PMC state K=2, raw + gate | 43 | 0.942 | 0.512 |
| PMC fixture | 1% | PMC state K=2, robust + gate | 12 | 0.552 | 0.488 |
| PMC fixture | 5% | PMC state K=2, robust + gate | 43 | 0.942 | 0.512 |
| PMC fixture | 1% | PMC state K=2, Hampel robust + gate | 12 | 0.552 | 0.488 |
| PMC fixture | 5% | PMC state K=2, Hampel robust + gate | 43 | 0.703 | 0.487 |
| PMC fixture | 1% | PMC state K=2, true model | 12 | 5.236 | 0.484 |
| PMC fixture | 5% | PMC state K=2, true model | 43 | 5.438 | 0.487 |
| PMC fixture | 1% | PMC state K=2, true model + gate | 12 | 0.552 | 0.484 |
| PMC fixture | 5% | PMC state K=2, true model + gate | 43 | 0.701 | 0.483 |
| PMC fixture | 1% | PMM (pmmforecast) | 12 | 3.023 | 0.568 |
| PMC fixture | 5% | PMM (pmmforecast) | 43 | 1.380 | 0.754 |
| PMC fixture | 1% | AR(1) | 12 | 4.079 | 0.566 |
| PMC fixture | 5% | AR(1) | 43 | 1.790 | 0.772 |
| PMC fixture | 1% | AR(1), raw + gate | 12 | 0.670 | 0.566 |
| PMC fixture | 5% | AR(1), raw + gate | 43 | 1.347 | 0.772 |
| PMC fixture | 1% | AR(1), robust + gate | 12 | 0.375 | 0.555 |
| PMC fixture | 5% | AR(1), robust + gate | 43 | 0.712 | 0.550 |
| PMC fixture | 1% | persistence | 12 | 7.813 | 0.507 |
| PMC fixture | 5% | persistence | 43 | 7.398 | 0.541 |


### Beijing Aotizhongxin with injected spikes (6 sd), 3 seeds per rate

CRPS averaged over h = 1…24 and the origins (and the seeds), and at h = 1 (the copula PMC ran on seed 0 only: compare it in the seed-0 table below):

| method | CRPS clean | CRPS h=1 clean | CRPS 1% | CRPS h=1 1% | CRPS 5% | CRPS h=1 5% |
|---|---|---|---|---|---|---|
| HMC-IN K=3 | 0.600 | 0.334 | 0.626 | 0.419 | 0.641 | 0.439 |
| HMC-IN K=3, raw + gate | 0.600 | 0.334 | 0.626 | 0.419 | 0.641 | 0.439 |
| HMC-IN K=3, robust | 0.601 | 0.334 | 0.627 | 0.420 | 0.641 | 0.439 |
| HMC-IN K=3, robust + gate | 0.601 | 0.334 | 0.627 | 0.420 | 0.641 | 0.439 |
| HMC-IN K=3, Hampel robust + gate | – | – | 0.602 | 0.337 | 0.601 | 0.336 |
| PMC pair K=3 | 0.557 | 0.199 | 0.556 | 0.197 | 0.561 | 0.207 |
| PMC pair K=3, raw + gate | 0.557 | 0.199 | 0.556 | 0.197 | 0.561 | 0.207 |
| PMC pair K=3, robust | 0.564 | 0.200 | 0.558 | 0.198 | 0.565 | 0.205 |
| PMC pair K=3, robust + gate | 0.564 | 0.200 | 0.558 | 0.198 | 0.566 | 0.205 |
| PMC pair K=3, Hampel robust + gate | – | – | 0.564 | 0.198 | 0.566 | 0.205 |
| PMM (pmmforecast) | 0.561 | 0.206 | 0.585 | 0.327 | 0.652 | 0.508 |
| AR(1) | 0.561 | 0.206 | 0.648 | 0.314 | 0.718 | 0.616 |
| AR(1), raw + gate | 0.589 | 0.272 | 0.648 | 0.302 | 0.717 | 0.592 |
| AR(1), robust + gate | 0.636 | 0.313 | 0.637 | 0.314 | 0.648 | 0.333 |
| persistence | 0.632 | 0.210 | 0.650 | 0.231 | 0.898 | 0.532 |

Seed 0 only (every method ran on the same contaminated series):

| method | CRPS clean | CRPS h=1 clean | CRPS 1% | CRPS h=1 1% | CRPS 5% | CRPS h=1 5% |
|---|---|---|---|---|---|---|
| HMC-IN K=3 | 0.600 | 0.334 | 0.626 | 0.419 | 0.642 | 0.436 |
| HMC-IN K=3, raw + gate | 0.600 | 0.334 | 0.626 | 0.419 | 0.642 | 0.436 |
| HMC-IN K=3, robust | 0.601 | 0.334 | 0.626 | 0.420 | 0.641 | 0.436 |
| HMC-IN K=3, robust + gate | 0.601 | 0.334 | 0.626 | 0.420 | 0.641 | 0.436 |
| HMC-IN K=3, Hampel robust + gate | – | – | 0.602 | 0.335 | 0.602 | 0.338 |
| PMC pair K=3 | 0.557 | 0.199 | 0.556 | 0.197 | 0.561 | 0.207 |
| PMC pair K=3, raw + gate | 0.557 | 0.199 | 0.556 | 0.197 | 0.561 | 0.207 |
| PMC pair K=3, robust | 0.564 | 0.200 | 0.558 | 0.198 | 0.565 | 0.205 |
| PMC pair K=3, robust + gate | 0.564 | 0.200 | 0.558 | 0.198 | 0.566 | 0.205 |
| PMC pair K=3, Hampel robust + gate | – | – | 0.564 | 0.198 | 0.566 | 0.205 |
| PMM (pmmforecast) | 0.561 | 0.206 | 0.595 | 0.348 | 0.649 | 0.496 |
| AR(1) | 0.561 | 0.206 | 0.649 | 0.309 | 0.713 | 0.588 |
| AR(1), raw + gate | 0.589 | 0.272 | 0.649 | 0.309 | 0.713 | 0.574 |
| AR(1), robust + gate | 0.636 | 0.313 | 0.636 | 0.313 | 0.650 | 0.329 |
| persistence | 0.632 | 0.210 | 0.631 | 0.210 | 0.817 | 0.415 |


### Intel Lab mote 20 as recorded (3 suspect readings of −38.4 °C in the training part)

CRPS averaged over h = 1…24 and the origins (and the seeds), and at h = 1:

| method | CRPS suspect removed | RMSE h=1 suspect removed | CRPS as recorded | RMSE h=1 as recorded |
|---|---|---|---|---|
| HMC-IN K=3 | 0.759 | 1.317 | 0.750 | 1.303 |
| HMC-IN K=3, raw + gate | 0.759 | 1.317 | 0.750 | 1.303 |
| HMC-IN K=3, robust | 0.777 | 1.325 | 0.777 | 1.325 |
| HMC-IN K=3, robust + gate | 0.778 | 1.335 | 0.778 | 1.335 |
| HMC-IN K=3, Hampel robust + gate | – | – | 0.778 | 1.335 |
| PMM (pmmforecast) | 0.098 | 0.019 | 0.252 | 0.066 |
| AR(1) | 0.092 | 0.021 | 0.969 | 0.221 |
| AR(1), raw + gate | 1.617 | 3.384 | 0.969 | 0.221 |
| AR(1), robust + gate | 4.940 | 5.873 | 4.940 | 5.873 |
| persistence | 0.083 | 0.021 | 0.083 | 0.021 |


### Robust fits on the real series

| case | family | fit | n_fits | converged | mask_n_flag | mask_detect | mask_false_per_1000 | prescreen_detect | degenerate | gap_nodes | quad_check | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aotizhongxin__clean | hmc_in_K3 | robust | 4 | True | 8 | – | 1.22 | – | False | 64 | 0.0e+00 | 9 |
| mote20__clean | hmc_in_K3 | robust | 5 | True | 164 | – | 9.02 | – | False | 64 | 0.0e+00 | 13 |
| tsnh4__clean | hmc_in_K3 | robust | 2 | True | 1 | – | 0.29 | – | False | 64 | 0.0e+00 | 2 |
| aotizhongxin__spikes_r0.01_k6_s0 | hmc_in_K3 | robust | 2 | True | 1 | 0.00 | 0.15 | – | False | 64 | 0.0e+00 | 2 |
| aotizhongxin__spikes_r0.01_k6_s0 | hmc_in_K3 | hampel | 4 | True | 79 | 1.00 | 1.23 | 0.94 | False | 64 | 0.0e+00 | 9 |
| mote20__raw | hmc_in_K3 | robust | 5 | True | 167 | 1.00 | 9.02 | – | False | 64 | 0.0e+00 | 11 |
| mote20__raw | hmc_in_K3 | hampel | 5 | True | 167 | 1.00 | 9.02 | 1.00 | False | 64 | 0.0e+00 | 11 |
| aotizhongxin__spikes_r0.01_k6_s1 | hmc_in_K3 | robust | 2 | True | 1 | 0.00 | 0.15 | – | False | 64 | 0.0e+00 | 2 |
| aotizhongxin__spikes_r0.01_k6_s1 | hmc_in_K3 | hampel | 4 | True | 72 | 1.00 | 1.23 | 0.95 | False | 64 | 0.0e+00 | 7 |
| aotizhongxin__spikes_r0.01_k6_s2 | hmc_in_K3 | robust | 7 | True | 12 | 0.18 | 0.15 | – | False | 64 | 0.0e+00 | 7 |
| aotizhongxin__spikes_r0.01_k6_s2 | hmc_in_K3 | hampel | 3 | True | 69 | 1.00 | 1.23 | 1.00 | False | 64 | 0.0e+00 | 8 |
| aotizhongxin__spikes_r0.05_k6_s0 | hmc_in_K3 | robust | 2 | True | 1 | 0.00 | 0.16 | – | False | 64 | 0.0e+00 | 2 |
| aotizhongxin__spikes_r0.05_k6_s0 | hmc_in_K3 | hampel | 4 | True | 329 | 1.00 | 1.12 | 0.97 | False | 64 | 0.0e+00 | 8 |
| aotizhongxin__spikes_r0.05_k6_s1 | hmc_in_K3 | robust | 1 | True | 0 | 0.00 | 0.00 | – | False | 64 | 0.0e+00 | 1 |
| aotizhongxin__spikes_r0.05_k6_s1 | hmc_in_K3 | hampel | 4 | True | 344 | 1.00 | 0.96 | 0.97 | False | 64 | 0.0e+00 | 8 |
| aotizhongxin__spikes_r0.05_k6_s2 | hmc_in_K3 | robust | 2 | True | 1 | 0.00 | 0.16 | – | False | 64 | 0.0e+00 | 2 |
| aotizhongxin__spikes_r0.05_k6_s2 | hmc_in_K3 | hampel | 5 | True | 352 | 1.00 | 1.13 | 0.96 | False | 64 | 0.0e+00 | 9 |
| tsnh4__clean | pmc_state_K2 | robust | 11 | False | 44 | – | 12.89 | – | True | 128 | 1.4e-04 | 54 |
| aotizhongxin__clean | pmc_pair_K3 | robust | 4 | True | 18 | – | 2.74 | – | False | 128 | 4.2e-03 | 162 |
| mote20__clean | ar1 | robust | 5 | True | 13147 | – | 722.72 | – | – | – | – | 3 |
| mote20__raw | ar1 | robust | 6 | True | 13150 | 1.00 | 722.72 | – | – | – | – | 3 |
| aotizhongxin__clean | ar1 | robust | 9 | True | 548 | – | 83.54 | – | – | – | – | 2 |
| tsnh4__clean | ar1 | robust | 7 | True | 140 | – | 41.01 | – | – | – | – | 1 |
| aotizhongxin__spikes_r0.01_k6_s0 | ar1 | robust | 10 | True | 612 | 1.00 | 83.37 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.01_k6_s1 | ar1 | robust | 10 | True | 599 | 1.00 | 82.36 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.01_k6_s2 | ar1 | robust | 11 | True | 601 | 1.00 | 83.09 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.05_k6_s0 | ar1 | robust | 11 | True | 853 | 1.00 | 85.12 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.05_k6_s1 | ar1 | robust | 11 | True | 850 | 1.00 | 82.29 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.05_k6_s2 | ar1 | robust | 11 | False | 825 | 1.00 | 77.23 | – | – | – | – | 2 |
| aotizhongxin__spikes_r0.05_k6_s0 | pmc_pair_K3 | robust | 4 | True | 21 | 0.01 | 2.89 | – | False | 256 | 2.8e-02 | 79 |
| aotizhongxin__spikes_r0.05_k6_s0 | pmc_pair_K3 | hampel | 11 | True | 21 | 0.01 | 2.89 | 0.97 | False | 256 | 2.8e-02 | 238 |
| aotizhongxin__spikes_r0.01_k6_s0 | pmc_pair_K3 | robust | 7 | True | 13 | 0.10 | 0.92 | – | False | 256 | 2.4e-02 | 98 |
| aotizhongxin__spikes_r0.01_k6_s0 | pmc_pair_K3 | hampel | 5 | True | 87 | 1.00 | 2.47 | 0.94 | True | 128 | 5.2e-03 | 133 |


### Robust fits on the fixtures (means over replicates)

| fixture | rate | fit | n | n_fits | converged | mask_detect | mask_false_per_1000 | gate_test_detect | gate_test_false_per_1000 | degenerate |
|---|---|---|---|---|---|---|---|---|---|---|
| hmc_in_gauss_k2 | 0% | hampel | 10 | 2.1 | 1.00 | – | 1.050 | – | 1.400 | 0.000 |
| hmc_in_gauss_k2 | 0% | raw | 10 | – | 0.00 | – | – | – | 1.300 | 0.000 |
| hmc_in_gauss_k2 | 0% | robust | 10 | 1.9 | 1.00 | – | 1.000 | – | 1.400 | 0.000 |
| hmc_in_gauss_k2 | 1% | hampel | 10 | 2.4 | 1.00 | 1.000 | 0.960 | 1.000 | 1.412 | 0.000 |
| hmc_in_gauss_k2 | 1% | raw | 10 | – | 0.00 | – | – | 0.989 | 0.403 | 0.000 |
| hmc_in_gauss_k2 | 1% | robust | 10 | 2.8 | 1.00 | 1.000 | 0.960 | 1.000 | 1.412 | 0.000 |
| hmc_in_gauss_k2 | 5% | hampel | 10 | 2.7 | 1.00 | 1.000 | 0.999 | 1.000 | 1.368 | 0.000 |
| hmc_in_gauss_k2 | 5% | raw | 10 | – | 0.00 | – | – | 0.004 | 0.000 | 0.000 |
| hmc_in_gauss_k2 | 5% | robust | 10 | 1.5 | 1.00 | 0.007 | 0.000 | 0.007 | 0.000 | 0.000 |
| pmc_gauss_k2 | 0% | hampel | 10 | 2.2 | 1.00 | – | 1.450 | – | 1.100 | 0.000 |
| pmc_gauss_k2 | 0% | raw | 10 | – | 0.00 | – | – | – | 0.900 | 0.000 |
| pmc_gauss_k2 | 0% | robust | 10 | 2.0 | 1.00 | – | 1.450 | – | 1.100 | 0.000 |
| pmc_gauss_k2 | 1% | hampel | 10 | 2.4 | 1.00 | 1.000 | 1.466 | 1.000 | 1.111 | 0.000 |
| pmc_gauss_k2 | 1% | raw | 10 | – | 0.00 | – | – | 0.772 | 0.000 | 0.000 |
| pmc_gauss_k2 | 1% | robust | 10 | 3.6 | 1.00 | 1.000 | 1.466 | 1.000 | 1.111 | 0.000 |
| pmc_gauss_k2 | 5% | hampel | 10 | 3.0 | 1.00 | 1.000 | 1.526 | 1.000 | 0.947 | 0.000 |
| pmc_gauss_k2 | 5% | raw | 10 | – | 0.00 | – | – | 0.000 | 0.000 | 0.000 |
| pmc_gauss_k2 | 5% | robust | 10 | 1.1 | 1.00 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |


### AR(1) with innovation gating: flags

| case | fit | phi | sigma2 | n_fits | converged | gate_train_n_flag | mask_n_flag | mask_detect | gate_test_n_flag | gate_test_detect | gate_test_false_per_1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mote20__clean | raw | 0.9999 | 0.002332 | – | – | 6478 | – | – | 1658 | – | 265.876 |
| mote20__clean | robust | 1.0000 | 0.000122 | 5 | True | – | 13147 | – | 5640 | – | 904.426 |
| mote20__raw | raw | 0.9523 | 1.03 | – | – | 3 | – | – | 0 | – | 0.000 |
| mote20__raw | robust | 1.0000 | 0.000122 | 6 | True | – | 13150 | 1.000 | 5640 | – | 904.426 |
| aotizhongxin__clean | raw | 0.9459 | 0.1164 | – | – | 182 | – | – | 53 | – | 24.503 |
| aotizhongxin__clean | robust | 0.9741 | 0.04698 | 9 | True | – | 548 | – | 258 | – | 119.279 |
| tsnh4__clean | raw | 0.9932 | 0.00998 | – | – | 32 | – | – | 4 | – | 3.515 |
| tsnh4__clean | robust | 0.9956 | 0.006309 | 7 | True | – | 140 | – | 15 | – | 13.181 |
| aotizhongxin__spikes_r0.01_k6_s0 | raw | 0.6708 | 0.8689 | – | – | 72 | – | – | 17 | 1.000 | 0.000 |
| aotizhongxin__spikes_r0.01_k6_s0 | robust | 0.9739 | 0.04726 | 10 | True | – | 612 | 1.000 | 273 | 1.000 | 119.292 |
| aotizhongxin__spikes_r0.01_k6_s1 | raw | 0.6817 | 0.7964 | – | – | 66 | – | – | 24 | 1.000 | 0.000 |
| aotizhongxin__spikes_r0.01_k6_s1 | robust | 0.9739 | 0.04722 | 10 | True | – | 599 | 1.000 | 279 | 1.000 | 119.215 |
| aotizhongxin__spikes_r0.01_k6_s2 | raw | 0.6908 | 0.7828 | – | – | 63 | – | – | 27 | 1.000 | 0.000 |
| aotizhongxin__spikes_r0.01_k6_s2 | robust | 0.9739 | 0.0474 | 11 | True | – | 601 | 1.000 | 278 | 1.000 | 117.509 |
| aotizhongxin__spikes_r0.05_k6_s0 | raw | 0.3066 | 2.815 | – | – | 265 | – | – | 91 | 0.784 | 0.000 |
| aotizhongxin__spikes_r0.05_k6_s0 | robust | 0.9740 | 0.04725 | 11 | True | – | 853 | 1.000 | 369 | 1.000 | 123.596 |
| aotizhongxin__spikes_r0.05_k6_s1 | raw | 0.2941 | 2.919 | – | – | 282 | – | – | 74 | 0.740 | 0.000 |
| aotizhongxin__spikes_r0.05_k6_s1 | robust | 0.9739 | 0.04723 | 11 | True | – | 850 | 1.000 | 350 | 1.000 | 121.183 |
| aotizhongxin__spikes_r0.05_k6_s2 | raw | 0.2908 | 2.975 | – | – | 283 | – | – | 73 | 0.785 | 0.000 |
| aotizhongxin__spikes_r0.05_k6_s2 | robust | 0.9732 | 0.049 | 11 | False | – | 825 | 1.000 | 335 | 1.000 | 116.908 |

