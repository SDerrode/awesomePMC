# UCI Human Activity Recognition Using Smartphones — local cache

This directory is the local cache used by
[`examples/uci_har_smartphone.ipynb`](../../examples/uci_har_smartphone.ipynb)
for the **UCI HAR** smartphone-accelerometer benchmark.

**Nothing in this directory is committed to the repository.** The
notebook downloads the dataset zip on first run (with the user's
explicit consent), extracts it here, and re-uses the cache on
subsequent runs.

If the download is skipped (no network, user declines, archive
unreachable, …) the notebook transparently falls back to a synthetic
3-D signal simulated from the bundled
[`hmc_in_mvn_k2_d3.toml`](../../prg/pmc/models/hmc_in_mvn_k2_d3.toml)
fixture — the rest of the notebook (model fit, ICE, SEM, evaluation)
works identically.

## Dataset

- **Title**: Human Activity Recognition Using Smartphones
- **UCI ID**: 240
- **URL**: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **License**: cited in the dataset's own `README.txt`. Please consult
  it before redistributing any derived artefact.

## Reference

D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz,
*A Public Domain Dataset for Human Activity Recognition Using
Smartphones*, ESANN 2013.
