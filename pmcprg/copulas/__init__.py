"""pmcprg.copulas — public API.

All copula families are imported automatically from CopulaEnum: adding a new
family only requires registering its entry in CopulaEnum (with the MODULE
field set) and writing the module file; this __init__.py stays untouched.
"""

import importlib

# Apply package-wide matplotlib defaults (white facecolor, dpi=150, font_size=12).
import pmcprg.plot_style  # noqa: F401

from pmcprg.copulas._base     import CopulaEnum, CopulaVirt, FitResult, GoFResult
from pmcprg.copulas.bivariate import BivariateLaw, ConditionalLaw, BivariateFitResult, BivariateBootstrapCI
from pmcprg.copulas._stderr   import (IndependenceLRTest, MleTauDiscrepancyTest,
                                   StandardErrors, SubmodelLRTest, independence_lr_test,
                                   mle_tau_discrepancy_test, standard_errors, submodel_lr_test)
from pmcprg.copulas._robust   import (DPD_ALPHAS, DPDAlphaSelection, DPDFit, dpd_fit,
                                   dpd_objective, integral_c_power, select_alpha)

# --- CopulaEnum-driven auto-import ---------------------------------------
# Every member with a MODULE set is imported and exposed in this namespace
# under its CLASS_NAME (e.g. "CopulaGaussian").
for _m in CopulaEnum:
    if _m.MODULE:
        globals()[_m.CLASS_NAME] = getattr(
            importlib.import_module(_m.MODULE), _m.CLASS_NAME
        )

# --- __all__: infrastructure + every registered family -------------------
__all__ = [
    # Infrastructure
    'CopulaEnum',
    'CopulaVirt',
    'FitResult',
    'GoFResult',
    # Bivariate law
    'BivariateLaw',
    'ConditionalLaw',
    'BivariateFitResult',
    'BivariateBootstrapCI',
    # Standard errors and the independence / sub-model LR tests (audit FR-4)
    'IndependenceLRTest',
    'StandardErrors',
    'SubmodelLRTest',
    'independence_lr_test',
    'standard_errors',
    'submodel_lr_test',
    # Robust options (audit FR-7): MLE-vs-tau diagnostic (b), density power
    # divergence (c)
    'MleTauDiscrepancyTest',
    'mle_tau_discrepancy_test',
    'DPD_ALPHAS',
    'DPDAlphaSelection',
    'DPDFit',
    'dpd_fit',
    'dpd_objective',
    'integral_c_power',
    'select_alpha',
    # Copula families (source: CopulaEnum)
    *(m.CLASS_NAME for m in CopulaEnum if m.MODULE),
]
