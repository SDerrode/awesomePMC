"""prg.copulas — public API.

Toutes les familles de copules sont importées automatiquement depuis
CopulaEnum : ajouter une nouvelle famille ne nécessite que d'enregistrer
son entrée dans CopulaEnum (champ MODULE renseigné) et d'écrire le fichier
du module ; cet __init__.py reste intact.
"""

import importlib

from prg.copulas._base     import CopulaEnum, CopulaVirt, FitResult, GoFResult
from prg.copulas.bivariate import BivariateLaw, ConditionalLaw, BivariateFitResult, BivariateBootstrapCI

# --- Auto-import piloté par CopulaEnum -----------------------------------
# Chaque membre dont MODULE est renseigné est importé et exposé dans ce
# namespace sous son CLASS_NAME (ex. "CopulaGaussian").
for _m in CopulaEnum:
    if _m.MODULE:
        globals()[_m.CLASS_NAME] = getattr(
            importlib.import_module(_m.MODULE), _m.CLASS_NAME
        )

# --- __all__ : infrastructure + toutes les familles enregistrées ---------
__all__ = [
    # Infrastructure
    'CopulaEnum',
    'CopulaVirt',
    'FitResult',
    'GoFResult',
    # Loi bivariée
    'BivariateLaw',
    'ConditionalLaw',
    'BivariateFitResult',
    'BivariateBootstrapCI',
    # Familles de copules (source : CopulaEnum)
    *(m.CLASS_NAME for m in CopulaEnum if m.MODULE),
]
