"""Tests for the copula layer — instantiation, PDF, majorant, exceptions."""
import pytest

from prg.copulas import (
    CopulaEnum,
    CopulaGaussian, CopulaStudent,
    CopulaGH, CopulaClayton, CopulaA12, CopulaA14,
    CopulaProduct, CopulaFGM, CopulaCubSec,
)
from prg.exceptions import CopulaParameterError, CopulaNotAvailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AVAILABLE = [
    CopulaGaussian(tau_k=0.5),
    CopulaStudent(tau_k=0.5),
    CopulaGH(tau_k=0.2),
    CopulaClayton(tau_k=0.2),
    CopulaA12(tau_k=0.5),
    CopulaA14(tau_k=0.5),
    CopulaProduct(tau_k=0.0),
    CopulaFGM(tau_k=0.1),
    CopulaCubSec(tau_k=0.1),
]

UV = [0.5, 0.7]


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_pdf_positive(cop):
    assert cop.pdf(UV) > 0.0


@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_majorant_geq_pdf(cop):
    u_left = UV[0]
    assert cop.majorant(u_left) >= cop.pdf(UV)


@pytest.mark.parametrize('cop', ALL_AVAILABLE)
def test_update_tau_k(cop):
    old_tau = cop.params['tau_k']
    new_tau = (cop.tau_min + cop.tau_max) / 2.0
    cop.update_tau_k(new_tau)
    assert cop.params['tau_k'] == pytest.approx(new_tau)
    cop.update_tau_k(old_tau)


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------

def test_bad_tau_k_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian(tau_k=5.0)


def test_bad_param_name_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian(rho=0.5)


def test_missing_tau_k_raises_parameter_error():
    with pytest.raises(CopulaParameterError):
        CopulaGaussian()


def test_parameter_error_is_value_error():
    with pytest.raises(ValueError):
        CopulaGaussian(tau_k=5.0)


def test_student_cdf_raises_not_implemented():
    cop = CopulaStudent(tau_k=0.5)
    with pytest.raises(NotImplementedError):
        cop.cdf(UV)


# ---------------------------------------------------------------------------
# CopulaEnum tests
# ---------------------------------------------------------------------------

def test_available_count():
    # 10 originals (FRANK now enabled) + JOE + 3 Survival + BB1 + AMH + Plackett = 17
    assert len(CopulaEnum.available()) == 17


def test_cubsec_long_name_english():
    assert CopulaEnum.CUBSEC.LONG_NAME == 'Cubic Section'


def test_from_short_name():
    assert CopulaEnum.from_short_name('Gauss') is CopulaEnum.GAUSSIAN


def test_favorite():
    assert CopulaEnum.favorite() is CopulaEnum.GAUSSIAN
