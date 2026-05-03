"""
Tests for BivariateLaw and ConditionalLaw.

Checks: construction, joint PDF/CDF/log-PDF, conditional PDF/CDF,
        sampling shapes, conditional_law factory, and input validation.
"""
import numpy as np
import pytest
import scipy.stats as st

from prg.copulas import (
    BivariateLaw, ConditionalLaw,
    CopulaGaussian, CopulaProduct, CopulaFGM, CopulaClayton,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEFT  = (st.norm,  0.0, 1.0)
RIGHT = (st.expon, 0.0, 1.0)


@pytest.fixture
def law_gaussian():
    law = BivariateLaw(CopulaGaussian(tau_k=0.5), LEFT, RIGHT)
    law.set_seed(0)
    return law


@pytest.fixture
def law_product():
    law = BivariateLaw(CopulaProduct(tau_k=0.0), LEFT, RIGHT)
    law.set_seed(0)
    return law


# ---------------------------------------------------------------------------
# 1. Construction and validation
# ---------------------------------------------------------------------------

def test_construction_ok(law_gaussian):
    assert law_gaussian.ndim == 2
    assert law_gaussian.copula is not None


def test_bad_left_margin_raises():
    with pytest.raises(ValueError, match='left_margin'):
        BivariateLaw(CopulaGaussian(tau_k=0.5), 'not_a_tuple', RIGHT)


def test_bad_right_margin_not_dist_raises():
    with pytest.raises(ValueError, match='right_margin'):
        BivariateLaw(CopulaGaussian(tau_k=0.5), LEFT, (42, 1.0))


def test_quantile_range_stored(law_gaussian):
    assert law_gaussian._quantile_range == (0.05, 0.95)


def test_custom_quantile_range():
    law = BivariateLaw(CopulaProduct(tau_k=0.0), LEFT, RIGHT, quantile_range=(0.1, 0.9))
    assert law._quantile_range == (0.1, 0.9)


# ---------------------------------------------------------------------------
# 2. Joint PDF / log-PDF / CDF
# ---------------------------------------------------------------------------

def test_pdf_positive(law_gaussian):
    assert law_gaussian.pdf([0.0, 1.0]) > 0.0


def test_log_pdf_consistent(law_gaussian):
    xy = [0.3, 0.5]
    assert abs(np.log(law_gaussian.pdf(xy)) - law_gaussian.log_pdf(xy)) < 1e-10


def test_cdf_in_unit_interval(law_gaussian):
    for (x, y) in [(-1.0, 0.5), (0.0, 1.0), (1.0, 2.0)]:
        c = law_gaussian.cdf([x, y])
        assert 0.0 <= c <= 1.0


def test_cdf_product_copula_factorises(law_product):
    """For the product copula: F(x,y) = F1(x) · F2(y)."""
    x, y = 0.5, 1.0
    f1 = st.norm.cdf(x,  0.0, 1.0)
    f2 = st.expon.cdf(y, 0.0, 1.0)
    assert abs(law_product.cdf([x, y]) - f1 * f2) < 1e-10


def test_pdf_normalises(law_product):
    """∫∫ f(x,y) dx dy ≈ 1 for the product copula (analytical check)."""
    # Grid covers 99.8% of each margin; residual strip contributes ~0.4% per dim.
    xs = np.linspace(st.norm.ppf(0.001),        st.norm.ppf(0.999),        50)
    ys = np.linspace(st.expon.ppf(0.001, 0., 1.), st.expon.ppf(0.999, 0., 1.), 50)
    Z  = np.array([[law_product.pdf([x, y]) for y in ys] for x in xs])
    integral = float(np.trapezoid(np.trapezoid(Z, ys, axis=1), xs))
    assert abs(integral - 1.0) < 0.03


# ---------------------------------------------------------------------------
# 3. Conditional PDF / CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_pdf_positive(law_gaussian, which):
    # y_obs is always a left-margin value (norm); y_cond must be in the support
    # of the conditioned-on margin: right=expon→[0,∞), left=norm→(-∞,∞).
    y_obs  = 0.5                          # in support of both margins
    y_conds = [0.1, 0.5, 1.5] if which == 'left' else [-1.0, 0.5, 1.5]
    for y_cond in y_conds:
        assert law_gaussian.conditional_pdf(y_cond, y_obs, which=which) > 0.0


@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_log_pdf_consistent(law_gaussian, which):
    y_obs, y_cond = 0.3, 0.8
    lp  = law_gaussian.conditional_log_pdf(y_cond, y_obs, which=which)
    p   = law_gaussian.conditional_pdf(y_cond,     y_obs, which=which)
    assert abs(np.log(p) - lp) < 1e-10


@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_cdf_monotone(law_gaussian, which):
    """Conditional CDF must be non-decreasing."""
    y_obs  = 0.0
    margin = st.expon if which == 'left' else st.norm
    ys     = np.linspace(margin.ppf(0.05, 0., 1.), margin.ppf(0.95, 0., 1.), 30)
    vals   = [law_gaussian.conditional_cdf(y, y_obs, which=which) for y in ys]
    assert np.all(np.diff(vals) >= -1e-9)


@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_cdf_bounds(law_gaussian, which):
    """Conditional CDF in [0, 1]."""
    y_obs  = 0.0
    margin = st.expon if which == 'left' else st.norm
    for y in np.linspace(margin.ppf(0.05, 0., 1.), margin.ppf(0.95, 0., 1.), 10):
        c = law_gaussian.conditional_cdf(y, y_obs, which=which)
        assert 0.0 - 1e-9 <= c <= 1.0 + 1e-9


def test_conditional_pdf_product_equals_marginal(law_product):
    """For the product copula, conditional PDF = marginal PDF (independence)."""
    y_obs = 0.5
    for y in [0.3, 1.0, 2.0]:
        cond_p   = law_product.conditional_pdf(y, y_obs, which='left')
        marginal = st.expon.pdf(y, 0.0, 1.0)
        assert abs(cond_p - marginal) < 1e-8


# ---------------------------------------------------------------------------
# 4. conditional_law factory  (ConditionalLaw)
# ---------------------------------------------------------------------------

def test_conditional_law_returns_correct_type(law_gaussian):
    cl = law_gaussian.conditional_law(0.0, which='left')
    assert isinstance(cl, ConditionalLaw)


@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_law_pdf_agrees(law_gaussian, which):
    y_obs, y_cond = 0.5, 1.0
    cl = law_gaussian.conditional_law(y_obs, which=which)
    assert abs(cl.pdf(y_cond) - law_gaussian.conditional_pdf(y_cond, y_obs, which=which)) < 1e-12


@pytest.mark.parametrize('which', ['left', 'right'])
def test_conditional_law_cdf_agrees(law_gaussian, which):
    y_obs, y_cond = 0.5, 1.0
    cl = law_gaussian.conditional_law(y_obs, which=which)
    assert abs(cl.cdf(y_cond) - law_gaussian.conditional_cdf(y_cond, y_obs, which=which)) < 1e-12


def test_conditional_law_bad_which_raises(law_gaussian):
    with pytest.raises(ValueError, match='which'):
        law_gaussian.conditional_law(0.0, which='both')


# ---------------------------------------------------------------------------
# 5. Sampling
# ---------------------------------------------------------------------------

def test_sample_shape(law_gaussian):
    s = law_gaussian.sample(10)
    assert s.shape == (10, 2)


def test_sample_1_shape(law_gaussian):
    s = law_gaussian.sample(1)
    assert s.shape == (1, 2)


def test_sample_conditional_shape(law_gaussian):
    s = law_gaussian.sample_conditional(0.0, which='left', n=7)
    assert s.shape == (7,)


@pytest.mark.parametrize('which,y_obs', [('left', 0.5), ('right', 1.0)])
def test_sample_conditional_via_law(law_gaussian, which, y_obs):
    # y_obs must be well inside the support of the conditioning margin
    # (expon with y_right=0 → u≈0 → Gaussian copula majorant diverges)
    cl = law_gaussian.conditional_law(y_obs, which=which)
    s  = cl.sample(5)
    assert s.shape == (5,)


def test_sample_reproducible(law_gaussian):
    law_gaussian.set_seed(123)
    s1 = law_gaussian.sample(5)
    law_gaussian.set_seed(123)
    s2 = law_gaussian.sample(5)
    np.testing.assert_array_equal(s1, s2)


def test_new_seed_changes_output(law_gaussian):
    law_gaussian.set_seed(1)
    s1 = law_gaussian.sample(5)
    law_gaussian.new_seed()
    s2 = law_gaussian.sample(5)
    assert not np.allclose(s1, s2)


def test_sample_conditional_bad_which_raises(law_gaussian):
    with pytest.raises(ValueError, match='which'):
        law_gaussian.sample_conditional(0.0, which='up')


# ---------------------------------------------------------------------------
# 6. RNG
# ---------------------------------------------------------------------------

def test_seed_property(law_gaussian):
    law_gaussian.set_seed(999)
    assert law_gaussian.seed == 999


def test_new_seed_returns_int(law_gaussian):
    s = law_gaussian.new_seed()
    assert isinstance(s, int)
