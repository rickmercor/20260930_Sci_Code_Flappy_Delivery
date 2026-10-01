"""
Compute one signed-prior single-effect posterior from residual sufficient statistics.

A signed functional annotation changes the conditional Gaussian prior mean of a single causal effect. Compute its posterior using residual sufficient statistics and a uniform prior over candidate variants. The signed-prior model is defined in SuSiNE, main Section 2.2 and Appendix 6.

Returns
-------
Float ndarray (3,p): selection probabilities, conditional means, conditional variances.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_effect_posterior(residual_cross: 'np.ndarray | list', diagonal: 'np.ndarray | list', noise: float, prior_variance: float, prior_mean: 'np.ndarray | list') -> 'np.ndarray':
    """Return a float ndarray of shape (3,p).

    Parameters and contract
    -----------------------
    residual_cross, diagonal and prior_mean are finite float arrays of shape
    (p,), p >= 1. residual_cross is X.T @ residual; diagonal contains positive
    diagonal entries of X.T @ X. noise is a positive residual variance, and
    prior_variance is a positive common conditional effect variance.
    A single factor selects one variant with uniform prior probability 1/p.
    Row 0 is the posterior variant selection probability; rows 1 and 2 are
    the effect mean and variance conditional on selecting each variant.
    Preserve variant order and signed prior means. Return unrounded values.
    Inputs satisfy these domains; invalid-input behavior is not tested.

    Returns
    -------
    Float ndarray (3,p): selection probabilities, conditional means, conditional variances."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_single_effect_posterior(residual_cross: 'np.ndarray | list', diagonal: 'np.ndarray | list', noise: float, prior_variance: float, prior_mean: 'np.ndarray | list') -> 'np.ndarray':
    residual_cross, diagonal, prior_mean = map(np.asarray, (residual_cross, diagonal, prior_mean))
    sampling = noise / diagonal
    variance = sampling * prior_variance / (sampling + prior_variance)
    mean = variance * (residual_cross / noise + prior_mean / prior_variance)
    estimate = residual_cross / diagonal
    logbf = -0.5 * np.log1p(prior_variance / sampling)
    logbf += 0.5 * estimate ** 2 / sampling - 0.5 * (estimate - prior_mean) ** 2 / (sampling + prior_variance)
    inclusion = np.exp(logbf - logsumexp(logbf))
    return np.stack((inclusion, mean, variance))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\n\n_base_args = ([3.0],[10.0],1.0,0.2,[0.4],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _float_array(result, (3, len(_base_args[0])))\n    return result\n", 'call': '_check_result(single_effect_posterior(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_single_effect_posterior(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([0.0, 0.0],[10.0, 10.0],1.0,0.2,[0.0, 0.0],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _float_array(result, (3, len(_base_args[0])))\n    return result\n", 'call': '_check_result(single_effect_posterior(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_single_effect_posterior(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([4.0, 4.0],[10.0, 10.0],1.0,0.2,[0.3, -0.3],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _float_array(result, (3, len(_base_args[0])))\n    return result\n", 'call': '_check_result(single_effect_posterior(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_single_effect_posterior(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([8.0, -2.0, 1.0],[20.0, 12.0, 5.0],1.7,0.13,[0.4, -0.2, 0.1],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _float_array(result, (3, len(_base_args[0])))\n    return result\n", 'call': '_check_result(single_effect_posterior(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_single_effect_posterior(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
