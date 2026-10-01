"""
Evaluate the complete Gaussian evidence lower bound of a supplied variational posterior.

The evidence lower bound compares a factorized signed-prior model with its prior and Gaussian likelihood. Its absolute value must include the n-sample Gaussian normalizing constant and normalized categorical priors. See SuSiNE Appendix 5.

Returns
-------
Unrounded Python float, complete Gaussian ELBO in nats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variational_objective(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, noise: float, prior_variance: float, prior_mean: 'np.ndarray | list', posterior: 'np.ndarray | list') -> float:
    """Return the unrounded Gaussian variational objective in nats as a float.

    Parameters and contract
    -----------------------
    gram (p,p), cross (p,), yty and positive integer n describe feasible
    sufficient statistics; use gram directly without rescaling. noise and
    prior_variance are positive variance scalars; prior_mean has shape (p,).
    posterior has shape (L,3,p): selection probabilities, conditional means,
    and strictly positive conditional variances. Probabilities are
    nonnegative and sum to one within each independent single-effect factor.
    Each prior selects uniformly among p variants. Include the complete
    n-sample Gaussian likelihood and categorical/normal KL terms, with
    zero-probability entropy terms interpreted by continuity. Do not mutate
    inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, complete Gaussian ELBO in nats."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_variational_objective(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, noise: float, prior_variance: float, prior_mean: 'np.ndarray | list', posterior: 'np.ndarray | list') -> float:
    posterior, prior_mean = map(np.asarray, (posterior, prior_mean))
    alpha, mean, variance = (posterior[:, 0], posterior[:, 1], posterior[:, 2])
    normal_kl = 0.5 * ((variance + (mean - prior_mean) ** 2) / prior_variance - 1 + np.log(prior_variance / variance))
    kl = np.sum(xlogy(alpha, alpha * posterior.shape[-1]) + alpha * normal_kl)
    erss = _oracle_expected_residual_ss(gram, cross, yty, posterior)
    return float(-0.5 * n * np.log(2 * np.pi * noise) - 0.5 * erss / noise - kl)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\n\n_base_args = (np.array([[10.0]]),np.array([3.0]),4.0,12,1.0,0.2,np.array([0.4]),np.array([[[1.0], [1 / 3], [1 / 15]]]),)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(variational_objective(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_variational_objective(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,8,1.4,0.7,np.array([0.3, -0.2]),P,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(variational_objective(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_variational_objective(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,9,0.6,0.3,np.array([-0.2, 0.5]),P,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(variational_objective(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_variational_objective(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,8,1.0,0.2,np.zeros(2),np.array([[[1.0, 0.0], [0.1, -0.2], [0.05, 0.1]]]),)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(variational_objective(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_variational_objective(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
