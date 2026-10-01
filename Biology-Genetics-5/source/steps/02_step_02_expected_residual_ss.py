"""
Evaluate expected Gaussian residual energy under independent single-effect factors.

Independent variational factors each select exactly one variant. Evaluate their expected Gaussian residual energy from sufficient statistics, retaining uncertainty within each factor as well as interactions between factors.

Returns
-------
Unrounded Python float, the expected residual sum of squares.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expected_residual_ss(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, posterior: 'np.ndarray | list') -> float:
    """Return the unrounded expected residual sum of squares as a float.

    Parameters and contract
    -----------------------
    gram is a finite positive-definite (p,p) Gram matrix, cross is a finite
    (p,) vector, and yty is y.T @ y. These are feasible sufficient statistics.
    posterior is a finite array of shape (L,3,p), L,p >= 1. Within each factor,
    row 0 contains nonnegative selection probabilities summing to one, row 1
    contains conditional means, and row 2 contains nonnegative conditional
    variances. Factors are independent; each selects exactly one variant.
    Include posterior uncertainty. Zero probabilities and zero variances are
    valid here. Do not change the supplied Gram scaling or mutate inputs.
    Inputs satisfy these domains; invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, the expected residual sum of squares."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_expected_residual_ss(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, posterior: 'np.ndarray | list') -> float:
    gram, cross, posterior = map(np.asarray, (gram, cross, posterior))
    alpha, mean, variance = (posterior[:, 0], posterior[:, 1], posterior[:, 2])
    component_mean = alpha * mean
    total = component_mean.sum(axis=0)
    second_moment = alpha * (mean ** 2 + variance)
    return float(yty - 2 * total @ cross + total @ gram @ total + np.sum(second_moment * np.diag(gram)) - np.einsum('li,ij,lj->', component_mean, gram, component_mean))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,P,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(expected_residual_ss(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_expected_residual_ss(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,np.array([[[0.5, 0.5], [0.0, 0.0], [0.2, 0.2]]]),)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(expected_residual_ss(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_expected_residual_ss(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[3.,1.],[1.,4.]])\nt=np.array([.5,-.4])\nP=np.array([[[.25,.75],[1.,-2.],[.2,.3]],[[.6,.4],[-.5,.8],[.1,.4]]])\n_base_args = (G,t,10.0,np.array([[[1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]]),)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(expected_residual_ss(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_expected_residual_ss(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = (np.array([[10.0]]),np.array([3.0]),4.0,np.array([[[1.0], [0.2], [0.1]], [[1.0], [-0.1], [0.2]]]),)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(expected_residual_ss(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_expected_residual_ss(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
