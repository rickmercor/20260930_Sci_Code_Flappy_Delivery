"""
Aggregate fitted inclusion vectors using the primary cluster-weight ensemble.

SuSiNE main Section 4.3 defines its primary cluster-weight ensemble. Compute the posterior inclusion summary and the resulting weight of each input fit using that rule, at unit temperature.

Returns
-------
Tuple of float ndarrays ((p,), (M,)): ensemble PIP and per-fit weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aggregate_ensemble(pips: 'np.ndarray | list', elbos: 'np.ndarray | list', labels: 'np.ndarray | list') -> 'tuple[np.ndarray, np.ndarray]':
    """Return (aggregate_pip, fit_weights), both unrounded float ndarrays.

    Parameters and contract
    -----------------------
    pips is a finite probability matrix (M,p), M,p >= 1. elbos is a finite
    length-M float array, and labels is a length-M integer array identifying
    clusters. Labels may be nonconsecutive or negative and have no numerical
    meaning. Apply SuSiNE's primary cluster-weight rule at unit temperature
    at both levels, not its naive ensemble. aggregate_pip has shape (p,);
    fit_weights has shape (M,) in original fit order and sums to one.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Tuple of float ndarrays ((p,), (M,)): ensemble PIP and per-fit weights."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_aggregate_ensemble(pips: 'np.ndarray | list', elbos: 'np.ndarray | list', labels: 'np.ndarray | list') -> 'tuple[np.ndarray, np.ndarray]':
    pips, elbos, labels = map(np.asarray, (pips, elbos, labels))
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    peaks = np.array([elbos[group].max() for group in groups])
    masses = np.exp(peaks - logsumexp(peaks))
    weights = np.zeros(len(pips))
    for group, mass in zip(groups, masses):
        weights[group] = mass * np.exp(elbos[group] - logsumexp(elbos[group]))
    return (weights @ pips, weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.2, 0.5]],[-10000.0],[8],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 2\n    m, p = np.asarray(_base_args[0]).shape\n    _float_array(result[0], (p,))\n    _float_array(result[1], (m,))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(aggregate_ensemble(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_aggregate_ensemble(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.1], [0.1], [0.9]],[-10.0, -10.0, -10.0],[0, 0, 1],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 2\n    m, p = np.asarray(_base_args[0]).shape\n    _float_array(result[0], (p,))\n    _float_array(result[1], (m,))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(aggregate_ensemble(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_aggregate_ensemble(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.1], [0.9]],[-10000.0, -10000.0],[0, 1],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 2\n    m, p = np.asarray(_base_args[0]).shape\n    _float_array(result[0], (p,))\n    _float_array(result[1], (m,))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(aggregate_ensemble(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_aggregate_ensemble(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.1, 0.8], [0.3, 0.7], [0.9, 0.2]],[-7.0, -9.0, -8.0],[9, 9, -2],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 2\n    m, p = np.asarray(_base_args[0]).shape\n    _float_array(result[0], (p,))\n    _float_array(result[1], (m,))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(aggregate_ensemble(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_aggregate_ensemble(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
