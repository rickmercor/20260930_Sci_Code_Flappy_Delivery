"""
Partition fitted inclusion vectors by credible-shift complete-linkage clustering.

SuSiNE groups variational fits by their credible-shift dissimilarity before ensemble aggregation. Implement the main Section 4.3 dissimilarity with complete linkage and the deterministic labeling convention below.

Returns
-------
Integer ndarray (M,), canonical zero-based cluster labels in input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def partition_fits(pips: 'np.ndarray | list', cutoff: float=0.05) -> 'np.ndarray':
    """Return an integer ndarray of canonical cluster labels, shape (M,).

    Parameters and contract
    -----------------------
    pips is a finite (M,p) array in [0,1], M,p >= 1; cutoff is nonnegative.
    Use the paper's credible-shift dissimilarity with complete linkage.
    Merge the closest pair while its linkage distance is <= cutoff.
    In an exact distance tie, choose the lexicographically smallest ordered
    pair of sorted original-member-index tuples. Clusters in such a pair
    are themselves ordered lexicographically. Label final clusters 0,1,...
    by ascending smallest original fit index, preserving input fit order.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Integer ndarray (M,), canonical zero-based cluster labels in input order."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_partition_fits(pips: 'np.ndarray | list', cutoff: float=0.05) -> 'np.ndarray':
    pips = np.asarray(pips)
    distances = np.max(np.maximum(pips[:, None], pips[None, :]) * np.abs(pips[:, None] - pips[None, :]), axis=2)
    groups = [(i,) for i in range(len(pips))]
    while len(groups) > 1:
        distance, left, right = min(((float(distances[np.ix_(g, h)].max()), g, h) for i, g in enumerate(groups) for h in groups[i + 1:]))
        if distance > cutoff:
            break
        groups.remove(left)
        groups.remove(right)
        groups.append(tuple(sorted(left + right)))
        groups.sort()
    labels = np.empty(len(pips), dtype=int)
    for label, group in enumerate(groups):
        labels[list(group)] = label
    return labels

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.2, 0.5]],0.05,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, np.ndarray) and result.shape == (len(_base_args[0]),)\n    assert result.dtype.kind in 'iu', 'Expected integer labels'\n    return result\n", 'call': '_check_result(partition_fits(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_partition_fits(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.1], [0.3], [0.5]],0.11,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, np.ndarray) and result.shape == (len(_base_args[0]),)\n    assert result.dtype.kind in 'iu', 'Expected integer labels'\n    return result\n", 'call': '_check_result(partition_fits(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_partition_fits(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.2, 0.5], [0.2, 0.5]],0.0,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, np.ndarray) and result.shape == (len(_base_args[0]),)\n    assert result.dtype.kind in 'iu', 'Expected integer labels'\n    return result\n", 'call': '_check_result(partition_fits(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_partition_fits(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.0], [0.5]],0.25,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, np.ndarray) and result.shape == (len(_base_args[0]),)\n    assert result.dtype.kind in 'iu', 'Expected integer labels'\n    return result\n", 'call': '_check_result(partition_fits(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_partition_fits(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[0.0], [1.0], [0.0]],0.0,)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, np.ndarray) and result.shape == (len(_base_args[0]),)\n    assert result.dtype.kind in 'iu', 'Expected integer labels'\n    return result\n", 'call': '_check_result(partition_fits(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_partition_fits(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
