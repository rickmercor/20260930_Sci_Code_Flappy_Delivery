"""
Run both signed-annotation ensembles and compute the final inclusion-probability contrast.

Quantify sensitivity of the cluster-weighted fine-mapping posterior to changing the sign of one functional annotation. Repeat the same finite model ensemble in both conditions and return the signed inclusion-probability contrast. Use the preceding scientific components; this is the final whole-pipeline function.

Returns
-------
Unrounded Python float, flipped-minus-original ensemble PIP at the zero-based variant index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def annotation_sensitivity(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', variant: int, scales: 'np.ndarray | list', variances: 'np.ndarray | list', effects: int=2, noise: float=1.0, sweeps: int=300, cutoff: float=0.05) -> float:
    """Return the unrounded flipped-minus-original variant PIP as a float.

    Parameters and contract
    -----------------------
    gram, cross, yty, n, annotation, effects, noise and sweeps have the same
    valid domains as fit_single_setting. variant is a zero-based integer
    index in [0,p). scales and variances are nonempty finite one-dimensional
    arrays; variances are positive. cutoff is nonnegative.
    Fit all Cartesian pairs in both conditions, with scales as the outer
    loop and variances as the inner loop. Flip only annotation[variant] in
    the second condition. Retain every fit, use the stated zero-start finite
    cyclic schedule, partition with the preceding complete-linkage convention
    and apply the primary unit-temperature cluster-weight ensemble. The
    returned contrast refers to the same variant index in both conditions.
    Do not round intermediate or final results, or mutate any input.
    Invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, flipped-minus-original ensemble PIP at the zero-based variant index."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_annotation_sensitivity(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', variant: int, scales: 'np.ndarray | list', variances: 'np.ndarray | list', effects: int=2, noise: float=1.0, sweeps: int=300, cutoff: float=0.05) -> float:
    values = []
    for flipped in (False, True):
        current = np.array(annotation, dtype=float, copy=True)
        if flipped:
            current[variant] *= -1
        fits = [_oracle_fit_single_setting(gram, cross, yty, n, current, scale, variance, effects, noise, sweeps) for scale in scales for variance in variances]
        pips = np.array([fit[0] for fit in fits])
        elbos = np.array([fit[1] for fit in fits])
        labels = _oracle_partition_fits(pips, cutoff)
        aggregate, _ = _oracle_aggregate_ensemble(pips, elbos, labels)
        values.append(aggregate[variant])
    return float(values[1] - values[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\nG=np.array([[60.0, 52.992, 5.76, 0.9, 16.62, -3.84], [52.992, 60.0, 11.1, -5.4, 11.94, -8.28], [5.76, 11.1, 60.0, -50.22, 9.57, -0.9], [0.9, -5.4, -50.22, 60.0, -0.45, 7.2], [16.62, 11.94, 9.57, -0.45, 60.0, 39.0], [-3.84, -8.28, -0.9, 7.2, 39.0, 60.0]])\nt=np.array([26.0, 24.0, -22.0, 20.0, 13.0, 10.0])\na=np.array([0.45, -0.35, -0.4, 0.5, 0.15, -0.25])\n_base_args = (G,t,60.0,61,a,1,[0.0, 0.4, 0.8, 1.2],[0.04, 0.12, 0.3],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(annotation_sensitivity(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_annotation_sensitivity(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[60.0, 52.992, 5.76, 0.9, 16.62, -3.84], [52.992, 60.0, 11.1, -5.4, 11.94, -8.28], [5.76, 11.1, 60.0, -50.22, 9.57, -0.9], [0.9, -5.4, -50.22, 60.0, -0.45, 7.2], [16.62, 11.94, 9.57, -0.45, 60.0, 39.0], [-3.84, -8.28, -0.9, 7.2, 39.0, 60.0]])\nt=np.array([26.0, 24.0, -22.0, 20.0, 13.0, 10.0])\na=np.array([0.45, -0.35, -0.4, 0.5, 0.15, -0.25])\n_base_args = (G,t,60.0,61,a,1,[0.0],[0.04, 0.12, 0.3],)\n_base_kwargs = {'sweeps':30}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(annotation_sensitivity(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_annotation_sensitivity(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[60.0, 52.992, 5.76, 0.9, 16.62, -3.84], [52.992, 60.0, 11.1, -5.4, 11.94, -8.28], [5.76, 11.1, 60.0, -50.22, 9.57, -0.9], [0.9, -5.4, -50.22, 60.0, -0.45, 7.2], [16.62, 11.94, 9.57, -0.45, 60.0, 39.0], [-3.84, -8.28, -0.9, 7.2, 39.0, 60.0]])\nt=np.array([26.0, 24.0, -22.0, 20.0, 13.0, 10.0])\na=np.array([0.45, -0.35, -0.4, 0.5, 0.15, -0.25])\na[1]*=-1\n_base_args = (G,t,60.0,61,a,1,[0.0, 0.4, 0.8, 1.2],[0.04, 0.12, 0.3],)\n_base_kwargs = {}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(annotation_sensitivity(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_annotation_sensitivity(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\n\n_base_args = ([[10.0]],[3.0],4.0,12,[0.4],0,[0.0, 0.5],[0.1, 0.2],)\n_base_kwargs = {'effects':1,'sweeps':3}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, (float, np.floating)) and np.isfinite(result), 'Expected a finite float scalar'\n    return result\n", 'call': '_check_result(annotation_sensitivity(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_annotation_sensitivity(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
