"""
Fit one fixed signed-prior setting using a prescribed finite cyclic update schedule.

Fit one fixed signed-prior hyperparameter setting of the SuSiNE single-effect model. The fixed initialization and finite cyclic update schedule are part of this experiment, not the authors' automatic search or convergence defaults.

Returns
-------
Tuple of (float ndarray (p,), Python float, float ndarray (effects,3,p)): PIP, ELBO, posterior.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_single_setting(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', scale: float, prior_variance: float, effects: int=2, noise: float=1.0, sweeps: int=300) -> 'tuple[np.ndarray, float, np.ndarray]':
    """Return (pip, elbo, posterior), without rounding.

    Parameters and contract
    -----------------------
    gram (p,p), cross (p,), yty and positive integer n are feasible finite
    sufficient statistics with positive-definite gram. annotation is (p,).
    scale is a finite scalar; prior_variance and noise are positive variance
    scalars. effects and sweeps are positive integers. The conditional prior
    mean is scale*annotation and variant selection priors are uniform.
    Initialize every component's expected coefficient vector to zero. Perform
    exactly sweeps full cyclic IBSS sweeps, in ascending component order,
    using each new update immediately. Do not rescale, update hyperparameters,
    prune, anneal or stop early. pip is a (p,) float ndarray; elbo is a float
    including the n-sample normalizer; posterior is (effects,3,p) with rows
    selection probability, conditional mean and conditional variance.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Tuple of (float ndarray (p,), Python float, float ndarray (effects,3,p)): PIP, ELBO, posterior."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, xlogy

def _oracle_fit_single_setting(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', scale: float, prior_variance: float, effects: int=2, noise: float=1.0, sweeps: int=300) -> 'tuple[np.ndarray, float, np.ndarray]':
    gram, cross, annotation = map(np.asarray, (gram, cross, annotation))
    p = len(cross)
    posterior = np.empty((effects, 3, p))
    component_mean = np.zeros((effects, p))
    prior_mean = scale * annotation
    for _ in range(sweeps):
        for ell in range(effects):
            residual_cross = cross - gram @ (component_mean.sum(axis=0) - component_mean[ell])
            posterior[ell] = _oracle_single_effect_posterior(residual_cross, np.diag(gram), noise, prior_variance, prior_mean)
            component_mean[ell] = posterior[ell, 0] * posterior[ell, 1]
    pip = 1 - np.prod(1 - posterior[:, 0], axis=0)
    elbo = _oracle_variational_objective(gram, cross, yty, n, noise, prior_variance, prior_mean, posterior)
    return (pip, elbo, posterior)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': "import numpy as np\nimport copy\nG=np.array([[12.,8.],[8.,15.]])\nt=np.array([5.,-2.])\na=np.array([-.4,.6])\n_base_args = (G,t,10.0,17,a,0.5,0.1,)\n_base_kwargs = {'effects':1,'noise':0.8,'sweeps':2}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 3\n    p = len(_base_args[1])\n    _float_array(result[0], (p,))\n    assert isinstance(result[1], (float, np.floating)) and np.isfinite(result[1]), 'Expected a finite ELBO float'\n    _float_array(result[2], (_base_kwargs.get('effects', 2), 3, p))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(fit_single_setting(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_fit_single_setting(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[12.,8.],[8.,15.]])\nt=np.array([5.,-2.])\na=np.array([-.4,.6])\n_base_args = (G,t,10.0,17,a,1.0,0.3,)\n_base_kwargs = {'effects':2,'noise':0.8,'sweeps':1}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 3\n    p = len(_base_args[1])\n    _float_array(result[0], (p,))\n    assert isinstance(result[1], (float, np.floating)) and np.isfinite(result[1]), 'Expected a finite ELBO float'\n    _float_array(result[2], (_base_kwargs.get('effects', 2), 3, p))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(fit_single_setting(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_fit_single_setting(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[12.,8.],[8.,15.]])\nt=np.array([5.,-2.])\na=np.array([-.4,.6])\n_base_args = (G,t,10.0,17,a,0.5,0.3,)\n_base_kwargs = {'effects':2,'noise':0.8,'sweeps':80}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 3\n    p = len(_base_args[1])\n    _float_array(result[0], (p,))\n    assert isinstance(result[1], (float, np.floating)) and np.isfinite(result[1]), 'Expected a finite ELBO float'\n    _float_array(result[2], (_base_kwargs.get('effects', 2), 3, p))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(fit_single_setting(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_fit_single_setting(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}, {'setup': "import numpy as np\nimport copy\nG=np.array([[12.,8.],[8.,15.]])\nt=np.array([5.,-2.])\na=np.array([-.4,.6])\n_base_args = (G,t,10.0,17,a,-0.4,0.2,)\n_base_kwargs = {'effects':3,'noise':1.2,'sweeps':15}\n_candidate_args, _candidate_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n_reference_args, _reference_kwargs = copy.deepcopy((_base_args, _base_kwargs))\n\ndef _same_input(left, right):\n    if isinstance(left, np.ndarray):\n        assert isinstance(right, np.ndarray) and left.dtype == right.dtype\n        np.testing.assert_array_equal(left, right)\n    elif isinstance(left, (tuple, list)):\n        assert type(left) is type(right) and len(left) == len(right)\n        for x, y in zip(left, right):\n            _same_input(x, y)\n    elif isinstance(left, dict):\n        assert left.keys() == right.keys()\n        for key in left:\n            _same_input(left[key], right[key])\n    else:\n        assert left == right\n\ndef _float_array(value, shape):\n    assert isinstance(value, np.ndarray), 'Expected an ndarray'\n    assert value.shape == shape, 'Incorrect output shape'\n    assert value.dtype.kind == 'f', 'Expected floating-point output'\n    assert np.isfinite(value).all(), 'Non-finite output'\n\ndef _check_result(result, args, kwargs):\n    _same_input((_base_args, _base_kwargs), (args, kwargs))\n    assert isinstance(result, tuple) and len(result) == 3\n    p = len(_base_args[1])\n    _float_array(result[0], (p,))\n    assert isinstance(result[1], (float, np.floating)) and np.isfinite(result[1]), 'Expected a finite ELBO float'\n    _float_array(result[2], (_base_kwargs.get('effects', 2), 3, p))\n    return np.concatenate([np.asarray(x).reshape(-1) for x in result])\n", 'call': '_check_result(fit_single_setting(*_candidate_args, **_candidate_kwargs), _candidate_args, _candidate_kwargs)', 'gold_call': '_check_result(_oracle_fit_single_setting(*_reference_args, **_reference_kwargs), _reference_args, _reference_kwargs)'}]
