"""
Fit the common preparation state and gain.

For $4$ to $12$ calibration rectangles, compose the preceding seven operations. Minimize the RMS standardized residual $(gZ_i-Z_i^{\rm obs})/\sigma_i$ in the supplied $x$, $\beta$, and positive-gain box. Every $\sigma_i$ is positive. Use multiple starts. Return $[\widehat x,\widehat\beta,\widehat g,r_w]$ accurate to absolute tolerances $10^{-4},10^{-3},10^{-4},10^{-4}$ respectively. Reject a nonfinite, rank-deficient, or nonunique optimum.

Returns
-------
Real (4,) array [x_hat,beta_hat,gain_hat,weighted_rms].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_beam_state(coefficients: 'ArrayLike',
                     jet_eta: 'ArrayLike',
                     jet_phi: 'ArrayLike',
                     calibration_gaps: 'ArrayLike',
                     calibration_windows: 'ArrayLike',
                     observations: 'ArrayLike',
                     uncertainties: 'ArrayLike',
                     z: float,
                     scale_ratio: float,
                     x_interval: 'ArrayLike',
                     azimuth_interval: 'ArrayLike',
                     gain_interval: 'ArrayLike',
                     initial_guess: 'ArrayLike') -> 'np.ndarray':
    """Fit the common preparation state and gain.

    Parameters
    ----------
    coefficients : array-like
        Finite real or complex hard coefficients convertible with ``np.asarray``
        to shape ``(2, 2, 3)``.
    jet_eta, jet_phi : array-like
        Real finite hard directions convertible with ``np.asarray`` to shape
        ``(3,)``.
    calibration_gaps, calibration_windows : array-like
        Real values convertible with ``np.asarray`` to shape ``(m, 2)`` for
        ``4 <= m <= 12`` in the step-06 veto domains.
    observations, uncertainties : array-like
        Real values convertible with ``np.asarray`` to shape ``(m,)``. Every
        uncertainty must be positive.
    z, scale_ratio : float
        Scalars in the step-05 domains.
    x_interval, azimuth_interval, gain_interval : array-like
        Strictly increasing closed bounds convertible with ``np.asarray`` to
        shape ``(2,)``. The x interval lies in ``[0.1, 0.49]``, the azimuth
        interval lies in ``[-pi, pi]`` with width at most ``pi``, and the gain
        interval is positive.
    initial_guess : array-like
        Real finite interior point convertible with ``np.asarray`` to shape
        ``(3,)`` and ordered as ``x, beta, gain``.

    Returns
    -------
    fit : numpy.ndarray
        Float64 array ``[x_hat, beta_hat, gain_hat, weighted_rms]`` of shape
        ``(4,)`` without rounding, accurate componentwise to absolute
        tolerances ``[1e-4, 1e-3, 1e-4, 1e-4]``.

    Raises
    ------
    ValueError
        If an input has an invalid type or shape, is complex where real is
        required, contains booleans, strings, objects, or nonfinite values, if
        geometries or intervals are invalid, uncertainties are nonpositive,
        the initial point is not strictly interior, or the optimum is nonfinite,
        rank deficient, or nonunique.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
def _fit_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (tuple, list)):
        return any(_fit_has_bool(item) for item in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == "b"
def _fit_array(value, shape=None, complex_ok=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid numeric input") from exc
    if _fit_has_bool(value) or array.dtype.kind not in ("iufc" if complex_ok else "iuf"):
        raise ValueError("invalid numeric type")
    if shape is not None and array.shape != shape:
        raise ValueError("invalid shape")
    array = array.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite input")
    return array
def _fit_residual(state, coefficients, angular, prefactors, observations,
                  uncertainties):
    beam = _oracle_gluon_beam_density(state[0], state[1])
    hard = _oracle_hard_spin_color_density(coefficients, beam)
    response = _oracle_color_spin_response(hard)
    predicted = np.array([
        np.sum(_oracle_finite_attachment_channels(response, one, prefactors)[:, :2])
        for one in angular])
    return (state[2] * predicted - observations) / uncertainties
def _oracle_infer_beam_state(coefficients: 'ArrayLike',
                             jet_eta: 'ArrayLike',
                             jet_phi: 'ArrayLike',
                             calibration_gaps: 'ArrayLike',
                             calibration_windows: 'ArrayLike',
                             observations: 'ArrayLike',
                             uncertainties: 'ArrayLike',
                             z: float,
                             scale_ratio: float,
                             x_interval: 'ArrayLike',
                             azimuth_interval: 'ArrayLike',
                             gain_interval: 'ArrayLike',
                             initial_guess: 'ArrayLike') -> 'np.ndarray':
    from scipy.optimize import least_squares
    coefficients = _fit_array(coefficients, (2, 2, 3), True)
    jet_eta = _fit_array(jet_eta, (3,))
    jet_phi = _fit_array(jet_phi, (3,))
    gaps = _fit_array(calibration_gaps)
    windows = _fit_array(calibration_windows)
    if gaps.ndim != 2 or gaps.shape[1:] != (2,) or not 4 <= gaps.shape[0] <= 12:
        raise ValueError("invalid calibration-gap shape")
    if windows.shape != gaps.shape:
        raise ValueError("calibration windows must match gaps")
    count = gaps.shape[0]
    observations = _fit_array(observations, (count,))
    uncertainties = _fit_array(uncertainties, (count,))
    if np.any(uncertainties <= 0):
        raise ValueError("uncertainties must be positive")
    x_interval = _fit_array(x_interval, (2,))
    azimuth_interval = _fit_array(azimuth_interval, (2,))
    gain_interval = _fit_array(gain_interval, (2,))
    initial_guess = _fit_array(initial_guess, (3,))
    if not (0.1 <= x_interval[0] < x_interval[1] <= 0.49):
        raise ValueError("invalid x interval")
    if not (-np.pi <= azimuth_interval[0] < azimuth_interval[1] <= np.pi):
        raise ValueError("invalid azimuth interval")
    if azimuth_interval[1] - azimuth_interval[0] > np.pi:
        raise ValueError("azimuth interval too wide")
    if not (0 < gain_interval[0] < gain_interval[1]):
        raise ValueError("invalid gain interval")
    lower = np.array([x_interval[0], azimuth_interval[0], gain_interval[0]])
    upper = np.array([x_interval[1], azimuth_interval[1], gain_interval[1]])
    if np.any(initial_guess <= lower) or np.any(initial_guess >= upper):
        raise ValueError("initial guess must be strictly interior")
    try:
        angular = [_oracle_finite_veto_tensors(jet_eta, jet_phi, gaps[row], windows[row])
                   for row in range(count)]
        prefactors = _oracle_finite_glauber_prefactors(z, scale_ratio)
    except ValueError as exc:
        raise ValueError("invalid calibration geometry") from exc

    fractions = np.array([[.5, .5, .5], [.25, .25, .25], [.75, .75, .75],
                          [.25, .75, .5], [.75, .25, .5]])
    starts = [initial_guess] + [lower + f * (upper - lower) for f in fractions]
    args = (coefficients, angular, prefactors, observations, uncertainties)
    fits = [least_squares(_fit_residual, start, args=args,
                          bounds=(lower, upper), xtol=1e-13,
                          ftol=1e-13, gtol=1e-13, max_nfev=1000)
            for start in starts]
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    state = np.asarray(best.x, dtype=float)
    weighted_rms = float(np.sqrt(np.mean(np.asarray(best.fun, dtype=float) ** 2)))
    singular = np.linalg.svd(np.asarray(best.jac, dtype=float), compute_uv=False)
    if singular.shape != (3,) or singular[-1] <= 1e-7 or not np.isfinite(weighted_rms):
        raise ValueError("calibration is not locally identifiable")
    near = [fit for fit in fits
            if np.sqrt(np.mean(np.asarray(fit.fun, dtype=float) ** 2)) <= weighted_rms + 1e-7]
    if any(np.linalg.norm(np.asarray(fit.x) - state, ord=np.inf) > 1e-5 for fit in near):
        raise ValueError("calibration is not unique in the box")
    return np.array([state[0], state[1], state[2], weighted_rms], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup_1 = """import numpy as np
C = np.array([[[1, 0.3j, 0.4], [0.2 + 0.1j, -0.5j, 0.1]], [[0.3, 0.4 + 0.2j, -0.2j], [-0.4j, 0.1, 0.6 + 0.2j]]], complex)
G = np.array([[-0.388452693071, -0.066945177637], [-0.634648645579, -0.066950804284], [-0.249444915622, 0.181745881676], [0.043867700795, 0.293867700795], [-0.632542909981, 0.215646222147], [-0.615492018129, 0.23602598849]])
W = np.array([[1.780818504124, 5.281016662726], [1.861422166882, 4.504401164242], [-2.215524504611, 0.730191864899], [2.632019017269, 4.8523490159], [1.466492004617, 3.841060778475], [1.07456417082, 3.994489652396]])
O = np.array([0.018043847301862, 0.018154855393881, -0.023966704615551, -0.012794744342497, 0.078921145280024, 0.085269744997212])
S = np.full(6, 0.0005)

def _checked_numeric(value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'iufc':
        raise ValueError('unexpected numerical result type or shape')
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite result')
    return np.stack((result.real, result.imag), axis=0)

def _checked_fit(value):
    return _checked_numeric(value, (4,)) / np.array([1.0, 10.0, 1.0, 1.0])

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_2 = """import numpy as np
C = np.array([[[1, 0.3j, 0.4], [0.2 + 0.1j, -0.5j, 0.1]], [[0.3, 0.4 + 0.2j, -0.2j], [-0.4j, 0.1, 0.6 + 0.2j]]], complex)
G = np.array([[-0.388452693071, -0.066945177637], [-0.634648645579, -0.066950804284], [-0.249444915622, 0.181745881676], [0.043867700795, 0.293867700795], [-0.632542909981, 0.215646222147], [-0.615492018129, 0.23602598849]])
W = np.array([[1.780818504124, 5.281016662726], [1.861422166882, 4.504401164242], [-2.215524504611, 0.730191864899], [2.632019017269, 4.8523490159], [1.466492004617, 3.841060778475], [1.07456417082, 3.994489652396]])
O = np.array([0.0195450424833246, 0.0206941802665082, -0.0261335364036163, -0.013285862236216, 0.085796841958744, 0.0922919938394491])
S = np.full(6, 0.0007)

def _checked_numeric(value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'iufc':
        raise ValueError('unexpected numerical result type or shape')
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite result')
    return np.stack((result.real, result.imag), axis=0)

def _checked_fit(value):
    return _checked_numeric(value, (4,)) / np.array([1.0, 10.0, 1.0, 1.0])

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_3 = """import numpy as np
C = np.array([[[1, 0.3j, 0.4], [0.2 + 0.1j, -0.5j, 0.1]], [[0.3, 0.4 + 0.2j, -0.2j], [-0.4j, 0.1, 0.6 + 0.2j]]], complex)
G = np.array([[-0.388452693071, -0.066945177637], [-0.634648645579, -0.066950804284], [-0.249444915622, 0.181745881676], [0.043867700795, 0.293867700795]])
W = np.array([[1.780818504124, 5.281016662726], [1.861422166882, 4.504401164242], [-2.215524504611, 0.730191864899], [2.632019017269, 4.8523490159]])
O = np.array([0.018011693134947403, 0.017847217885189705, -0.024753511399387442, -0.013271167042784346])
S = np.full(4, 0.0005)

def _checked_numeric(value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'iufc':
        raise ValueError('unexpected numerical result type or shape')
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite result')
    return np.stack((result.real, result.imag), axis=0)

def _checked_fit(value):
    return _checked_numeric(value, (4,)) / np.array([1.0, 10.0, 1.0, 1.0])

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_4 = """import numpy as np
def reject(fn,*a):
    try:
        fn(*a)
    except ValueError:
        return 1.0
    except Exception:
        return 0.0
    return 0.0

C=np.ones((2,2,3),complex)
G=np.zeros((4,2))
W=np.ones((4,2))
O=np.zeros(4)

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_5 = """import numpy as np
def reject(fn,*a):
    try:
        fn(*a)
    except ValueError:
        return 1.0
    except Exception:
        return 0.0
    return 0.0

C=np.ones((2,2,3),complex)
G=np.zeros((4,2))
W=np.ones((4,2))
O=np.zeros(4)
S=np.ones(4)

def _independent(value):
    return np.array(value, copy=True)
"""

    return [
        {
            "setup": setup_1,
            "call": '_checked_fit(infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]))',
            "gold_call": '_checked_fit(_oracle_infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]))',
            "tol": 0.0001,
        },
        {
            "setup": setup_2,
            "call": '_checked_fit(infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.28, 0.75, 0.98]))',
            "gold_call": '_checked_fit(_oracle_infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.28, 0.75, 0.98]))',
            "tol": 0.0001,
        },
        {
            "setup": setup_3,
            "call": '_checked_fit(infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0]))',
            "gold_call": '_checked_fit(_oracle_infer_beam_state(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0]))',
            "tol": 0.0001,
        },
        {
            "setup": setup_4,
            "call": 'reject(infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), [1, 0, 1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), [1, 0, 1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
        },
        {
            "setup": setup_5,
            "call": 'reject(infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G)[:3], _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G)[:3], _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
        },
        {
            "setup": setup_5,
            "call": 'reject(infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [1.1, 0.9], [0.3, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [1.1, 0.9], [0.3, 0.7, 1.0])',
        },
        {
            "setup": setup_5,
            "call": 'reject(infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.2, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_infer_beam_state, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.2, 0.7, 1.0])',
        },
    ]
