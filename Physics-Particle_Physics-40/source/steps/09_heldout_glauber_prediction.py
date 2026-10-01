"""
Return the gain-adjusted held-out finite percentage coefficient.

Infer the calibration state, then use the same unrounded fitted state and gain without refitting. Compose the seven preceding physical operations for the held-out rectangle, sum the diagonal and coherence columns over all attachments, and multiply by the fitted gain. Return the scalar to absolute accuracy $5\times10^{-6}$.

Returns
-------
Finite real gain-adjusted held-out percentage coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def heldout_glauber_prediction(coefficients: 'ArrayLike',
                               jet_eta: 'ArrayLike',
                               jet_phi: 'ArrayLike',
                               calibration_gaps: 'ArrayLike',
                               calibration_windows: 'ArrayLike',
                               observations: 'ArrayLike',
                               uncertainties: 'ArrayLike',
                               prediction_gap: 'ArrayLike',
                               prediction_window: 'ArrayLike',
                               z: float,
                               scale_ratio: float,
                               x_interval: 'ArrayLike',
                               azimuth_interval: 'ArrayLike',
                               gain_interval: 'ArrayLike',
                               initial_guess: 'ArrayLike') -> float:
    """Predict the gain-adjusted coefficient for one held-out veto.

    Parameters
    ----------
    coefficients : array-like
        Finite real or complex hard coefficients convertible with ``np.asarray``
        to shape ``(2, 2, 3)``.
    jet_eta, jet_phi : array-like
        Real finite hard directions convertible with ``np.asarray`` to shape
        ``(3,)``.
    calibration_gaps, calibration_windows : array-like
        Real calibration geometries convertible with ``np.asarray`` to shape
        ``(m, 2)`` for ``4 <= m <= 12``.
    observations, uncertainties : array-like
        Real values convertible with ``np.asarray`` to shape ``(m,)`` with
        strictly positive uncertainties.
    prediction_gap, prediction_window : array-like
        Real held-out veto endpoint values convertible with ``np.asarray`` to
        shape ``(2,)`` in step-06 domains.
    z, scale_ratio : float
        Scalars in the step-05 domains.
    x_interval, azimuth_interval, gain_interval : array-like
        Real fit-bound values convertible with ``np.asarray`` to shape ``(2,)``
        in step-08 domains.
    initial_guess : array-like
        Real strictly interior fit point convertible with ``np.asarray`` to
        shape ``(3,)``.

    Returns
    -------
    prediction : float
        Finite gain-adjusted held-out percentage coefficient without rounding,
        accurate to absolute tolerance ``5e-6``.

    Raises
    ------
    ValueError
        If any input violates a preceding-step type, shape, finiteness, range,
        positivity, geometry, or identifiability contract, or if the final
        prediction is nonfinite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
def _oracle_heldout_glauber_prediction(coefficients: 'ArrayLike',
                                       jet_eta: 'ArrayLike',
                                       jet_phi: 'ArrayLike',
                                       calibration_gaps: 'ArrayLike',
                                       calibration_windows: 'ArrayLike',
                                       observations: 'ArrayLike',
                                       uncertainties: 'ArrayLike',
                                       prediction_gap: 'ArrayLike',
                                       prediction_window: 'ArrayLike',
                                       z: float,
                                       scale_ratio: float,
                                       x_interval: 'ArrayLike',
                                       azimuth_interval: 'ArrayLike',
                                       gain_interval: 'ArrayLike',
                                       initial_guess: 'ArrayLike') -> float:
    try:
        inferred = _oracle_infer_beam_state(
            coefficients, jet_eta, jet_phi, calibration_gaps,
            calibration_windows, observations, uncertainties, z, scale_ratio,
            x_interval, azimuth_interval, gain_interval, initial_guess)
        beam = _oracle_gluon_beam_density(inferred[0], inferred[1])
        hard = _oracle_hard_spin_color_density(coefficients, beam)
        response = _oracle_color_spin_response(hard)
        angular = _oracle_finite_veto_tensors(
            jet_eta, jet_phi, prediction_gap, prediction_window)
        prefactors = _oracle_finite_glauber_prefactors(z, scale_ratio)
        channels = _oracle_finite_attachment_channels(response, angular, prefactors)
    except ValueError as exc:
        raise ValueError("invalid held-out prediction input") from exc
    answer = float(inferred[2] * np.sum(channels[:, :2]))
    if not np.isfinite(answer):
        raise ValueError("nonfinite held-out prediction")
    return answer

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

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_2 = """import numpy as np
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

    setup_3 = """import numpy as np
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

    return [
        {
            "setup": setup_1,
            "call": '_checked_numeric(heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.5606526523527, 0.1208558340278], [0.3469686708724, 3.70322056162], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]), ())',
            "gold_call": '_checked_numeric(_oracle_heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.5606526523527, 0.1208558340278], [0.3469686708724, 3.70322056162], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]), ())',
            "tol": 5e-06,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.4, 0.2], [-1.2, 1.4], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.5, 0.95]), ())',
            "gold_call": '_checked_numeric(_oracle_heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.4, 0.2], [-1.2, 1.4], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.5, 0.95]), ())',
            "tol": 5e-06,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.7, 0.7], [-3.141592653589793, 3.141592653589793], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]), ())',
            "gold_call": '_checked_numeric(_oracle_heldout_glauber_prediction(_independent(C), [1.55, -1.05, 1.0], [0.15, 2.3, -1.45], _independent(G), _independent(W), _independent(O), _independent(S), [-0.7, 0.7], [-3.141592653589793, 3.141592653589793], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.34, 0.83, 1.0]), ())',
            "tol": 5e-06,
        },
        {
            "setup": setup_2,
            "call": 'reject(heldout_glauber_prediction, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), [0.2, -0.2], [-1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_heldout_glauber_prediction, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), _independent(S), [0.2, -0.2], [-1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
        },
        {
            "setup": setup_3,
            "call": 'reject(heldout_glauber_prediction, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), [1, 1, 1, 0], [-0.2, 0.2], [-1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
            "gold_call": 'reject(_oracle_heldout_glauber_prediction, _independent(C), [1.5, -1.2, 1.1], [0.1, 2.2, -1.4], _independent(G), _independent(W), _independent(O), [1, 1, 1, 0], [-0.2, 0.2], [-1, 1], 0.32, 1.0, [0.2, 0.45], [0.15, 1.25], [0.85, 1.15], [0.3, 0.7, 1.0])',
        },
    ]
