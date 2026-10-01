"""
A GLV candidate is retained as a stable state only when it is an admissible steady state, no absent species has a positive invasion rate, and the full ecological Jacobian has spectral abscissa below -tol. The strict spectral margin excludes neutral equilibria, while the non-invasibility test implements saturation before stability. Invalid support solutions remain ordinary zero classifications rather than disappearing from the support-indexed arrays.

Returns
-------
Integer array of shape (Q,), equal to one for stable states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classify_stable_states(
    support_masks: np.ndarray,
    equilibria: np.ndarray,
    invasion_rates: np.ndarray,
    spectral_abscissae: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    '''Classify support-restricted candidates as asymptotically stable states.

    Parameters
    ----------
    support_masks : np.ndarray
        Binary support array of shape (Q, N).
    equilibria : np.ndarray
        Candidate equilibria of shape (Q, N), with all-NaN invalid rows.
    invasion_rates : np.ndarray
        Per-capita growth vectors of shape (Q, N), aligned with equilibria.
    spectral_abscissae : np.ndarray
        Maximum real Jacobian eigenvalues of shape (Q,), with NaN for invalid
        candidates.
    tol : float
        Finite positive tolerance for abundance, residual, and stability tests.

    Raises
    ------
    ValueError
        If shapes are inconsistent, masks are non-binary, finite and NaN rows
        are misaligned, or tol is not finite and positive.

    Returns
    -------
    stable_flags : np.ndarray
        Integer array of shape (Q,), equal to one exactly for stable states.
    '''
    return stable_flags  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_classify_stable_states(
    support_masks: np.ndarray,
    equilibria: np.ndarray,
    invasion_rates: np.ndarray,
    spectral_abscissae: np.ndarray,
    tol: float = 1e-9,
) -> np.ndarray:
    """Reference implementation."""
    support_masks = np.asarray(support_masks)
    equilibria = np.asarray(equilibria, dtype=float)
    invasion_rates = np.asarray(invasion_rates, dtype=float)
    spectral_abscissae = np.asarray(spectral_abscissae, dtype=float)
    if support_masks.ndim != 2:
        raise ValueError("support_masks must be two dimensional")
    if not np.all((support_masks == 0) | (support_masks == 1)):
        raise ValueError("support_masks must be binary")
    if equilibria.shape != support_masks.shape or invasion_rates.shape != support_masks.shape:
        raise ValueError("equilibria and invasion_rates must match support_masks")
    if spectral_abscissae.shape != (support_masks.shape[0],):
        raise ValueError("spectral_abscissae must have shape (Q,)")
    if not isinstance(tol, (int, float, np.integer, np.floating)) or not np.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    equilibrium_finite = np.all(np.isfinite(equilibria), axis=1)
    equilibrium_nan = np.all(np.isnan(equilibria), axis=1)
    rate_finite = np.all(np.isfinite(invasion_rates), axis=1)
    rate_nan = np.all(np.isnan(invasion_rates), axis=1)
    if not np.all(equilibrium_finite | equilibrium_nan):
        raise ValueError("each equilibrium row must be finite or all NaN")
    if not np.all(rate_finite | rate_nan):
        raise ValueError("each invasion-rate row must be finite or all NaN")
    if not np.array_equal(equilibrium_finite, rate_finite):
        raise ValueError("finite equilibrium and invasion-rate rows must align")
    if not np.array_equal(equilibrium_finite, np.isfinite(spectral_abscissae)):
        raise ValueError("finite equilibria and spectral abscissae must align")

    stable_flags = np.zeros(support_masks.shape[0], dtype=int)
    for row in np.flatnonzero(equilibrium_finite):
        resident = support_masks[row].astype(bool)
        absent = ~resident
        equilibrium = equilibria[row]
        rates = invasion_rates[row]
        support_consistent = np.all(equilibrium[resident] > tol) and np.all(
            np.abs(equilibrium[absent]) <= tol
        )
        steady = np.all(np.abs(rates[resident]) <= tol)
        saturated = np.all(rates[absent] <= tol)
        stable = spectral_abscissae[row] < -tol
        stable_flags[row] = int(support_consistent and steady and saturated and stable)
    return stable_flags

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''support_masks = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
equilibria = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]])
invasion_rates = np.array([[1.0, 1.0], [0.0, -0.3], [-0.4, 0.0], [np.nan, np.nan]])
spectral_abscissae = np.array([1.0, -0.3, -0.4, np.nan])
tol = 1e-9
''',
            "call": "classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae, tol).tolist()",
            "gold_call": "_oracle_classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae, tol).tolist()",
        },
        {
            "setup": '''support_masks = np.array([[1]])
equilibria = np.array([[1.0]])
invasion_rates = np.array([[0.0]])
spectral_abscissae = np.array([-1.0])
tol = 1e-9
''',
            "call": "classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae, tol).tolist()",
            "gold_call": "_oracle_classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae, tol).tolist()",
        },
        {
            "setup": '''support_masks = np.array([[1, 0]])
equilibria = np.array([[1.0, 0.0]])
invasion_rates = np.array([[0.0, -1.0]])
spectral_abscissae = np.array([-1.0, -0.5])
def run_model():
    try:
        classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_classify_stable_states(support_masks, equilibria, invasion_rates, spectral_abscissae)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
''',
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
