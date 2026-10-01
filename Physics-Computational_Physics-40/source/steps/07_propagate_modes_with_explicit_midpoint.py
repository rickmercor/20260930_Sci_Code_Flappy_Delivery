"""
Advance every independent Fourier state to the requested final time.

Midpoint Runge–Kutta approximates each root-matched linear evolution to second order.

Returns
-------
np.ndarray: complex final-mode array of shape (m, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Real
import numpy as np
def propagate_midpoint_modes(
    generators: np.ndarray,
    initial_modes: np.ndarray,
    delta_t: float,
    final_time: float,
) -> np.ndarray:
    """Return the final complex mode rows after exact-count midpoint steps.

    Parameters
    ----------
    generators : np.ndarray
        Complex array with shape ``(m, 3, 3)``.
    initial_modes : np.ndarray
        Complex array with shape ``(m, 3)``.
    delta_t, final_time : float
        Positive step and nonnegative exactly reachable final time.

    Returns
    -------
    np.ndarray
        Complex final-mode array with shape ``(m, 3)``.

    Raises
    ------
    ValueError
        If mode data or time controls are invalid, including a nonintegral step count.
    """
    return np.empty_like(np.asarray(initial_modes, dtype=complex))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Real
import numpy as np
def _oracle_propagate_midpoint_modes(
    generators: np.ndarray,
    initial_modes: np.ndarray,
    delta_t: float,
    final_time: float,
) -> np.ndarray:
    """Reference implementation."""
    matrices = np.asarray(generators, dtype=complex)
    state = np.asarray(initial_modes, dtype=complex).copy()
    if isinstance(delta_t, bool) or not isinstance(delta_t, Real):
        raise ValueError("delta_t must be a real scalar")
    if isinstance(final_time, bool) or not isinstance(final_time, Real):
        raise ValueError("final_time must be a real scalar")
    step = float(delta_t)
    stop = float(final_time)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("generators must have shape (m, 3, 3)")
    if state.shape != (matrices.shape[0], 3) or matrices.shape[0] == 0:
        raise ValueError("initial_modes must have shape (m, 3)")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(state)):
        raise ValueError("mode data must be finite")
    if not np.isfinite(step) or step <= 0.0 or not np.isfinite(stop) or stop < 0.0:
        raise ValueError("time controls are outside their finite domains")
    count = round(stop / step)
    if not np.isclose(count * step, stop, rtol=0.0, atol=1e-12):
        raise ValueError("final_time must be an integer multiple of delta_t")
    for _ in range(count):
        slope_one = np.einsum("mij,mj->mi", matrices, state)
        midpoint = state + 0.5 * step * slope_one
        slope_two = np.einsum("mij,mj->mi", matrices, midpoint)
        state += step * slope_two
    if not np.all(np.isfinite(state)):
        raise ValueError("midpoint propagation produced non-finite modes")
    return state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {"setup": "g = np.array([[[0, -1j, 0], [-1j, 0, -0.2j], [0.1, -2j, -0.1]]]); y = np.array([[0.01+0.02j, 0j, 0.01+0.02j]])", "call": "float(np.sum(propagate_midpoint_modes(g, y, 0.01, 0.1).real))", "gold_call": "float(np.sum(_oracle_propagate_midpoint_modes(g, y, 0.01, 0.1).real))"},
        {"setup": "g = np.zeros((1, 3, 3), complex); y = np.array([[1+2j, 3+4j, 5+6j]])", "call": "float(np.linalg.norm(propagate_midpoint_modes(g, y, 0.1, 0.0)))", "gold_call": "float(np.linalg.norm(_oracle_propagate_midpoint_modes(g, y, 0.1, 0.0)))"},
        {"setup": "g = np.zeros((1, 3, 3)); y = np.zeros((1, 3));\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2", "call": "status(lambda: propagate_midpoint_modes(g, y, 0.03, 0.1))", "gold_call": "status(lambda: _oracle_propagate_midpoint_modes(g, y, 0.03, 0.1))"},
    ]
