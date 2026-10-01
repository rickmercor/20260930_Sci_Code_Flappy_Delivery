"""
Form the complex positive-wave-number density, velocity, and pressure states.

A density-modulated isothermal Maxwellian begins with p=n and zero bulk velocity.

Returns
-------
np.ndarray: complex mode array of shape (m, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def initialize_isothermal_modes(
    generators: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    """Return complex initial rows ``[n,u,p]`` aligned to the generators.

    Parameters
    ----------
    generators : np.ndarray
        Complex array with shape ``(m, 3, 3)``.
    amplitudes, phases : np.ndarray
        Real vectors of length ``m`` for cosine amplitudes and phases.

    Returns
    -------
    np.ndarray
        Complex initial-mode array with shape ``(m, 3)``.

    Raises
    ------
    ValueError
        If the arrays are invalid, misaligned, nonfinite, or contain a negative amplitude.
    """
    return np.empty((np.asarray(generators).shape[0], 3), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_initialize_isothermal_modes(
    generators: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray
) -> np.ndarray:
    """Reference implementation."""
    matrices = np.asarray(generators, dtype=complex)
    amp = np.asarray(amplitudes, dtype=float)
    phase = np.asarray(phases, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or matrices.shape[0] == 0:
        raise ValueError("generators must have shape (m, 3, 3)")
    if amp.ndim != 1 or phase.ndim != 1 or amp.shape != phase.shape:
        raise ValueError("amplitudes and phases must be aligned vectors")
    if amp.size != matrices.shape[0]:
        raise ValueError("mode controls must align with generators")
    if not np.all(np.isfinite(matrices)) or not np.all(np.isfinite(amp)):
        raise ValueError("generators and amplitudes must be finite")
    if not np.all(np.isfinite(phase)) or np.any(amp < 0.0):
        raise ValueError("phases must be finite and amplitudes nonnegative")
    density = 0.5 * amp * np.exp(1j * phase)
    modes = np.zeros((amp.size, 3), dtype=complex)
    modes[:, 0] = density
    modes[:, 2] = density
    return modes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {
            "setup": "generators = np.ones((3, 3, 3), dtype=complex); amplitudes = np.array([0.02, 0.01, 0.03]); phases = np.array([0.2, -0.4, 0.7])",
            "call": "initialize_isothermal_modes(generators, amplitudes, phases)",
            "gold_call": "_oracle_initialize_isothermal_modes(generators, amplitudes, phases)",
        },
        {
            "setup": "generators = np.zeros((1, 3, 3)); amplitudes = np.array([0.0]); phases = np.array([1.2])",
            "call": "float(np.linalg.norm(initialize_isothermal_modes(generators, amplitudes, phases)))",
            "gold_call": "float(np.linalg.norm(_oracle_initialize_isothermal_modes(generators, amplitudes, phases)))",
        },
        {
            "setup": "generators = np.zeros((2, 3, 3)); amplitudes = np.ones(1); phases = np.zeros(1);\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "status(lambda: initialize_isothermal_modes(generators, amplitudes, phases))",
            "gold_call": "status(lambda: _oracle_initialize_isothermal_modes(generators, amplitudes, phases))",
        },
    ]
