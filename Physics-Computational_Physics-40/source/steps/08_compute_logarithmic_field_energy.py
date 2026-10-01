"""
Reduce the initial and final density spectra to one normalized diagnostic.

Poisson scaling makes each modal electric energy proportional to |n_k/k| squared.

Returns
-------
float: base-10 logarithm of the normalized electric-field energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_log_field_energy(
    kappa: np.ndarray, initial_modes: np.ndarray, final_modes: np.ndarray
) -> float:
    """Return ``log10(E_final / E_initial)`` as a native float.

    Parameters
    ----------
    kappa : np.ndarray
        Positive wave-number vector of length ``m``.
    initial_modes, final_modes : np.ndarray
        Complex state arrays with shape ``(m, 3)``.

    Returns
    -------
    float
        Base-10 logarithmic normalized field energy.

    Raises
    ------
    ValueError
        If the arrays are invalid or either field energy is nonpositive or nonfinite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_log_field_energy(
    kappa: np.ndarray, initial_modes: np.ndarray, final_modes: np.ndarray
) -> float:
    """Reference implementation."""
    wave_numbers = np.asarray(kappa, dtype=float)
    initial = np.asarray(initial_modes, dtype=complex)
    final = np.asarray(final_modes, dtype=complex)
    if wave_numbers.ndim != 1 or wave_numbers.size == 0:
        raise ValueError("kappa must be a nonempty vector")
    expected = (wave_numbers.size, 3)
    if initial.shape != expected or final.shape != expected:
        raise ValueError("mode arrays must have shape (len(kappa), 3)")
    if not np.all(np.isfinite(wave_numbers)) or np.any(wave_numbers <= 0.0):
        raise ValueError("wave numbers must be finite and positive")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
        raise ValueError("mode arrays must be finite")
    initial_energy = float(np.sum(np.abs(initial[:, 0] / wave_numbers) ** 2))
    final_energy = float(np.sum(np.abs(final[:, 0] / wave_numbers) ** 2))
    if initial_energy <= 0.0 or final_energy <= 0.0:
        raise ValueError("both field energies must be strictly positive")
    result = float(np.log10(final_energy / initial_energy))
    if not np.isfinite(result):
        raise ValueError("logarithmic field-energy ratio must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Check Poisson weighting, the density column, and boundary behavior."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "k = np.array([0.2, 0.5, 0.9])\n"
                "y0 = np.array([[1+1j, 0.2-0.1j, 0.4+0.2j], "
                "[2-1j, -0.3+0.4j, 0.7-0.3j], "
                "[0.5+0.8j, 0.6+0.1j, 1.1+0j]], dtype=complex)\n"
                "yf = y0 * np.array([[0.3, 0.7, 0.2], "
                "[0.8, 0.1, 0.45], [0.05, 0.6, 0.9]])"
            ),
            "call": "compute_log_field_energy(k, y0, yf)",
            "gold_call": "_oracle_compute_log_field_energy(k, y0, yf)",
        },
        {
            "setup": "import numpy as np; k = np.array([0.4]); y0 = np.array([[1+0j, 0j, 1+0j]]); yf = y0.copy()",
            "call": "compute_log_field_energy(k, y0, yf)",
            "gold_call": "_oracle_compute_log_field_energy(k, y0, yf)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "k = np.array([0.4])\n"
                "y0 = np.zeros((1, 3))\n"
                "yf = np.ones((1, 3))\n"
                "def status(fn):\n"
                "    try:\n"
                "        fn()\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "status(lambda: compute_log_field_energy(k, y0, yf))",
            "gold_call": "status(lambda: _oracle_compute_log_field_energy(k, y0, yf))",
        },
    ]
