"""
Construct deterministic starting estimates for each normalized wave number.

Bohm–Gross dispersion and weak Landau damping select the physical Langmuir-root branch.

Returns
-------
np.ndarray: float rows [kappa, Re(zeta seed), Im(zeta seed)].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_kinetic_root_seeds(kappa: np.ndarray) -> np.ndarray:
    """Return deterministic Bohm--Gross seed rows.

    For every ``k = kappa[j]``, use
    ``omega_r = sqrt(1 + 3*k**2)``,
    ``gamma = -sqrt(pi/8)*k**(-3)*exp(-1/(2*k**2) - 3/2)``, and
    ``zeta = (omega_r + 1j*gamma)/(sqrt(2)*k)``. Return
    ``[k, zeta.real, zeta.imag]`` in each row.

    Parameters
    ----------
    kappa : np.ndarray
        Strictly increasing positive normalized wave numbers.

    Returns
    -------
    np.ndarray
        Float array with shape ``(len(kappa), 3)``.

    Raises
    ------
    ValueError
        If ``kappa`` is not a finite, positive, strictly increasing nonempty vector.
    """
    return np.empty((np.asarray(kappa).size, 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_kinetic_root_seeds(kappa: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    values = np.asarray(kappa, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("kappa must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("kappa values must be finite and positive")
    if values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise ValueError("kappa values must be strictly increasing")

    omega_real = np.sqrt(1.0 + 3.0 * values**2)
    gamma = -np.sqrt(np.pi / 8.0) * np.exp(
        -0.5 / values**2 - 1.5
    ) / values**3
    zeta_real = omega_real / (np.sqrt(2.0) * values)
    zeta_imag = gamma / (np.sqrt(2.0) * values)
    seeds = np.column_stack((values, zeta_real, zeta_imag))
    if not np.all(np.isfinite(seeds)):
        raise ValueError("the asymptotic seed is not finite")
    return seeds.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    return [
        {
            "setup": "kappa = np.array([0.18, 0.40, 0.82])",
            "call": "float(np.sum(build_kinetic_root_seeds(kappa) * np.arange(1, 10).reshape(3, 3)))",
            "gold_call": "float(np.sum(_oracle_build_kinetic_root_seeds(kappa) * np.arange(1, 10).reshape(3, 3)))",
        },
        {
            "setup": "kappa = np.array([0.31])",
            "call": "float(np.sum(build_kinetic_root_seeds(kappa)))",
            "gold_call": "float(np.sum(_oracle_build_kinetic_root_seeds(kappa)))",
        },
        {
            "setup": "kappa = np.array([0.4, 0.2]);\ndef status(fn):\n    try: fn(); return 0\n    except ValueError: return 1\n    except Exception: return 2",
            "call": "status(lambda: build_kinetic_root_seeds(kappa))",
            "gold_call": "status(lambda: _oracle_build_kinetic_root_seeds(kappa))",
        },
    ]
