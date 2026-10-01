"""
Run the full finite-bending chain-scission audit over the three declared configurations. Each configuration is a chain of N bonds at base force f0 multiplied by force_scale, with (N, f0) = (5, 0.20), (4, 0.15) and (3, 0.25). Report a row of: the converged rupture threshold of the first bond, the converged threshold of the bond at index N//2, the activation barrier of the first bond in kT, the largest bond activation barrier in kT, and the sum of the bond activation barriers in kT. Use De = 1.0, a = 2.15, le = 1.0, beta = 279.0, equilibrium bond angle 69 degrees, and a bending stiffness such that beta*kphi*pi**2 = 1820. Assemble by calling the earlier sub-problem functions. Raise ValueError on a non-positive or nonfinite force_scale.

The audit spans three chain lengths and forces so that the end bonds, the interior bonds and the overall barrier budget are all exercised.

Returns
-------
return (3, 5) float64: audit row per configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scission_audit(force_scale):
    """force_scale: positive multiplier on each configuration's base force.
    Returns (3, 5) float64 whose rows are [first-bond threshold, threshold at
    index N//2, first-bond barrier in kT, largest bond barrier in kT, sum of
    bond barriers in kT]. Assembled by calling the earlier sub-problem
    functions. Raises ValueError on a non-positive or nonfinite force_scale."""
    return np.zeros((3, 5))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): finite-bending chain-scission audit."""

import numpy as np

_DE, _AA, _LE, _BETA = 1.0, 2.15, 1.0, 279.0
_PHI_E = 69.0 * np.pi / 180.0
_KPHI = 1820.0 / (np.pi ** 2 * 279.0)
_BASE = {1: (5, 0.20), 2: (4, 0.15), 3: (3, 0.25)}


def _oracle_scission_audit(force_scale):
    if isinstance(force_scale, bool) or not np.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("force_scale must be positive and finite")
    rows = []
    for v in (1, 2, 3):
        N, f0 = _BASE[v]
        f = f0 * force_scale
        thr = _oracle_self_consistent_thresholds(N, f, _BETA, _DE, _AA, _LE, _KPHI, _PHI_E)
        bb = _oracle_bond_barriers(N, f, _BETA, _DE, _AA, _LE, _KPHI, _PHI_E)
        rows.append([float(thr[0]), float(thr[N // 2]), float(bb[0]),
                     float(bb.max()), float(bb.sum())])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'force_scale = 1.0', "call": 'scission_audit(force_scale)', "gold_call": '_oracle_scission_audit(force_scale)', "tol": 1e-07},
        {"setup": 'force_scale = 0.95', "call": 'scission_audit(force_scale)', "gold_call": '_oracle_scission_audit(force_scale)', "tol": 1e-07},
        {"setup": 'force_scale = 1.1', "call": 'scission_audit(force_scale)', "gold_call": '_oracle_scission_audit(force_scale)', "tol": 1e-07},
        {"setup": 'force_scale = 0.4', "call": 'scission_audit(force_scale)', "gold_call": '_oracle_scission_audit(force_scale)', "tol": 1e-07},
        {"setup": 'force_scale = 2.6', "call": 'scission_audit(force_scale)', "gold_call": '_oracle_scission_audit(force_scale)', "tol": 1e-07},
    ]
