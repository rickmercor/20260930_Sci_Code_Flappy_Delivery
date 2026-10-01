"""
Run the full dual-horizon audit over the three declared bar configurations. For each configuration build its point cloud, families, stretch state and damage at the declared base critical stretch multiplied by sc_scale, then report a row of: the internal virial (the sum over points of the internal force density times the reference coordinate times the point volume), the critical time step, the mean local damage over the points, and the total number of intact bonds. Assemble by calling the earlier sub-problem functions. Use E = 1.0, A = 1.0 and rho = 1.0.

The audit contrasts the internal state and the stable time-step bound across three non-uniformly discretised bars, including one with a stronger horizon factor.

Returns
-------
return (3, 4) float64: audit row per configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pd_fracture_audit(sc_scale):
    """sc_scale: positive multiplier on the declared base critical stretch.
    Builds the three declared configurations and returns a float64 array (3, 4)
    whose rows are [internal virial, critical time step, mean local damage,
    number of intact bonds]. Assembled by calling the earlier sub-problem
    functions. Raises ValueError on a non-positive or nonfinite sc_scale."""
    return np.zeros((3, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): dual-horizon fracture audit."""

import numpy as np

_E = 1.0
_A = 1.0
_RHO = 1.0
_SC_BASE = 0.16125
_PAR = {1: (4, 4, 2.515, 3, 0.10), 2: (3, 5, 2.515, 5, 0.08), 3: (5, 3, 3.015, 7, 0.10)}


def _build(variant):
    nC, nF, m, a, dC = _PAR[variant]
    dF = dC / 2.0
    xs = [0.0]
    for _ in range(nC - 1):
        xs.append(xs[-1] + dC)
    for _ in range(nF):
        xs.append(xs[-1] + dF)
    X = np.array(xs, dtype=np.float64)
    N = X.size
    sp = np.empty(N)
    sp[:nC] = dC
    sp[nC:] = dF
    sp[nC - 1] = 0.5 * (dC + dF)
    V = sp * _A
    delta = m * sp
    pert = np.array([(((i + 1) * (i + 2) + a * (i + 3)) % 17) - 8 for i in range(N)], dtype=np.float64)
    u = 0.010 * X + 0.0015 * pert
    return X, V, delta, u


def _oracle_pd_fracture_audit(sc_scale):
    if isinstance(sc_scale, bool) or not np.isfinite(sc_scale) or sc_scale <= 0:
        raise ValueError("sc_scale must be positive and finite")
    rows = []
    for variant in (1, 2, 3):
        X, V, delta, u = _build(variant)
        H = _oracle_families(X, delta)
        s = _oracle_bond_stretch(X, u)
        mu = _oracle_damage_state(s, H, _SC_BASE * sc_scale)
        F = _oracle_internal_force(X, u, V, delta, H, mu)
        D = _oracle_point_damage(mu, H, V)
        hc = _oracle_critical_time_step(X, V, delta, H, _RHO)
        rows.append([float(np.sum(F * X * V)), hc, float(np.mean(D)), float(np.sum(mu))])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'sc_scale = 1.0', "call": 'pd_fracture_audit(sc_scale)', "gold_call": '_oracle_pd_fracture_audit(sc_scale)', "tol": 1e-09},
        {"setup": 'sc_scale = 0.85', "call": 'pd_fracture_audit(sc_scale)', "gold_call": '_oracle_pd_fracture_audit(sc_scale)', "tol": 1e-09},
        {"setup": 'sc_scale = 1.3', "call": 'pd_fracture_audit(sc_scale)', "gold_call": '_oracle_pd_fracture_audit(sc_scale)', "tol": 1e-09},
    ]
