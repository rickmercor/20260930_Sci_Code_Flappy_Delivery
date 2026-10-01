"""
The orchestrator. It must call the six earlier functions rather than reimplementing them, and assemble the force components, the local damage, the per-point micro-moduli, the force magnitudes and a diagnostics row.

The audit assembles the complete discrete state of a dual-horizon bond-based peridynamic body at one instant.

Returns
-------
numpy float64 array of shape (6, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_audit(X, x, deltas, V, E, Gc, mode, z, A):
    """Orchestrator. Returns a numpy float64 array of shape (6, n)."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_MODES = ("1d", "3d", "plane_stress", "plane_strain")

def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("non-positive or non-finite " + name)
    return v

def _mode(m):
    s = str(m).strip().lower()
    if s not in _MODES:
        raise ValueError("unknown mode " + s)
    return s

def _points(P, name):
    A = _f64(P)
    if A.ndim != 2 or A.shape[1] != 2 or A.shape[0] < 1 or not np.all(np.isfinite(A)):
        raise ValueError("bad point array " + name)
    return A

def _oracle_pd_audit(X, x, deltas, V, E, Gc, mode, z, A):
    """Assembles the dual-horizon bond-based peridynamic state. Returns (6, n)."""
    P = _points(X, "X")
    n = P.shape[0]
    d = _f64(deltas)
    if d.shape != (n,):
        raise ValueError("bad horizon array")
    cj = np.array([_oracle_pd_micromodulus(E, float(d[j]), mode, z, A)[0] for j in range(n)],
                  dtype=np.float64)
    sc = _oracle_pd_critical_stretch(Gc, E, float(np.min(d)), mode)[0]
    sets = _oracle_pd_dual_horizon_sets(X, deltas)
    state = _oracle_pd_bond_state(X, x, sc)
    F = _oracle_pd_pairwise_force(X, x, sets, state, cj, V)
    D = _oracle_pd_point_damage(sets, state, V)
    out = np.zeros((6, n), dtype=np.float64)
    out[0, :] = F[:, 0]
    out[1, :] = F[:, 1]
    out[2, :] = D
    out[3, :] = cj
    out[4, :] = np.linalg.norm(F, axis=1)
    out[5, 0] = float(np.sum(np.linalg.norm(F, axis=1)))
    out[5, 1] = sc
    if n > 2:
        out[5, 2] = float(np.sum(sets[0:n, :]))
    if n > 3:
        out[5, 3] = float(np.sum(D))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(51)\ndd=_hor(51)\nV=_vol(51)\n',
            "call": "pd_audit(X, x, dd, V, 210e9, 135.0, 'plane_strain', 0.001, 1e-4)",
            "gold_call": "_oracle_pd_audit(X, x, dd, V, 210e9, 135.0, 'plane_strain', 0.001, 1e-4)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(52,8,0.03)\ndd=_hor(52,8,0.012,0.024)\nV=_vol(52,8)\n',
            "call": "pd_audit(X, x, dd, V, 72e9, 204.0, 'plane_stress', 0.002, 1e-4)",
            "gold_call": "_oracle_pd_audit(X, x, dd, V, 72e9, 204.0, 'plane_stress', 0.002, 1e-4)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(54,6,0.02)\ndd=np.full(6,0.05)\nV=_vol(54,6)\n',
            "call": "pd_audit(X, x, dd, V, 72e9, 135.0, 'plane_strain', 0.0015, 1e-4)",
            "gold_call": "_oracle_pd_audit(X, x, dd, V, 72e9, 135.0, 'plane_strain', 0.0015, 1e-4)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(53,5,0.012)\ndd=_hor(53,5,0.007,0.015)\nV=_vol(53,5)\n',
            "call": "pd_audit(X, x, dd, V, 190e9, 135.0, 'plane_strain', 0.0015, 2e-4)",
            "gold_call": "_oracle_pd_audit(X, x, dd, V, 190e9, 135.0, 'plane_strain', 0.0015, 2e-4)",
        },
    ]
