"""
Compute the local damage at each material point from the intactness flags, the neighbour relations and the point volumes.

Local damage summarises how much of a point's neighbourhood has failed. It is the standard post-processing field used to visualise crack paths in peridynamic simulations.

Returns
-------
numpy float64 array of shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_point_damage(sets, state, V):
    """Return the local damage at each material point. Returns a numpy float64 array of
    shape (n,)."""
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

def _oracle_pd_point_damage(sets, state, V):
    """Eq (30): D_i = 1 - sum_{j in H_i} mu_ij V_j / sum_{j in H_i} V_j (volume weighted,
    over the point's OWN horizon only). Returns (n,)."""
    St = _f64(sets)
    Sb = _f64(state)
    if St.ndim != 2 or St.shape[0] % 2 != 0:
        raise ValueError("bad sets array")
    n = St.shape[0] // 2
    if St.shape != (2 * n, n) or Sb.shape != (2 * n, n):
        raise ValueError("bad sets or state array")
    Vv = _f64(V)
    if Vv.shape != (n,) or np.any(Vv <= 0.0):
        raise ValueError("bad volume array")
    H = St[0:n, :]
    MU = Sb[n:2 * n, :]
    num = (H * MU * Vv[None, :]).sum(axis=1)
    den = (H * Vv[None, :]).sum(axis=1)
    if np.any(den <= 0.0):
        raise ValueError("a material point has an empty horizon")
    return 1.0 - num / den

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(41)\ndd=_hor(41)\nV=_vol(41)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 6e-4)\n',
            "call": 'pd_point_damage(st, bs, V)',
            "gold_call": '_oracle_pd_point_damage(st, bs, V)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(42,8,0.03)\ndd=_hor(42,8,0.012,0.024)\nV=_vol(42,8)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 2e-4)\n',
            "call": 'pd_point_damage(st, bs, V)',
            "gold_call": '_oracle_pd_point_damage(st, bs, V)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(43,6,0.02)\ndd=_hor(43,6,0.010,0.020)\nV=_vol(43,6)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 5e-4)\n',
            "call": 'pd_point_damage(st, bs, V)',
            "gold_call": '_oracle_pd_point_damage(st, bs, V)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(44,6,0.02)\ndd=np.full(6,0.05)\nV=_vol(44,6)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 1e3)\n',
            "call": 'pd_point_damage(st, bs, V)',
            "gold_call": '_oracle_pd_point_damage(st, bs, V)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(45,6,0.02)\ndd=_hor(45,6,0.012,0.026)\nV=_vol(45,6)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 1e-12)\n',
            "call": 'pd_point_damage(st, bs, V)',
            "gold_call": '_oracle_pd_point_damage(st, bs, V)',
        },
    ]
