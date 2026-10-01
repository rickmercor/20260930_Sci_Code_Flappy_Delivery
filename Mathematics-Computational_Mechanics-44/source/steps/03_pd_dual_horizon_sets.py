"""
For a cloud of material points with individually assigned horizons, determine for each point which other points lie inside its own horizon and, separately, which other points have it inside theirs.

When horizons vary in space the neighbour relation stops being symmetric: a point can lie inside its neighbour's horizon without that neighbour lying inside its own. Keeping both relations is what allows the formulation to restore the balance of linear momentum.

Returns
-------
numpy float64 array of shape (2n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_dual_horizon_sets(X, deltas):
    """Return the horizon membership flags and the dual-horizon membership flags for a
    cloud of n material points. Returns a numpy float64 array of shape (2n, n)."""
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

def _oracle_pd_dual_horizon_sets(X, deltas):
    """H_i = { j != i : ||X_j - X_i|| <= delta_i } and the dual set H'_i = { j : i in H_j }.
    Returns a (2n, n) array: first n rows the horizon flags, next n rows the dual-horizon flags."""
    P = _points(X, "X")
    n = P.shape[0]
    d = _f64(deltas)
    if d.shape != (n,) or not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise ValueError("bad horizon array")
    R = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    H = (R <= d[:, None]).astype(np.float64)
    np.fill_diagonal(H, 0.0)
    out = np.zeros((2 * n, n), dtype=np.float64)
    out[0:n, :] = H
    out[n:2 * n, :] = H.T          # j in H'_i  <=>  i in H_j
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,_=_cloud(11)\ndd=_hor(11)\n',
            "call": 'pd_dual_horizon_sets(X, dd)',
            "gold_call": '_oracle_pd_dual_horizon_sets(X, dd)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,_=_cloud(12,8,0.03)\ndd=_hor(12,8,0.010,0.022)\n',
            "call": 'pd_dual_horizon_sets(X, dd)',
            "gold_call": '_oracle_pd_dual_horizon_sets(X, dd)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,_=_cloud(13,5,0.015)\ndd=_hor(13,5,0.006,0.014)\n',
            "call": 'pd_dual_horizon_sets(X, dd)',
            "gold_call": '_oracle_pd_dual_horizon_sets(X, dd)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX=np.array([[0.0,0.0],[0.005,0.0]])\ndd=np.array([0.01,0.01])\n',
            "call": 'pd_dual_horizon_sets(X, dd)',
            "gold_call": '_oracle_pd_dual_horizon_sets(X, dd)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,_=_cloud(14,6,0.02)\ndd=np.full(6,0.05)\n',
            "call": 'pd_dual_horizon_sets(X, dd)',
            "gold_call": '_oracle_pd_dual_horizon_sets(X, dd)',
        },
    ]
