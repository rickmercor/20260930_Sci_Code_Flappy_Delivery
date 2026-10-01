"""
For a cloud of material points in a reference and a current configuration, compute the stretch of every bond and whether each bond is still intact given a critical stretch.

The bond stretch is the relative change in length of the segment joining two material points. It drives both the force law and the irreversible failure criterion.

Returns
-------
numpy float64 array of shape (2n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_bond_state(X, x, s_c):
    """Return the bond stretches and the bond intactness flags for a cloud of n material
    points. Returns a numpy float64 array of shape (2n, n)."""
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

def _oracle_pd_bond_state(X, x, s_c):
    """Bond stretch s_ij = (||x_ij|| - ||X_ij||) / ||X_ij|| and the intactness flag
    mu_ij = 1 while s_ij < s_c, else 0. Returns (2n, n): stretches then mu."""
    P = _points(X, "X")
    Q = _points(x, "x")
    if Q.shape != P.shape:
        raise ValueError("current and reference configurations differ in shape")
    sc = _pos(s_c, "s_c")
    n = P.shape[0]
    L0 = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    L1 = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
    D = np.eye(n)
    S = (L1 - L0) / (L0 + D)       # diagonal guarded, then zeroed
    np.fill_diagonal(S, 0.0)
    MU = (S < sc).astype(np.float64)
    np.fill_diagonal(MU, 0.0)
    out = np.zeros((2 * n, n), dtype=np.float64)
    out[0:n, :] = S
    out[n:2 * n, :] = MU
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(21)\nsc=6e-4\n',
            "call": 'pd_bond_state(X, x, sc)',
            "gold_call": '_oracle_pd_bond_state(X, x, sc)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(22,8,0.03)\nsc=2e-4\n',
            "call": 'pd_bond_state(X, x, sc)',
            "gold_call": '_oracle_pd_bond_state(X, x, sc)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(23,5,0.01)\nsc=1e-3\n',
            "call": 'pd_bond_state(X, x, sc)',
            "gold_call": '_oracle_pd_bond_state(X, x, sc)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(24,6,0.02)\nsc=1e3\n',
            "call": 'pd_bond_state(X, x, sc)',
            "gold_call": '_oracle_pd_bond_state(X, x, sc)',
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(25,6,0.02)\nsc=1e-12\n',
            "call": 'pd_bond_state(X, x, sc)',
            "gold_call": '_oracle_pd_bond_state(X, x, sc)',
        },
    ]
