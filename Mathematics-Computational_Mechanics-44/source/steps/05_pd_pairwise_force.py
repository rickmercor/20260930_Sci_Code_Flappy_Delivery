"""
Assemble the internal force density at each material point from the bond stretches, the intactness flags, the neighbour relations, the per-point micro-moduli and the point volumes.

Each intact bond contributes a force along the current bond direction, weighted by the bond stiffness and the neighbour's volume. Which stiffness applies to a given bond is fixed by the formulation.

Returns
-------
numpy float64 array of shape (n, 2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_pairwise_force(X, x, sets, state, c_j, V):
    """Return the internal force density at each material point. Returns a numpy float64
    array of shape (n, 2)."""
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

def _oracle_pd_pairwise_force(X, x, sets, state, c_j, V):
    """Eq (17): f_i = sum over H_i of c(delta_j) mu_ij s_ij (x_ij/||x_ij||) V_j
    PLUS sum over H'_i of c(delta_i) mu_ij s_ij (x_ij/||x_ij||) V_j. The own-horizon term
    carries the NEIGHBOUR's micro-modulus, the dual term the POINT's OWN; the minus sign of
    Eq (17) cancels against x_ji = -x_ij. A pair in both sets contributes twice. Returns (n, 2)."""
    P = _points(X, "X")
    Q = _points(x, "x")
    n = P.shape[0]
    St = _f64(sets)
    Sb = _f64(state)
    if St.shape != (2 * n, n) or Sb.shape != (2 * n, n):
        raise ValueError("bad sets or state array")
    cj = _f64(c_j)
    Vv = _f64(V)
    if cj.shape != (n,) or Vv.shape != (n,):
        raise ValueError("bad micro-modulus or volume array")
    if np.any(Vv <= 0.0) or not np.all(np.isfinite(cj)):
        raise ValueError("non-positive volume or non-finite micro-modulus")
    H = St[0:n, :]                 # j is inside i's own horizon
    DUAL = St[n:2 * n, :]          # i is inside j's horizon
    S = Sb[0:n, :]
    MU = Sb[n:2 * n, :]
    dx = Q[None, :, :] - Q[:, None, :]
    L = np.linalg.norm(dx, axis=2)
    L = np.where(L == 0.0, 1.0, L)
    unit = dx / L[:, :, None]
    # Eq (17): sum_{j in H_i} f_ij V_j - sum_{j in H'_i} f_ji V_j.
    # f_ij carries the NEIGHBOUR's micro-modulus c(delta_j); f_ji carries c(delta_i), and
    # because x_ji = -x_ij the minus sign flips it back to a positive contribution. A pair
    # that lies in BOTH sets therefore contributes twice, once under each micro-modulus -
    # the two sums stay separate rather than collapsing into one neighbour list.
    w = H * MU * S * cj[None, :] * Vv[None, :] + DUAL * MU * S * cj[:, None] * Vv[None, :]
    return np.einsum("ij,ijk->ik", w, unit)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(31)\ndd=_hor(31)\nV=_vol(31)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 6e-4)\ncj=np.array([_oracle_pd_micromodulus(210e9, float(t), 'plane_strain', 0.001, 1e-4)[0] for t in dd])\n",
            "call": 'pd_pairwise_force(X, x, st, bs, cj, V)',
            "gold_call": '_oracle_pd_pairwise_force(X, x, st, bs, cj, V)',
        },
        {
            "setup": "import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(32,8,0.03)\ndd=_hor(32,8,0.010,0.022)\nV=_vol(32,8)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 3e-4)\ncj=np.array([_oracle_pd_micromodulus(72e9, float(t), 'plane_stress', 0.002, 1e-4)[0] for t in dd])\n",
            "call": 'pd_pairwise_force(X, x, st, bs, cj, V)',
            "gold_call": '_oracle_pd_pairwise_force(X, x, st, bs, cj, V)',
        },
        {
            "setup": "import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(33,5,0.012)\ndd=_hor(33,5,0.006,0.015)\nV=_vol(33,5)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 9e-4)\ncj=np.array([_oracle_pd_micromodulus(190e9, float(t), 'plane_strain', 0.0015, 1e-4)[0] for t in dd])\n",
            "call": 'pd_pairwise_force(X, x, st, bs, cj, V)',
            "gold_call": '_oracle_pd_pairwise_force(X, x, st, bs, cj, V)',
        },
        {
            "setup": "import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(34,6,0.02)\ndd=np.full(6,0.05)\nV=_vol(34,6)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 1e3)\ncj=np.array([_oracle_pd_micromodulus(72e9, float(t), 'plane_strain', 0.0015, 1e-4)[0] for t in dd])\n",
            "call": 'pd_pairwise_force(X, x, st, bs, cj, V)',
            "gold_call": '_oracle_pd_pairwise_force(X, x, st, bs, cj, V)',
        },
        {
            "setup": "import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nX,x=_cloud(35,6,0.02)\ndd=_hor(35,6,0.010,0.024)\nV=_vol(35,6)\nst=_oracle_pd_dual_horizon_sets(X, dd)\nbs=_oracle_pd_bond_state(X, x, 1e-12)\ncj=np.array([_oracle_pd_micromodulus(72e9, float(t), 'plane_strain', 0.0015, 1e-4)[0] for t in dd])\n",
            "call": 'pd_pairwise_force(X, x, st, bs, cj, V)',
            "gold_call": '_oracle_pd_pairwise_force(X, x, st, bs, cj, V)',
        },
    ]
