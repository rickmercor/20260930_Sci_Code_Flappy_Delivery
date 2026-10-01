"""
Convert the material's Young's modulus and a horizon into the micro-modulus this formulation uses for the requested modelling scenario, and report it with the Poisson's ratio the scenario constrains and the inputs.

Bond-based peridynamics replaces the local stress divergence with an integral of pairwise bond forces over a neighbourhood. The bond stiffness is calibrated by matching the peridynamic strain energy density to the classical one under isotropic expansion, which also fixes the Poisson's ratio the formulation can represent.

Returns
-------
numpy float64 array of shape (4,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_micromodulus(E, delta, mode, z, A):
    """Return the dual-horizon bond-based peridynamic micro-modulus for the given
    modelling scenario, together with the constrained Poisson's ratio and the inputs.
    Returns a numpy float64 array of shape (4,)."""
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

def _oracle_pd_micromodulus(E, delta, mode, z, A):
    """Eq (25): DHBB-PD micro-modulus, half the classical SHBB-PD value."""
    e = _pos(E, "E")
    d = _pos(delta, "delta")
    m = _mode(mode)
    if m == "1d":
        a = _pos(A, "A")
        c, nu = e / (d * d * a), 0.0      # 1D bar imposes no Poisson constraint; spec fixes 0.0
    elif m == "3d":
        c, nu = 6.0 * e / (np.pi * d ** 4), 0.25
    elif m == "plane_stress":
        t = _pos(z, "z")
        c, nu = 9.0 * e / (2.0 * np.pi * d ** 3 * t), 1.0 / 3.0
    else:
        t = _pos(z, "z")
        c, nu = 24.0 * e / (5.0 * np.pi * d ** 3 * t), 0.25
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError("non-finite or non-positive micro-modulus")
    return np.array([c, nu, e, d], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=210e9\nd=0.01\nz=0.001\nA=1e-4\n',
            "call": "pd_micromodulus(E, d, '3d', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, '3d', z, A)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=72e9\nd=0.005\nz=0.002\nA=1e-4\n',
            "call": "pd_micromodulus(E, d, 'plane_stress', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, 'plane_stress', z, A)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=190e9\nd=0.012\nz=0.0015\nA=2e-4\n',
            "call": "pd_micromodulus(E, d, 'plane_strain', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, 'plane_strain', z, A)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=70e9\nd=0.02\nz=0.001\nA=5e-5\n',
            "call": "pd_micromodulus(E, d, '1d', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, '1d', z, A)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=1.0\nd=1e-4\nz=1e-6\nA=1e-8\n',
            "call": "pd_micromodulus(E, d, 'plane_strain', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, 'plane_strain', z, A)",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nE=1e12\nd=1.0\nz=10.0\nA=1.0\n',
            "call": "pd_micromodulus(E, d, '3d', z, A)",
            "gold_call": "_oracle_pd_micromodulus(E, d, '3d', z, A)",
        },
    ]
