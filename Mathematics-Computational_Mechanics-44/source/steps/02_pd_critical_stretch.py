"""
Convert a critical energy release rate into the critical bond stretch for the requested modelling scenario, and report it with the inputs.

Fracture enters bond-based peridynamics through irreversible bond breakage once a bond exceeds a critical stretch. The threshold follows from equating the energy needed to break every bond crossing a unit fracture area to the critical energy release rate.

Returns
-------
numpy float64 array of shape (4,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pd_critical_stretch(Gc, E, delta, mode):
    """Return the critical bond stretch for the given modelling scenario, together with
    the inputs. Returns a numpy float64 array of shape (4,)."""
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

def _oracle_pd_critical_stretch(Gc, E, delta, mode):
    """Eq (29): critical bond stretch from the critical energy release rate."""
    g = _pos(Gc, "Gc")
    e = _pos(E, "E")
    d = _pos(delta, "delta")
    m = _mode(mode)
    if m == "3d":
        s = np.sqrt(5.0 * g / (6.0 * e * d))
    elif m == "plane_stress":
        s = np.sqrt(4.0 * np.pi * g / (9.0 * e * d))
    elif m == "plane_strain":
        s = np.sqrt(5.0 * np.pi * g / (12.0 * e * d))
    else:
        raise ValueError("critical stretch undefined for mode " + m)
    return np.array([s, g, e, d], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nGc=135.0\nE=72e9\nd=0.01\n',
            "call": "pd_critical_stretch(Gc, E, d, '3d')",
            "gold_call": "_oracle_pd_critical_stretch(Gc, E, d, '3d')",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nGc=204.0\nE=210e9\nd=0.004\n',
            "call": "pd_critical_stretch(Gc, E, d, 'plane_stress')",
            "gold_call": "_oracle_pd_critical_stretch(Gc, E, d, 'plane_stress')",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nGc=135.0\nE=72e9\nd=0.008\n',
            "call": "pd_critical_stretch(Gc, E, d, 'plane_strain')",
            "gold_call": "_oracle_pd_critical_stretch(Gc, E, d, 'plane_strain')",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nGc=1e-3\nE=1e12\nd=1.0\n',
            "call": "pd_critical_stretch(Gc, E, d, '3d')",
            "gold_call": "_oracle_pd_critical_stretch(Gc, E, d, '3d')",
        },
        {
            "setup": 'import numpy as np\ndef _cloud(seed, n=6, spread=0.02):\n    rng = np.random.default_rng(seed)\n    X = rng.uniform(0.0, spread, size=(n, 2))\n    u = rng.normal(size=(n, 2)) * spread * 0.02\n    return X, X + u\ndef _hor(seed, n=6, lo=0.008, hi=0.016):\n    rng = np.random.default_rng(seed + 500)\n    return rng.uniform(lo, hi, size=n)\ndef _vol(seed, n=6):\n    rng = np.random.default_rng(seed + 900)\n    return rng.uniform(0.5e-6, 2.0e-6, size=n)\nGc=5e4\nE=1e6\nd=1e-4\n',
            "call": "pd_critical_stretch(Gc, E, d, 'plane_stress')",
            "gold_call": "_oracle_pd_critical_stretch(Gc, E, d, 'plane_stress')",
        },
    ]
