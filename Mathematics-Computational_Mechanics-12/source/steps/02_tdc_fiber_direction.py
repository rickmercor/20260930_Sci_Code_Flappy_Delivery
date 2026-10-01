"""
Obtain the embedded fiber tangent field from the level-set function and normalise it.

A fiber family is defined implicitly as the intersection of level sets of a scalar function with the explicitly defined membrane surface.

Returns
-------
numpy float64 array of shape (4, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_fiber_direction(normal, grad_phi):
    """normal: (3,) surface unit normal. grad_phi: (3,) full gradient of the level-set function.
    Returns a numpy float64 array of shape (4, 3): row 0 the surface gradient of the level-set
    function, row 1 the (non-unit) fiber tangent field, row 2 the unit fiber tangent, and row 3 the
    fiber tangent magnitude followed by its inner products with the normal and the surface
    gradient.

    

    Raises
    ------
    ValueError
        Non-finite or wrong-size vectors, zero normal, or zero surface-gradient magnitude.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np



def _f64(a):
    np = __import__("numpy")
    return np.asarray(a, dtype=np.float64)

def _unit(v, what):
    np = __import__("numpy")
    v = _f64(v).ravel()
    if v.size != 3:
        raise ValueError("bad " + what)
    if not np.all(np.isfinite(v)):
        raise ValueError("non-finite " + what)
    nrm = float(np.linalg.norm(v))
    if nrm <= 0.0:
        raise ValueError("zero-length " + what)
    return v / nrm

def _mat3(a, what):
    np = __import__("numpy")
    a = _f64(a)
    if a.shape != (3, 3) or not np.all(np.isfinite(a)):
        raise ValueError("bad " + what)
    return a

def _oracle_tdc_fiber_direction(normal, grad_phi):
    """Fiber tangent from the level-set surface gradient."""
    np = __import__("numpy")
    n = _unit(normal, "normal")
    g = _f64(grad_phi).ravel()
    if g.size != 3 or not np.all(np.isfinite(g)):
        raise ValueError("bad grad_phi")
    p = np.eye(3) - np.outer(n, n)
    g_surf = p @ g                          # surface gradient is the tangential part
    t_star = np.cross(n, g_surf)            # Eq (2.6), n x gradGamma phi
    ns = float(np.linalg.norm(t_star))
    if ns <= 0.0:
        raise ValueError("degenerate fiber direction")
    t = t_star / ns                         # Eq (2.7)
    out = np.zeros((4, 3), dtype=np.float64)
    out[0, :] = g_surf
    out[1, :] = t_star
    out[2, :] = t
    out[3, 0] = ns
    out[3, 1] = float(np.dot(t, n))         # fiber is tangential: must vanish
    out[3, 2] = float(np.dot(t, g_surf))    # orthogonal to the surface gradient
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(11, 0.08)\n',
            "call": 'tdc_fiber_direction(n, gphi)',
            "gold_call": '_oracle_tdc_fiber_direction(n, gphi)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(12, 0.08)\n',
            "call": 'tdc_fiber_direction(n, gphi)',
            "gold_call": '_oracle_tdc_fiber_direction(n, gphi)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(13, 0.08)\n',
            "call": 'tdc_fiber_direction(n, gphi)',
            "gold_call": '_oracle_tdc_fiber_direction(n, gphi)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(14, 0.08)\n',
            "call": 'tdc_fiber_direction(n, gphi)',
            "gold_call": '_oracle_tdc_fiber_direction(n, gphi)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_fiber_direction([0, 0, 1], [0, 0, 0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_fiber_direction([0, 0, 1], [0, 0, 0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
