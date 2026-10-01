"""
Form the directional surface gradient of the displacement field for the given configuration.

The surface operators of tangential differential calculus are the full operators restricted to the tangent space.

Returns
-------
numpy float64 array of shape (3, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_directional_gradient(grad_u, projector):
    """grad_u: (3, 3) full gradient of the displacement field. projector: (3, 3) tangential
    projector of the relevant configuration.
    Returns the directional surface gradient as a numpy float64 array of shape (3, 3).

    

    Raises
    ------
    ValueError
        Either input is not a finite (3, 3) array.
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

def _oracle_tdc_directional_gradient(grad_u, projector):
    """Directional surface gradient: the full gradient projected onto the tangent space."""
    np = __import__("numpy")
    if np.asarray(grad_u).shape != (3, 3):
        raise ValueError("gradient must have shape (3, 3)")
    G = _mat3(grad_u, "grad_u")
    P = _mat3(projector, "projector")
    return (G @ P).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(21, 0.08)\nP = np.eye(3) - np.outer(N, N)\n',
            "call": 'tdc_directional_gradient(Grad_u, P)',
            "gold_call": '_oracle_tdc_directional_gradient(Grad_u, P)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(22, 0.08)\nP = np.eye(3) - np.outer(N, N)\n',
            "call": 'tdc_directional_gradient(Grad_u, P)',
            "gold_call": '_oracle_tdc_directional_gradient(Grad_u, P)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(23, 0.08)\nP = np.eye(3) - np.outer(N, N)\n',
            "call": 'tdc_directional_gradient(Grad_u, P)',
            "gold_call": '_oracle_tdc_directional_gradient(Grad_u, P)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_directional_gradient(np.zeros((2, 3)), np.eye(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_directional_gradient(np.zeros((2, 3)), np.eye(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
