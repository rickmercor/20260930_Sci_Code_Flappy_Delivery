"""
Assemble the in-plane surface deformation gradient, its inverse, and both transposes, each from the projector of the correct configuration.

The membrane deformation is described by an in-plane deformation gradient relating the undeformed and deformed manifolds.

Returns
-------
numpy float64 array of shape (12, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_deformation_gradient(normal_X, normal_x, Grad_u, grad_u):
    """normal_X, normal_x: (3,) unit normals of the undeformed and deformed configuration.
    Grad_u: (3, 3) displacement gradient with respect to the undeformed configuration.
    grad_u: (3, 3) displacement gradient with respect to the deformed configuration.
    Returns a numpy float64 array of shape (12, 3): rows 0-2 the in-plane surface deformation
    gradient, rows 3-5 its inverse, rows 6-8 its transpose, rows 9-11 the transpose of its
    inverse.

    

    Raises
    ------
    ValueError
        A normal is non-finite, has other than three entries or is zero; either displacement gradient is not a finite (3, 3) array.
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

def _oracle_tdc_deformation_gradient(normal_X, normal_x, Grad_u, grad_u):
    """F from the UNDEFORMED projector, its inverse from the DEFORMED one."""
    np = __import__("numpy")
    if np.asarray(normal_X).size != 3:
        raise ValueError("normal_X must have three entries")
    N = _unit(normal_X, "normal_X")
    n = _unit(normal_x, "normal_x")
    P = np.eye(3) - np.outer(N, N)
    p = np.eye(3) - np.outer(n, n)
    GU = _mat3(Grad_u, "Grad_u")
    gu = _mat3(grad_u, "grad_u")
    F = P + _oracle_tdc_directional_gradient(GU, P)          # Eq (3.2)
    Finv = p - _oracle_tdc_directional_gradient(gu, p)       # Eq (3.3): formula, not matrix inversion
    out = np.zeros((12, 3), dtype=np.float64)
    out[0:3, :] = F
    out[3:6, :] = Finv
    out[6:9, :] = F.T                                 # Eq (3.4)
    out[9:12, :] = Finv.T                             # Eq (3.5)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(31, 0.08)\n',
            "call": 'tdc_deformation_gradient(N, n, Grad_u, grad_u)',
            "gold_call": '_oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(32, 0.15)\n',
            "call": 'tdc_deformation_gradient(N, n, Grad_u, grad_u)',
            "gold_call": '_oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(33, 0.03)\n',
            "call": 'tdc_deformation_gradient(N, n, Grad_u, grad_u)',
            "gold_call": '_oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(34, 0.1)\n',
            "call": 'tdc_deformation_gradient(N, n, Grad_u, grad_u)',
            "gold_call": '_oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_deformation_gradient([0, 0], [0, 0, 1], np.zeros((3, 3)), np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_deformation_gradient([0, 0], [0, 0, 1], np.zeros((3, 3)), np.zeros((3, 3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
