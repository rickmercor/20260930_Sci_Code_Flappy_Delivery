"""
Verify the paper's algebraic identities for the surface deformation gradient and report its rank and singularity.

Because the in-plane deformation gradient is rank deficient, its products with its inverse do not return the identity.

Returns
-------
numpy float64 array of shape (8,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_deformation_properties(F, Finv, P, p):
    """F, Finv: (3, 3) surface deformation gradient and its inverse. P, p: (3, 3) tangential
    projectors of the undeformed and deformed configuration.
    Returns a numpy float64 array of shape (8,) holding the residuals of the paper's four identities
    followed by the ranks of the two tensors, the magnitude of the determinant of the deformation
    gradient, and the deviation of its product with its inverse from the identity matrix.

    Use the maximum absolute entry for all four identity residuals and the final deviation; do not use a Frobenius norm. The identity order is Finv F - P, F Finv - p, p F P - F, P Finv p - Finv. Both ranks use singular-value tolerance 1e-10.

    Raises
    ------
    ValueError
        Any input is not a finite (3, 3) array.
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

def _oracle_tdc_deformation_properties(F, Finv, P, p):
    """The four rank-two identities of the surface deformation gradient."""
    np = __import__("numpy")
    if np.asarray(F).shape != (3, 3):
        raise ValueError("F must have shape (3, 3)")
    F = _mat3(F, "F"); Fi = _mat3(Finv, "Finv")
    P = _mat3(P, "P"); pp = _mat3(p, "p")
    out = np.zeros(8, dtype=np.float64)
    out[0] = float(np.max(np.abs(Fi @ F - P)))        # (i)   Finv F = P
    out[1] = float(np.max(np.abs(F @ Fi - pp)))       # (ii)  F Finv = p
    out[2] = float(np.max(np.abs(pp @ F @ P - F)))    # (iii) F = p F P
    out[3] = float(np.max(np.abs(P @ Fi @ pp - Fi)))  # (iv)  Finv = P Finv p
    out[4] = float(np.linalg.matrix_rank(F, tol=1e-10))
    out[5] = float(np.linalg.matrix_rank(Fi, tol=1e-10))
    out[6] = float(np.abs(np.linalg.det(F)))          # singular: vanishes
    out[7] = float(np.max(np.abs(F @ Fi - np.eye(3))))  # NOT the identity
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(41, 0.08)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\n',
            "call": 'tdc_deformation_properties(F, Fi, P, p)',
            "gold_call": '_oracle_tdc_deformation_properties(F, Fi, P, p)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(42, 0.15)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\n',
            "call": 'tdc_deformation_properties(F, Fi, P, p)',
            "gold_call": '_oracle_tdc_deformation_properties(F, Fi, P, p)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(43, 0.05)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\n',
            "call": 'tdc_deformation_properties(F, Fi, P, p)',
            "gold_call": '_oracle_tdc_deformation_properties(F, Fi, P, p)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_deformation_properties(np.zeros((2, 3)), np.eye(3), np.eye(3), np.eye(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_deformation_properties(np.zeros((2, 3)), np.eye(3), np.eye(3), np.eye(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
