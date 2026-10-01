"""
Build the tangential projector of a surface and the conormal vector of its boundary triad, and report the projector's defining properties. The boundary tangent supplied to this step lies in the membrane surface, as the source assumes along the boundary; it is orthogonal to the surface normal.

Quantities on a curved surface are handled by projecting onto its tangent space; the boundary triad is later used to prescribe conditions.

Returns
-------
numpy float64 array of shape (5, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_surface_frame(normal, t_tilde):
    """normal: (3,) array-like surface unit normal. t_tilde: (3,) array-like boundary tangent.
    Returns a numpy float64 array of shape (5, 3): rows 0-2 hold the tangential projector, row 3
    holds the conormal vector, and row 4 holds the projector rank, its idempotency residual and the
    norm of the projector applied to the normal.

    Normalise the two input vectors before computing the frame. Their absolute inner product must be at most 1e-10. Use the boundary conormal t_tilde cross normal. The idempotency residual is the maximum absolute entry; the projected-normal norm is Euclidean. Rank uses singular-value tolerance 1e-12.

    Raises
    ------
    ValueError
        Non-finite or wrong-size vectors, zero normal/tangent, or absolute dot product of the normalised vectors above 1e-10.
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

def _oracle_tdc_surface_frame(normal, t_tilde):
    """Tangential projector and the boundary triad."""
    np = __import__("numpy")
    n = _unit(normal, "normal")
    tt = _unit(t_tilde, "t_tilde")
    if abs(float(n @ tt)) > 1e-10:
        raise ValueError("boundary tangent must be orthogonal to normal")
    p = np.eye(3) - np.outer(n, n)          # Eq (2.3)
    m = np.cross(tt, n)                     # Eq (2.2), this order
    out = np.zeros((5, 3), dtype=np.float64)
    out[0:3, :] = p
    out[3, :] = m
    out[4, 0] = float(np.linalg.matrix_rank(p, tol=1e-12))
    out[4, 1] = float(np.max(np.abs(p @ p - p)))     # idempotent
    out[4, 2] = float(np.linalg.norm(p @ n))         # p n = 0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(1, 0.08)\n',
            "call": 'tdc_surface_frame(N, tt)',
            "gold_call": '_oracle_tdc_surface_frame(N, tt)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(2, 0.08)\n',
            "call": 'tdc_surface_frame(N, tt)',
            "gold_call": '_oracle_tdc_surface_frame(N, tt)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(3, 0.08)\n',
            "call": 'tdc_surface_frame(N, tt)',
            "gold_call": '_oracle_tdc_surface_frame(N, tt)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(4, 0.08)\n',
            "call": 'tdc_surface_frame(N, tt)',
            "gold_call": '_oracle_tdc_surface_frame(N, tt)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_surface_frame([0, 0, 1], [0, 0, 1])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_surface_frame([0, 0, 1], [0, 0, 1])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
