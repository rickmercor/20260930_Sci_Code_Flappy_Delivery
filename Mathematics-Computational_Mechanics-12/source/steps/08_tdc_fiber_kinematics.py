"""
Build the fiber projector from the unit fiber tangent and form the fiber deformation gradient and the fiber right Cauchy-Green tensor by the same construction the source uses for the membrane. Report the three fiber principal stretches. The fiber tensor is not of the same rank as the membrane one, and the relation that fixes the two transverse fiber stretches from the first is the one the source prescribes for incompressible fibers; it is not the membrane relation.

The source defines the fiber deformation gradient and its inverse exactly as for the membrane but with the fiber projector in place of the tangential projector, and derives the fiber stretch from the single non-zero eigenvalue of the fiber right Cauchy-Green tensor.

Returns
-------
numpy float64 array of shape (6, 3): rows 0 to 2 the fiber right Cauchy-Green tensor, row 3 the three fiber principal stretches, row 4 the rank of that tensor together with its two remaining eigenvalues, row 5 the unit fiber tangent.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_fiber_kinematics(normal_X, grad_phi, Grad_u):
    """normal_X: (3,) undeformed surface unit normal. grad_phi: (3,) full gradient of the
    level-set function. Grad_u: (3, 3) full material gradient of the displacement.
    Returns a numpy float64 array of shape (6, 3): rows 0-2 the fiber right Cauchy-Green
    tensor, row 3 the three fiber principal stretches, row 4 the rank of that tensor together
    with its two remaining eigenvalues, and row 5 the unit fiber tangent.

    

    Raises
    ------
    ValueError
        Invalid normal or gradient shapes/finiteness, zero normal or surface fiber tangent, or nonpositive axial fiber stretch.
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

def _ogden(mu, alpha, what):
    np = __import__("numpy")
    mu = _f64(mu).ravel()
    alpha = _f64(alpha).ravel()
    if mu.size != alpha.size or mu.size == 0:
        raise ValueError("bad " + what)
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(alpha))):
        raise ValueError("non-finite " + what)
    if np.any(alpha == 0.0):
        raise ValueError("zero exponent in " + what)
    return mu, alpha


def _oracle_tdc_fiber_kinematics(normal_X, grad_phi, Grad_u):
    """Fiber projector, fiber deformation gradient and the fiber principal stretches."""
    np = __import__("numpy")
    N = _unit(normal_X, "normal_X")
    GU = _mat3(Grad_u, "Grad_u")
    T = _oracle_tdc_fiber_direction(N, grad_phi)[2, :]      # unit fiber tangent
    PY = np.outer(T, T)                                     # fiber projector, rank one
    FY = PY + _oracle_tdc_directional_gradient(GU, PY)      # Eq (4.1)
    CY = FY.T @ FY                                          # Eq (4.5), rank one
    w = np.sort(np.linalg.eigvalsh(CY))[::-1]
    if w[0] <= 0.0:
        raise ValueError("degenerate fiber stretch")
    l1 = float(np.sqrt(w[0]))                               # Eq (4.6) fiber stretch
    l23 = 1.0 / np.sqrt(l1)                                 # incompressible fibers, source rule
    out = np.zeros((6, 3), dtype=np.float64)
    out[0:3, :] = CY
    out[3, :] = np.array([l1, l23, l23])
    out[4, :] = np.array([float(np.linalg.matrix_rank(CY, tol=1e-12)), float(w[1]), float(w[2])])
    out[5, :] = T
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(71, 0.08)\n',
            'call': 'tdc_fiber_kinematics(N, gphi, Grad_u)',
            'gold_call': '_oracle_tdc_fiber_kinematics(N, gphi, Grad_u)',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(72, 0.30)\n',
            'call': 'tdc_fiber_kinematics(N, gphi, Grad_u)',
            'gold_call': '_oracle_tdc_fiber_kinematics(N, gphi, Grad_u)',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(73, 0.0)\n',
            'call': 'tdc_fiber_kinematics(N, gphi, Grad_u)',
            'gold_call': '_oracle_tdc_fiber_kinematics(N, gphi, Grad_u)',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(74, 0.02)\nN = np.array([0.0, 0.0, 1.0])\ngphi = np.array([1.0, 0.0, 0.0])\n',
            'call': 'tdc_fiber_kinematics(N, gphi, Grad_u)',
            'gold_call': '_oracle_tdc_fiber_kinematics(N, gphi, Grad_u)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_fiber_kinematics([0,0,1], [0,0,0], np.zeros((3,3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_fiber_kinematics([0,0,1], [0,0,0], np.zeros((3,3)))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
