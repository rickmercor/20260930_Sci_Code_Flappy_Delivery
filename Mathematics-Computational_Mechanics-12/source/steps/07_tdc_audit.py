"""
Integrate the previous six functions into the declared six-row audit.

The audit collects the fiber orientation, the stretches, and the identity residuals that confirm the rank-deficient kinematics.

Returns
-------
numpy float64 array of shape (6, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_audit(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u, thickness_T):
    """All arguments carry the meanings fixed by the earlier steps.
    Returns a numpy float64 array of shape (6, 3).

    normal_X and normal_x are material and spatial normals. t_tilde is a material boundary tangent, and grad_phi is a material gradient. Use the material frame and material fiber direction. Rows are: 0 unit material fiber tangent; 1 membrane stretches; 2 first three step-05 residuals; 3 fourth residual, absolute determinant, inverse-product deviation; 4 deformed thickness, material projector rank, maximum-absolute-entry directional-gradient consistency residual; 5 sandwich residual, spectral reconstruction residual, volume ratio. The consistency residual compares the material directional gradient with F - P. Use maximum-absolute-entry residuals, with the step-05 identity order.

    Raises
    ------
    ValueError
        Invalid vectors or gradients as in earlier steps, nonpositive/non-finite thickness, a boundary tangent not orthogonal to normal_X within 1e-10 after normalisation, or degenerate membrane/fiber stretches.
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

def _oracle_tdc_audit(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u, thickness_T):
    """Integrates the six earlier steps."""
    np = __import__("numpy")
    if not np.isfinite(thickness_T) or thickness_T <= 0.0:
        raise ValueError("bad thickness")
    frame = _oracle_tdc_surface_frame(normal_X, t_tilde)
    fib = _oracle_tdc_fiber_direction(normal_X, grad_phi)
    dg = _oracle_tdc_deformation_gradient(normal_X, normal_x, Grad_u, grad_u)
    F = dg[0:3, :]; Finv = dg[3:6, :]
    N = _unit(normal_X, "normal_X"); n = _unit(normal_x, "normal_x")
    P = frame[0:3, :]
    p = np.eye(3) - np.outer(n, n)
    props = _oracle_tdc_deformation_properties(F, Finv, P, p)
    strain = _oracle_tdc_strain_measures(F, thickness_T)
    dirG = _oracle_tdc_directional_gradient(_mat3(Grad_u, "Grad_u"), P)   # explicit use of step 3
    out = np.zeros((6, 3), dtype=np.float64)
    out[0, :] = fib[2, :]                    # unit fiber tangent
    out[1, :] = strain[3, :]                 # principal stretches
    out[2, :] = props[0:3]                   # identities (i)-(iii) residuals
    out[3, :] = np.array([props[3], props[6], props[7]])
    out[4, :] = np.array([strain[4, 0], frame[4, 0],
                           float(np.max(np.abs(dirG - (F - P))))])
    out[5, :] = np.array([float(np.max(np.abs(F - p @ F @ P))), strain[6, 0], strain[6, 1]])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(61, 0.08)\nT = 0.02\n',
            "call": 'tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
            "gold_call": '_oracle_tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(62, 0.15)\nT = 0.005\n',
            "call": 'tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
            "gold_call": '_oracle_tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(63, 0.05)\nT = 0.05\n',
            "call": 'tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
            "gold_call": '_oracle_tdc_audit(N, n, tt, gphi, Grad_u, grad_u, T)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_audit([0,0,1], [0,0,1], [1,0,0], [1,0,0], np.zeros((3,3)), np.zeros((3,3)), -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_audit([0,0,1], [0,0,1], [1,0,0], [1,0,0], np.zeros((3,3)), np.zeros((3,3)), -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
