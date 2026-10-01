"""
Evaluate the right Cauchy-Green tensor, its principal stretches, the incompressible third stretch, the deformed thickness and the spectral reconstruction.

Strain on the membrane is measured in material coordinates, and incompressibility fixes the stretch associated with the normal direction.

Returns
-------
numpy float64 array of shape (7, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_strain_measures(F, thickness_T):
    """F: (3, 3) surface deformation gradient. thickness_T: positive float undeformed thickness.
    Returns a numpy float64 array of shape (7, 3): rows 0-2 the right Cauchy-Green tensor, row 3 the
    three principal stretches in descending in-plane order followed by the normal stretch, row 4 the deformed thickness together with the
    tensor rank and its smallest eigenvalue, row 5 the first principal direction, and row 6 the
    spectral reconstruction residual and the volume ratio.

    Order the two in-plane stretches descending, then the incompressible normal stretch. For the first principal direction, project coordinate vectors e0, e1, e2 onto the largest-eigenvalue eigenspace and normalise the first projection with norm above 1e-12. Eigenvalues within 1e-12 * max(1, max(abs(eigenvalues))) of the largest belong to that eigenspace. Orient the direction so its largest-magnitude component (lowest index on a tie) is positive. Rank uses tolerance 1e-10. Use maximum absolute entry for the reconstruction residual; row 6 column 2 is zero.

    Raises
    ------
    ValueError
        F is not a finite (3, 3) array, thickness is non-finite or nonpositive, or fewer than two positive principal stretches exist.
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

def _oracle_tdc_strain_measures(F, thickness_T):
    """Right Cauchy-Green, principal stretches, incompressible third stretch, thickness."""
    np = __import__("numpy")
    F = _mat3(F, "F")
    T = float(thickness_T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("bad thickness")
    C = F.T @ F                                        # Eq (3.6)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]                        # descending: two non-zero, one zero
    w = w[order]; V = V[:, order]
    l1 = float(np.sqrt(max(w[0], 0.0)))
    l2 = float(np.sqrt(max(w[1], 0.0)))
    if l1 <= 0.0 or l2 <= 0.0:
        raise ValueError("degenerate stretches")
    l3 = 1.0 / (l1 * l2)                               # incompressibility J = 1
    t_def = l3 * T                                     # deformed thickness
    Crec = (l1 ** 2) * np.outer(V[:, 0], V[:, 0]) + (l2 ** 2) * np.outer(V[:, 1], V[:, 1])
    out = np.zeros((7, 3), dtype=np.float64)
    out[0:3, :] = C
    out[3, :] = np.array([l1, l2, l3])
    out[4, :] = np.array([t_def, float(np.linalg.matrix_rank(C, tol=1e-10)), float(w[2])])
    eig_tol = 1e-12 * max(1.0, float(np.max(np.abs(w))))
    top = V[:, np.abs(w - w[0]) <= eig_tol]
    projector = top @ top.T
    for axis in range(3):
        direction = projector[:, axis]
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1e-12:
            direction = direction / magnitude
            break
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0.0:
        direction = -direction
    out[5, :] = direction
    out[6, :] = np.array([float(np.max(np.abs(Crec - C))), l1 * l2 * l3, 0.0])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(51, 0.08)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\nT = 0.02\n',
            "call": 'tdc_strain_measures(F, T)',
            "gold_call": '_oracle_tdc_strain_measures(F, T)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(52, 0.15)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\nT = 0.005\n',
            "call": 'tdc_strain_measures(F, T)',
            "gold_call": '_oracle_tdc_strain_measures(F, T)',
        },
        {
            "setup": 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(53, 0.03)\ndg = _oracle_tdc_deformation_gradient(N, n, Grad_u, grad_u)\nF = dg[0:3, :]\nFi = dg[3:6, :]\nP = np.eye(3) - np.outer(N, N)\np = np.eye(3) - np.outer(n, n)\nT = 0.1\n',
            "call": 'tdc_strain_measures(F, T)',
            "gold_call": '_oracle_tdc_strain_measures(F, T)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_strain_measures(np.eye(3), -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_strain_measures(np.eye(3), -1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
