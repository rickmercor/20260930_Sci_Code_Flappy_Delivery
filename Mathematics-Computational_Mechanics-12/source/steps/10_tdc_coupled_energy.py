"""
The final orchestrator. Call the earlier step functions rather than reimplementing them. Chain the membrane kinematics and the fiber kinematics, evaluate both stored energies, and combine them into the total coupled energy density the way the source's total energy of the coupled model does. The membrane term carries the membrane thickness. The fiber term is not added bare: the source weights it by the fiber cross-section the section on the numerical setup fixes, and by the length measure of the un-normalised fiber tangent field, so the fiber energy density becomes a surface density like the membrane one.

The source evaluates the total energy of the coupled model as the membrane energy weighted by the membrane thickness plus, for each fiber family, the fiber energy weighted by the fiber cross-section and by the length measure that carries the fiber density onto the surface.

Returns
-------
numpy float64 array of shape (10, 3): row 0 the two membrane stretches and the fiber stretch, row 1 the membrane energy density, the fiber energy density and the total coupled energy density, row 2 the three fiber principal stretches, row 3 the thickness-weighted membrane term, the weighted fiber term and the norm of the un-normalised fiber tangent; rows 4-9 hold the complete membrane audit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_coupled_energy(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u,
                       thickness_T, mu_m, alpha_m, mu_f, alpha_f):
    """All arguments carry the meanings fixed by the earlier steps.
    Returns a numpy float64 array of shape (10, 3).

    t_tilde is a material boundary tangent and grad_phi is a material gradient. Call the membrane audit and use its stretches; append its complete six-row result. Return shape (10, 3): row 0 membrane stretches and axial fiber stretch; row 1 membrane energy, fiber energy, total energy; row 2 three fiber stretches; row 3 thickness-weighted membrane energy, weighted fiber energy, unnormalised material fiber-tangent length; rows 4-9 the six audit rows. The requested scalar remains entry [1, 2].

    Raises
    ------
    ValueError
        Any earlier-step input condition fails, including a material boundary tangent not orthogonal to normal_X within 1e-10 after normalisation, or thickness is not finite and positive.
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


def _oracle_tdc_coupled_energy(normal_X, normal_x, t_tilde, grad_phi, Grad_u, grad_u,
                               thickness_T, mu_m, alpha_m, mu_f, alpha_f):
    """Total stored energy density of the coupled fiber-membrane model."""
    np = __import__("numpy")
    T = float(thickness_T)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("bad thickness")
    N = _unit(normal_X, "normal_X")
    audit = _oracle_tdc_audit(N, normal_x, t_tilde, grad_phi, Grad_u, grad_u, T)
    l1, l2 = float(audit[1, 0]), float(audit[1, 1])
    fk = _oracle_tdc_fiber_kinematics(N, grad_phi, Grad_u)
    lf = float(fk[3, 0])
    en = _oracle_tdc_strain_energies(np.array([l1, l2]), lf, mu_m, alpha_m, mu_f, alpha_f)
    Wm, Wf = float(en[0, 0]), float(en[0, 1])
    t_star_norm = float(_oracle_tdc_fiber_direction(N, grad_phi)[3, 0])   # |T*|, Eq (2.6)
    A_tilde = T                                              # Sec 5: A~ = B T with B = 1
    Wtot = T * Wm + A_tilde * Wf * t_star_norm               # Eq (7.4)
    out = np.zeros((10, 3), dtype=np.float64)
    out[0, :] = np.array([l1, l2, lf])
    out[1, :] = np.array([Wm, Wf, Wtot])
    out[2, :] = fk[3, :]
    out[3, :] = np.array([T * Wm, A_tilde * Wf * t_star_norm, t_star_norm])
    out[4:10, :] = audit
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(61, 0.08)\nT = 0.02\n',
            'call': 'tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
            'gold_call': '_oracle_tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(81, 0.25)\nT = 0.5\n',
            'call': 'tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
            'gold_call': '_oracle_tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(82, 0.0)\nT = 1.0\n',
            'call': 'tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
            'gold_call': '_oracle_tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
        },
        {
            'setup': 'import numpy as np\ndef _kin(seed, scale=0.08):\n    rng = np.random.default_rng(seed)\n    A = rng.normal(size=(3, 3)) * scale\n    Ffull = np.eye(3) + A\n    v = rng.normal(size=3); N = v / np.linalg.norm(v)\n    nn = np.linalg.inv(Ffull).T @ N; n = nn / np.linalg.norm(nn)\n    Grad_u = A\n    grad_u = np.eye(3) - np.linalg.inv(Ffull)\n    tt = rng.normal(size=3); tt = tt - np.dot(tt, N) * N; tt = tt / np.linalg.norm(tt)\n    gphi = rng.normal(size=3)\n    return N, n, tt, gphi, Grad_u, grad_u, rng\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nN, n, tt, gphi, Grad_u, grad_u, rng = _kin(83, 0.12)\nT = 0.02\nMU_F = [0.4225, 0.05]\nAL_F = [2.0, -1.5]\n',
            'call': 'tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
            'gold_call': '_oracle_tdc_coupled_energy(N, n, tt, gphi, Grad_u, grad_u, T, MU_M, AL_M, MU_F, AL_F).ravel()',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_coupled_energy([0,0,1], [0,0,1], [1,0,0], [1,0,0], np.zeros((3,3)), np.zeros((3,3)), -1.0, [0.63], [1.3], [0.4225], [2.0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_coupled_energy([0,0,1], [0,0,1], [1,0,0], [1,0,0], np.zeros((3,3)), np.zeros((3,3)), -1.0, [0.63], [1.3], [0.4225], [2.0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
