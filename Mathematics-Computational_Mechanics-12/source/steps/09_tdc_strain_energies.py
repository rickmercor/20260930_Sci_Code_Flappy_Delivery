"""
Evaluate the stored energy density of the membrane and of the fibers for general multi-term Ogden laws, together with the two derivatives the source needs for its stress tensors. Both energies are written for incompressible material, but the membrane and the fiber eliminate their transverse stretches by different relations, so the two energy expressions do not have the same form.

The membrane energy is the general Ogden form reduced to a surface with the volume ratio set to one. The fiber energy is the corresponding one-dimensional form written on the fiber stretch alone.

Returns
-------
numpy float64 array of shape (2, 3): row 0 the membrane energy density, the fiber energy density and their sum, row 1 the derivative of the membrane energy with respect to the first membrane stretch, the derivative of the fiber energy with respect to the fiber stretch, and zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def tdc_strain_energies(lam_m, lam_f, mu_m, alpha_m, mu_f, alpha_f):
    """lam_m: (2,) membrane principal stretches. lam_f: positive float fiber stretch.
    mu_m, alpha_m: membrane Ogden moduli and exponents. mu_f, alpha_f: fiber Ogden moduli and
    exponents.
    Returns a numpy float64 array of shape (2, 3): row 0 the membrane energy density, the fiber
    energy density and their sum, row 1 the derivative of the membrane energy with respect to
    the first membrane stretch, the derivative of the fiber energy with respect to the fiber
    stretch, and zero.

    

    Raises
    ------
    ValueError
        Membrane stretches do not have two finite positive entries, the fiber stretch is not finite and positive, or either material law has empty/mismatched/non-finite parameter arrays or a zero exponent.
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


def _oracle_tdc_strain_energies(lam_m, lam_f, mu_m, alpha_m, mu_f, alpha_f):
    """Membrane and fiber stored energy densities."""
    np = __import__("numpy")
    lm = _f64(lam_m).ravel()
    if lm.size != 2 or not np.all(np.isfinite(lm)) or np.any(lm <= 0.0):
        raise ValueError("bad membrane stretches")
    lf = float(lam_f)
    if not np.isfinite(lf) or lf <= 0.0:
        raise ValueError("bad fiber stretch")
    mm, am = _ogden(mu_m, alpha_m, "membrane law")
    mf, af = _ogden(mu_f, alpha_f, "fiber law")
    l1, l2 = float(lm[0]), float(lm[1])
    Wm = float(np.sum(mm / am * (l1 ** am + l2 ** am + (l1 * l2) ** (-am) - 3.0)))   # Eq (3.13)
    Wf = float(np.sum(mf / af * (lf ** af + 2.0 * lf ** (-af / 2.0) - 3.0)))          # Eq (4.12)
    dWm1 = float(np.sum(mm * (l1 ** (am - 1.0) - l1 ** (-am - 1.0) * l2 ** (-am))))
    dWf = float(np.sum(mf * (lf ** (af - 1.0) - lf ** (-af / 2.0 - 1.0))))            # Eq (4.13)
    out = np.zeros((2, 3), dtype=np.float64)
    out[0, :] = np.array([Wm, Wf, Wm + Wf])
    out[1, :] = np.array([dWm1, dWf, 0.0])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nlam_m = np.array([1.10, 0.90])\nlam_f = 1.05\n',
            'call': 'tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
            'gold_call': '_oracle_tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
        },
        {
            'setup': 'import numpy as np\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nlam_m = np.array([1.0, 1.0])\nlam_f = 1.0\n',
            'call': 'tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
            'gold_call': '_oracle_tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
        },
        {
            'setup': 'import numpy as np\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nlam_m = np.array([1.85, 0.62])\nlam_f = 0.74\n',
            'call': 'tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
            'gold_call': '_oracle_tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
        },
        {
            'setup': 'import numpy as np\nMU_M = [0.63, 0.0012, -0.01]\nAL_M = [1.3, 5.0, -2.0]\nMU_F = [0.4225]\nAL_F = [2.0]\nlam_m = np.array([0.9858501033, 0.8147078555])\nlam_f = 0.8395585057\n',
            'call': 'tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
            'gold_call': '_oracle_tdc_strain_energies(lam_m, lam_f, MU_M, AL_M, MU_F, AL_F)',
        },
        {'setup': 'import numpy as np\ndef run_model_invalid():\n    try:\n        tdc_strain_energies([1.0,-1.0], 1.0, [0.63], [1.3], [0.4225], [2.0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_tdc_strain_energies([1.0,-1.0], 1.0, [0.63], [1.3], [0.4225], [2.0])\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'},
    ]
