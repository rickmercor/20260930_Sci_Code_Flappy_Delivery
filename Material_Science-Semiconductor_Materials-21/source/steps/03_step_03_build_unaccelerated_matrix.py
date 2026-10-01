"""
Assemble the two-by-two error amplification matrix of the unaccelerated inner loop.

In the unaccelerated loop nothing stands between the kinetic sweep and the next equilibrium: the new energy densities are obtained by integrating the freshly swept error over the solid angle, and those energies immediately drive the following sweep. Each species' scattering source is a linear combination of the two energy-density errors carried over from the previous iterate, so sweeping and then integrating multiplies that combination by a single solid-angle integral - the plain one - evaluated at that species' own argument. The electron argument is built on the electron-phonon relaxation scale and the phonon argument on the combined phonon transport scale, which is the one place where the reduction of the two phonon channels reaches the kinetic side of the calculation rather than the macroscopic side. The squared-projection integral plays no part here; it enters only through the closure term of the accelerated scheme. The matrix that results is the entire content of the plain loop, and it also survives as a factor inside the accelerated operator, so an error in it corrupts both convergence rates rather than one.

Returns
-------
np.ndarray of shape (2, 2), float: the unaccelerated amplification matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_unaccelerated_matrix(coefficients: np.ndarray, moments_e: np.ndarray,
                               moments_p: np.ndarray) -> np.ndarray:
    """Assemble the amplification matrix of the unaccelerated inner loop.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (11,) as returned by ``compute_system_coefficients``.
    moments_e, moments_p : np.ndarray
        Shape (2,) each, as returned by ``compute_angular_moments`` at the
        electron and at the phonon argument respectively.

    Returns
    -------
    unaccelerated : np.ndarray
        Real array of shape (2, 2) mapping the electron and phonon
        energy-density error amplitudes of one iterate onto those of the next,
        electron row first.

    Raises
    ------
    ValueError
        If ``coefficients`` is not a finite array of shape (11,), or if
        ``moments_e`` or ``moments_p`` is not a finite array of shape (2,).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros((2, 2), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_unaccelerated_matrix(coefficients: np.ndarray, moments_e: np.ndarray,
                                       moments_p: np.ndarray) -> np.ndarray:
    import numpy as np

    coefficients = np.asarray(coefficients, dtype=float).ravel()
    moments_e = np.asarray(moments_e, dtype=float).ravel()
    moments_p = np.asarray(moments_p, dtype=float).ravel()
    if coefficients.shape != (11,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be a finite array of shape (11,)")
    if (moments_e.shape != (2,) or moments_p.shape != (2,)
            or not np.all(np.isfinite(moments_e))
            or not np.all(np.isfinite(moments_p))):
        raise ValueError("moments_e and moments_p must be finite arrays of shape (2,)")

    c1, c2, c3, c4 = coefficients[7:11]
    return np.array([[c1 * moments_e[0], c2 * moments_e[0]],
                     [c3 * moments_p[0], c4 * moments_p[0]]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the exact benchmark peak inputs and reported plain matrix.
        {"setup": ("import numpy as np\nkn = 10.0 ** 0.5\n"
                   "co = _oracle_compute_system_coefficients(kn, kn, kn)\n"
                   "me = _oracle_compute_angular_moments(kn)\n"
                   "mp = _oracle_compute_angular_moments(float(co[0]))\n"),
         "call": "build_unaccelerated_matrix(co, me, mp)",
         "gold_call": "_oracle_build_unaccelerated_matrix(co, me, mp)"},
        # Normal: a transition-regime coefficient set with its two integral pairs.
        {"setup": ("import numpy as np\n"
                   "co = np.array([1.25, 0.2, 0.2, 0.41, 0.41, 0.102, 0.305, 0.0311, 0.0311, 0.0204, 0.0512])\n"
                   "me = np.array([5.982, 1.052])\nmp = np.array([9.204, 2.413])\n"),
         "call": "build_unaccelerated_matrix(co, me, mp)",
         "gold_call": "_oracle_build_unaccelerated_matrix(co, me, mp)"},
        # Edge: strongly asymmetric species, ballistic phonons.
        {"setup": ("import numpy as np\n"
                   "co = np.array([40.0, 0.02, 0.05, 3.1, 0.2, 0.9, 12.0, 0.07, 0.005, 0.002, 0.06])\n"
                   "me = np.array([12.0, 4.0])\nmp = np.array([0.05, 0.0002])\n"),
         "call": "build_unaccelerated_matrix(co, me, mp)",
         "gold_call": "_oracle_build_unaccelerated_matrix(co, me, mp)"},
        # Edge: isotropic integrals, as reached at the diffusive end of the sweep.
        {"setup": ("import numpy as np\n"
                   "co = np.array([0.005, 100.0, 100.0, 0.0016, 0.0016, 0.0004, 0.0012,"
                   " 0.0271, 0.0433, 0.0155, 0.0637])\n"
                   "iso = np.array([4 * np.pi, 4 * np.pi / 3])\n"),
         "call": "build_unaccelerated_matrix(co, iso, iso)",
         "gold_call": "_oracle_build_unaccelerated_matrix(co, iso, iso)"},
        # Boundary: a species whose scattering source vanishes gives a zero row.
        {"setup": ("import numpy as np\n"
                   "co = np.array([1.0, 0.5, 0.5, 0.2, 0.2, 0.1, 0.3, 0.0, 0.0, 0.03, 0.04])\n"
                   "me = np.array([9.0, 2.0])\nmp = np.array([7.0, 1.5])\n"),
         "call": "build_unaccelerated_matrix(co, me, mp)",
         "gold_call": "_oracle_build_unaccelerated_matrix(co, me, mp)"},
        # Invalid: non-finite moments must not silently produce a non-finite matrix.
        {"setup": ("import numpy as np\n"
                   "co = np.ones(11)\nme = np.array([np.nan, 1.0])\nmp = np.ones(2)\n"
                   "def run_model():\n"
                   "    try:\n        build_unaccelerated_matrix(co, me, mp)\n        return 0\n"
                   "    except ValueError:\n        return 1\n"
                   "    except Exception:\n        return 2\n"
                   "def run_gold():\n"
                   "    try:\n        _oracle_build_unaccelerated_matrix(co, me, mp)\n        return 0\n"
                   "    except ValueError:\n        return 1\n"
                   "    except Exception:\n        return 2\n"),
         "call": "run_model()",
         "gold_call": "run_gold()"},
    ]
