"""
Return the RPA correlation energy of the source's low-level method for a particle-hole space described by the vector delta_eps of orbital-energy differences (one entry per particle-hole pair) and the symmetric matrix v_ph of interaction integrals between pairs (element [I, J] = (ia|jb) for pairs I = (i, a) and J = (j, b), in the same pair order): build the RPA matrices from these two ingredients as the source's Eq. (2.8) does, with the sign of the coupling block fixed by its Appendix B, solve the source's eigenproblem, and evaluate the correlation-energy functional the source uses (its Eq. (2.22), Appendix B). Return 0.0 for an empty pair space.

In the random-phase approximation the neutral excitations of a mean-field reference follow from an eigenproblem in the particle-hole space, and the correlation energy of the reference can be written in terms of the excitation energies. The source uses one specific flavour of RPA and one specific energy functional; both are fixed by the two ingredients passed here, which describe a spin-orbital particle-hole basis.

Returns
-------
float, the RPA correlation energy of the given particle-hole space (native Python float).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rpa_correlation_energy(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> float:
    """Return the RPA correlation energy of the source's low-level method for a particle-hole space described by the vector delta_eps of orbital-energy differences (one entry per particle-hole pair) and the symmetric matrix v_ph of interaction integrals between pairs (element [I, J] = (ia|jb) for pairs I = (i, a) and J = (j, b), in the same pair order): build the RPA matrices from these two ingredients as the source's Eq. (2.8) does, with the sign of the coupling block fixed by its Appendix B, solve the source's eigenproblem, and evaluate the correlation-energy functional the source uses (its Eq. (2.22), Appendix B). Return 0.0 for an empty pair space.

    Parameters
    ----------
    delta_eps : numpy.ndarray
        One-dimensional array of length n of positive orbital-energy differences, one per particle-hole pair.
    v_ph : numpy.ndarray
        Symmetric (n, n) matrix of interaction integrals (ia|jb) between the pairs, in the order of delta_eps.

    Returns
    -------
    e_corr : float
        The RPA correlation energy in the energy units of the inputs (0.0 for n = 0).

    Raises
    ------
    ValueError
        If delta_eps has a non-positive entry, v_ph is not symmetric, or the shapes do not match.
    """
    return e_corr

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _rpa_solve(delta_eps, v_ph):
    """direct RPA in the recast form (A-B)^1/2 (A+B) (A-B)^1/2 Z = Z Omega^2 with A = diag(delta_eps) + v_ph,
    B = v_ph; returns Omega (ascending) and X+Y (columns = modes, same order), Eq. (2.9)-(2.10)."""
    d = np.asarray(delta_eps, dtype=np.float64)
    v = np.asarray(v_ph, dtype=np.float64)
    if np.any(d <= 0):
        raise ValueError("delta_eps must be positive")
    sq = np.sqrt(d)
    M = sq[:, None] * (np.diag(d) + 2.0 * v) * sq[None, :]
    M = 0.5 * (M + M.T)
    w2, Z = np.linalg.eigh(M)
    if np.any(w2 <= 0):
        raise ValueError("RPA instability: non-positive squared excitation energy")
    om = np.sqrt(w2)
    xpy = (sq[:, None] * Z) / np.sqrt(om)[None, :]          # (A-B)^1/2 Z Omega^-1/2
    return om, xpy


def _oracle_rpa_correlation_energy(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> float:
    d = _check_array(delta_eps, "delta_eps", ndim=1)
    n = d.shape[0]
    v = _check_array(v_ph, "v_ph", shape=(n, n))
    if n == 0:
        return 0.0
    if np.any(d <= 0) or not np.allclose(v, v.T, atol=1e-10):
        raise ValueError("delta_eps must be positive and v_ph symmetric")
    om, _ = _rpa_solve(d, v)
    # Klein functional, Eq. (2.22) = 1/2 Tr[Omega - A], Eq. (B.9), with A = diag(delta_eps) + v_ph
    return float(0.5 * (np.sum(om) - np.sum(d) - np.trace(v)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_eps = np.array([1.0, 1.5, 2.2])\nv_ph = np.array([[0.30, 0.10, 0.05], [0.10, 0.25, 0.08], [0.05, 0.08, 0.20]])\n",
            "call": "rpa_correlation_energy(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_correlation_energy(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nvs = np.array([[0.4, 0.15], [0.15, 0.3]])\nv_ph = np.kron(np.ones((2, 2)), vs)\ndelta_eps = np.array([1.2, 2.0, 1.2, 2.0])\n",
            "call": "rpa_correlation_energy(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_correlation_energy(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nrng = np.random.default_rng(7)\nM = rng.standard_normal((6, 6))\nv_ph = 0.1 * (M @ M.T)\ndelta_eps = np.linspace(1.0, 3.0, 6)\n",
            "call": "rpa_correlation_energy(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_correlation_energy(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ndelta_eps = np.zeros(0)\nv_ph = np.zeros((0, 0))\n",
            "call": "rpa_correlation_energy(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_correlation_energy(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ndelta_eps = np.array([1.0, -1.5, 2.2])\nv_ph = np.array([[0.30, 0.10, 0.05], [0.10, 0.25, 0.08], [0.05, 0.08, 0.20]])\ndef run_model():\n    try:\n        rpa_correlation_energy(delta_eps, v_ph)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rpa_correlation_energy(delta_eps, v_ph)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
