"""
For a particle-hole space described exactly as in step 03 (same delta_eps and v_ph, same pair order), return the n x n static kernel matrix K of the source's RPA: the sum over the modes nu of the source's eigenproblem, built from the two inputs as in step 03, of the outer product of the mode's transition-density vector (X+Y)^nu with itself divided by the mode's excitation energy Omega_nu, with the eigenvectors normalised as the source's Eq. (2.11) prescribes. K is the matrix through which the zero-frequency limit of the source's screened interaction, Eq. (2.12), is expressed once step 06 contracts it with the bare integrals and applies the pole factor of that limit. Return a (0, 0) array for an empty pair space.

The frequency-dependent screened interaction of the RPA has a pole at every neutral excitation, with a residue given by the transition density of that excitation; its static limit is therefore a single matrix in the particle-hole basis that can be contracted with bare integrals to screen any matrix element.

Returns
-------
numpy.ndarray of float64 with shape (n, n), n = len(delta_eps).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rpa_static_kernel(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> "np.ndarray":
    """For a particle-hole space described exactly as in step 03 (same delta_eps and v_ph, same pair order), return the n x n static kernel matrix K of the source's RPA: the sum over the modes nu of the source's eigenproblem, built from the two inputs as in step 03, of the outer product of the mode's transition-density vector (X+Y)^nu with itself divided by the mode's excitation energy Omega_nu, with the eigenvectors normalised as the source's Eq. (2.11) prescribes. K is the matrix through which the zero-frequency limit of the source's screened interaction, Eq. (2.12), is expressed once step 06 contracts it with the bare integrals and applies the pole factor of that limit. Return a (0, 0) array for an empty pair space.

    Parameters
    ----------
    delta_eps : numpy.ndarray
        One-dimensional array of length n of positive orbital-energy differences, one per particle-hole pair.
    v_ph : numpy.ndarray
        Symmetric (n, n) matrix of interaction integrals (ia|jb) between the pairs, in the order of delta_eps.

    Returns
    -------
    kernel : numpy.ndarray
        Symmetric (n, n) array, float64: sum over modes of (X+Y)^nu (X+Y)^nu^T / Omega_nu; shape (0, 0) for n = 0.

    Raises
    ------
    ValueError
        If delta_eps has a non-positive entry, v_ph is not symmetric, or the shapes do not match.
    """
    return kernel

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


def _oracle_rpa_static_kernel(delta_eps: "np.ndarray", v_ph: "np.ndarray") -> "np.ndarray":
    d = _check_array(delta_eps, "delta_eps", ndim=1)
    n = d.shape[0]
    v = _check_array(v_ph, "v_ph", shape=(n, n))
    if n == 0:
        return np.zeros((0, 0))
    if np.any(d <= 0) or not np.allclose(v, v.T, atol=1e-10):
        raise ValueError("delta_eps must be positive and v_ph symmetric")
    om, xpy = _rpa_solve(d, v)
    # sum_nu (X+Y)^nu (X+Y)^nu^T / Omega_nu  (= (A+B)^-1): the omega = 0 limit of the pole sum in Eq. (2.12)
    return (xpy / om[None, :]) @ xpy.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndelta_eps = np.array([1.0, 1.5, 2.2])\nv_ph = np.array([[0.30, 0.10, 0.05], [0.10, 0.25, 0.08], [0.05, 0.08, 0.20]])\n",
            "call": "rpa_static_kernel(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_static_kernel(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nvs = np.array([[0.4, 0.15], [0.15, 0.3]])\nv_ph = np.kron(np.ones((2, 2)), vs)\ndelta_eps = np.array([1.2, 2.0, 1.2, 2.0])\n",
            "call": "rpa_static_kernel(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_static_kernel(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nrng = np.random.default_rng(7)\nM = rng.standard_normal((6, 6))\nv_ph = 0.1 * (M @ M.T)\ndelta_eps = np.linspace(1.0, 3.0, 6)\n",
            "call": "rpa_static_kernel(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_static_kernel(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ndelta_eps = np.zeros(0)\nv_ph = np.zeros((0, 0))\n",
            "call": "rpa_static_kernel(delta_eps, v_ph)",
            "gold_call": "_oracle_rpa_static_kernel(delta_eps, v_ph)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\ndelta_eps = np.array([1.0, 1.5, 2.2])\nv_ph = np.array([[0.30, 0.10, 0.05], [0.20, 0.25, 0.08], [0.05, 0.08, 0.20]])\ndef run_model():\n    try:\n        rpa_static_kernel(delta_eps, v_ph)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_rpa_static_kernel(delta_eps, v_ph)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
