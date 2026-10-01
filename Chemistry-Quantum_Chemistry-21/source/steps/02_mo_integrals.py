"""
Transform the two-electron interaction of the chain of step 01 (on-site U, Ohno V_ij with parameter kappa, unit site spacing, so that in the site basis the only non-zero integrals are (ii|jj) = V_ij with V_ii = U) to the basis of the orbitals C returned by step 01 and return the full four-index tensor of two-electron integrals in chemists' notation, element [p, q, r, s] = (pq|rs) = sum_ij C[i, p] C[i, q] V_ij C[j, r] C[j, s].

Every method used by the source works with one- and two-electron integrals in the molecular-orbital basis; for a site-diagonal interaction the four-index transformation reduces to a contraction of the orbital pair densities with the site-pair repulsion matrix.

Returns
-------
numpy.ndarray of float64 with shape (n, n, n, n), n = C.shape[0], element [p, q, r, s] = (pq|rs).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mo_integrals(C: "np.ndarray", U: float, kappa: float) -> "np.ndarray":
    """Transform the two-electron interaction of the chain of step 01 (on-site U, Ohno V_ij with parameter kappa, unit site spacing, so that in the site basis the only non-zero integrals are (ii|jj) = V_ij with V_ii = U) to the basis of the orbitals C returned by step 01 and return the full four-index tensor of two-electron integrals in chemists' notation, element [p, q, r, s] = (pq|rs) = sum_ij C[i, p] C[i, q] V_ij C[j, r] C[j, s].

    Parameters
    ----------
    C : numpy.ndarray
        Square (n, n) orbital coefficient matrix, C[i, p] the coefficient of site i in orbital p.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.

    Returns
    -------
    eri : numpy.ndarray
        Array of shape (n, n, n, n), float64, with eri[p, q, r, s] = (pq|rs).

    Raises
    ------
    ValueError
        If C is not a finite square matrix or U or kappa is not positive.
    """
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_scalar(x, name, positive=False, nonneg=False):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0:
        raise ValueError("%s must be positive" % name)
    if nonneg and x < 0:
        raise ValueError("%s must be non-negative" % name)
    return x


def _check_array(a, name, shape=None, ndim=None):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("%s must be finite" % name)
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("%s must have shape %s" % (name, tuple(shape)))
    if ndim is not None and a.ndim != ndim:
        raise ValueError("%s must be %d-dimensional" % (name, ndim))
    return a


def _oracle_mo_integrals(C: "np.ndarray", U: float, kappa: float) -> "np.ndarray":
    C = _check_array(C, "C", ndim=2)
    n = C.shape[0]
    if C.shape[1] != n:
        raise ValueError("C must be square")
    U = _check_scalar(U, "U", positive=True)
    kappa = _check_scalar(kappa, "kappa", positive=True)
    idx = np.arange(n)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    # (pq|rs) = sum_ij C_ip C_iq V_ij C_jr C_js   (site-diagonal PPP interaction)
    Q = np.einsum('ip,iq->ipq', C, C)                # charge distributions of orbital pairs
    return np.einsum('ipq,ij,jrs->pqrs', Q, V, Q, optimize=True)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nC = _oracle_ppp_rhf(6, 1.0, 0.1, 3.0, 4.0)[1:]\nU, kappa = 3.0, 4.0\n",
            "call": "mo_integrals(C, U, kappa)",
            "gold_call": "_oracle_mo_integrals(C, U, kappa)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nC = _oracle_ppp_rhf(8, 1.0, 0.07, 4.0, 4.0)[1:]\nU, kappa = 4.0, 4.0\n",
            "call": "mo_integrals(C, U, kappa)",
            "gold_call": "_oracle_mo_integrals(C, U, kappa)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nC = _oracle_ppp_rhf(10, 1.0, 0.07, 4.0, 6.0)[1:]\nU, kappa = 4.0, 6.0\n",
            "call": "mo_integrals(C, U, kappa)",
            "gold_call": "_oracle_mo_integrals(C, U, kappa)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nC = np.eye(3)\nU, kappa = 2.0, 1.0\n",
            "call": "mo_integrals(C, U, kappa)",
            "gold_call": "_oracle_mo_integrals(C, U, kappa)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nC = np.ones((3, 2))\nU, kappa = 4.0, 6.0\ndef run_model():\n    try:\n        mo_integrals(C, U, kappa)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mo_integrals(C, U, kappa)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
