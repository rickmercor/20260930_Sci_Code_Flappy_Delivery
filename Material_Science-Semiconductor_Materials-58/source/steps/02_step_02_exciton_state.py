"""
Step 02 - Binding energy and wave function of the interlayer exciton.

Bound exciton of the layered two-band model: binding energy and momentum-space wave function.

In the dilute limit the condensate is built from the lowest bound state of one conduction electron and one
valence hole interacting through the interlayer Coulomb kernel V_q = 4 pi exp(-q d)/q (excitonic units, see the
previous step). That state is the lowest solution of the two-dimensional Schroedinger equation

    k^2 phi(k) - (V_d phi)(k) = -E_b phi(k),

where (V_d phi) is the exchange integral of the previous step. For a monolayer (d = 0) the solution is the
two-dimensional hydrogenic 1s state, with E_b = 4 Ry* and phi(k) = sqrt(2 pi) / (1 + k^2/4)^(3/2) in the
normalisation used here; for d > 0 the kernel is softer at short distance, the state is less bound and no closed
form exists, so the eigenproblem has to be solved numerically to the same 1e-9 relative accuracy as the exchange
integral. The wave function is normalised so that int d^2k/(2 pi)^2 phi(k)^2 = 1 and its sign is fixed by
phi(0) > 0.

Inputs: d >= 0 (a_B*); k_out, a one-dimensional array of finite non-negative magnitudes (0 allowed). Output: a
one-dimensional array [E_b, phi(k_out[0]), phi(k_out[1]), ...] of length len(k_out) + 1, E_b in Ry* and phi in
units of a_B*. Raises ValueError if d is negative or not finite or if k_out is not a one-dimensional array of
finite non-negative numbers.

Returns
-------
numpy.ndarray of shape (len(k_out) + 1,): [E_b (Ry*), phi(k_out) (a_B*)] with int d^2k/(2 pi)^2 phi^2 = 1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def exciton_state(d: float, k_out: np.ndarray) -> np.ndarray:
    '''Binding energy and normalised 1s wave function of the interlayer exciton.

    Parameters
    ----------
    d : float
        Interlayer distance in units of a_B*, >= 0.
    k_out : np.ndarray
        One-dimensional array of finite non-negative wavevector magnitudes (0 allowed) at which phi is wanted.

    Returns
    -------
    result : np.ndarray
        Shape (len(k_out) + 1,): [E_b, phi(k_out[0]), ..., phi(k_out[-1])] with E_b in Ry* and phi normalised
        to int d^2k/(2 pi)^2 phi^2 = 1 with phi(0) > 0, each converged to a relative accuracy of 1e-9.

    Raises
    ------
    ValueError
        If d is negative or not finite, or if k_out is not a one-dimensional array of finite non-negative numbers.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def _exciton(d, _EXCITONS={}):
    """Lowest eigenpair of k^2 - V_d on the trusted nodes: (E_b, phi on trusted nodes), phi normalised and positive."""
    key = round(float(d), 12)
    if key in _EXCITONS:
        return _EXCITONS[key]
    op = _operator(d)
    k = op["k"]
    A = np.diag(k * k) - op["M"]
    # the operator is symmetric under the measure up to quadrature error: start from the symmetrised problem,
    # then polish the lowest eigenpair of the actual matrix by inverse iteration with Rayleigh-quotient shifts
    sq = np.sqrt(op["mu"])
    S = A * (sq[:, None] / sq[None, :])
    vals, vecs = eigh(0.5 * (S + S.T))
    lam = vals[0]
    phi = vecs[:, 0] / sq
    for _ in range(3):
        v = np.linalg.solve(A - lam * np.eye(len(k)), phi)
        v /= np.sqrt(np.sum(v * v))
        lam = (v @ (A @ v)) / (v @ v)
        phi = v
    phi = phi / np.sqrt(np.sum(op["mu"] * phi * phi))
    if np.sum(phi[k < 0.5]) < 0:
        phi = -phi
    _EXCITONS[key] = (-lam, phi)
    return _EXCITONS[key]


def _full_values(op, fT):
    """Values of a trusted-node vector on the full grid, the innermost panel filled by the operator's extrapolation."""
    return np.concatenate([op["L"] @ fT, fT])


def _oracle_exciton_state(d: float, k_out: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    d = _check_scalar(d, "d", nonneg=True)
    k_out = _check_k_array(k_out, "k_out")
    Eb, phi = _exciton(d)
    op = _operator(d)
    # Nystrom interpolation: phi(k) = (V_d phi)(k) / (k^2 + E_b)
    vphi = _apply_to_grid_values(_full_values(op, phi), k_out, d)
    return np.concatenate([[Eb], vphi / (k_out ** 2 + Eb)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer, must reproduce E_b = 4 Ry* and the 2D hydrogenic 1s function ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 0.3, 1.0, 2.0, 4.0, 9.0])\n"
                     "ref = np.concatenate([[4.0], np.sqrt(2*np.pi)/(1 + k_out**2/4)**1.5])\n",
            "call": "exciton_state(0.0, k_out.copy()) / ref",
            "gold_call": "_oracle_exciton_state(0.0, k_out.copy()) / ref",
            "tol": 1e-6,
        },
        # --- Normal: heterobilayer at a quarter Bohr radius ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 0.5, 1.5, 3.0, 8.0])\n",
            "call": "exciton_state(0.25, k_out.copy())",
            "gold_call": "_oracle_exciton_state(0.25, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Boundary: half a Bohr radius, wave function requested at the singular scale of the kernel ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.002, 0.07, 0.9, 2.2, 20.0])\n",
            "call": "exciton_state(0.5, k_out.copy())",
            "gold_call": "_oracle_exciton_state(0.5, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Edge: a full Bohr radius of separation, weakly bound state ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "k_out = np.array([0.0, 1.0, 3.5])\n",
            "call": "exciton_state(1.0, k_out.copy())",
            "gold_call": "_oracle_exciton_state(1.0, k_out.copy())",
            "tol": 1e-6,
        },
        # --- Invalid: a negative wavevector in k_out must raise ValueError (0 returned, 1 raised) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(0.25, np.array([0.5, -1.0]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(exciton_state)",
            "gold_call": "_probe(_oracle_exciton_state)",
        },
    ]
