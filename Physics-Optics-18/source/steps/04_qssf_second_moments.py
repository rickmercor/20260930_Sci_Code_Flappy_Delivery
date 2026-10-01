"""
Select the requested output modes while retaining every input vacuum channel. Construct their normal and anomalous moments and the real grouped-quadrature covariance, validating the canonical map and the window indices.

Extract the reduced Gaussian state second-order statistical moments and



construct the continuous-variable covariance matrix on a designated spectral



window.







Given the global Bogoliubov transformation matrices U and V of dimension



(Nt, Nt) and a subset of frequency mode indices W of size n, the windowed



submatrices are:



    U_W = U[W, :],    V_W = V[W, :]







The normal and anomalous second-order correlation matrices on the subspace W



are:



    N_W = V_W^* * V_W^T



    M_W = U_W * V_W^T



where N_W is Hermitian positive semidefinite and M_W is symmetric.







From the quadrature operators q_k = (a_k + a_k^dagger) / sqrt(2) and



p_k = -i * (a_k - a_k^dagger) / sqrt(2), the 2n x 2n real symmetric



continuous-variable covariance matrix V_q is assembled in standard block



form:



    V_q = [[B + Re(M_W),   Im(M_W) + Im(N_W)],



           [Im(M_W) - Im(N_W), B - Re(M_W)]]



where:



    B = Re(N_W) + 0.5 * I_n







For the vacuum state (U = I, V = 0), N_W = 0, M_W = 0, and V_q = 0.5 *



I_{2n}, saturating the Robertson-Schrödinger uncertainty relation V_q +



(i/2) * Omega >= 0.







U,V must be finite matching nonempty square matrices satisfying both



normalized canonical commutator errors <= 1e-9. The window is a



one-dimensional integer array of unique indices in [0, Nt); preserve its



order. An empty integer window returns shape (0, 0). Invalid data raises



ValueError. Inputs are not modified. Remove only floating-point asymmetry by



averaging the assembled matrix with its transpose.







Parameters



----------



U : np.ndarray



    Two-dimensional complex array of shape (Nt, Nt) representing the normal



    Bogoliubov matrix.



V : np.ndarray



    Two-dimensional complex array of shape (Nt, Nt) representing the



    anomalous Bogoliubov matrix.



window : np.ndarray



    One-dimensional integer array of mode indices corresponding to the



    spectral window W.







Returns



-------



np.ndarray



    Two-dimensional real symmetric array of shape (2n, 2n), where n =



    len(window), representing the continuous-variable Gaussian covariance



    matrix V_q.

Returns
-------
real symmetric float ndarray (2*n, 2*n), n=len(window).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_second_moments(
    U: "np.ndarray", V: "np.ndarray", window: "np.ndarray"
) -> "np.ndarray":
    """
    Calculate the second-order moments and assemble the 2n x 2n covariance
    matrix V_q.

    Parameters
    ----------
    U : "np.ndarray"
        Normal Bogoliubov matrix, shape (Nt, Nt).
    V : "np.ndarray"
        Anomalous Bogoliubov matrix, shape (Nt, Nt).
    window : "np.ndarray"
        Integer array of selected frequency mode indices, shape (n,).

    Returns
    -------
    np.ndarray
        Real symmetric covariance matrix V_q of shape (2n, 2n).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qssf_second_moments(
    U: "np.ndarray", V: "np.ndarray", window: "np.ndarray"
) -> "np.ndarray":
    U, V = _qssf_pair(U, V)
    window = np.asarray(window)
    if window.ndim != 1 or window.dtype.kind not in "iu":
        raise ValueError("window must be a one-dimensional integer array")
    if (
        np.any(window < 0)
        or np.any(window >= U.shape[0])
        or len(np.unique(window)) != len(window)
    ):
        raise ValueError("window indices must be unique and in range")
    errors = _oracle_qssf_symplecticity_check(U, V)
    if np.max(errors) > 1e-9:
        raise ValueError("U and V must define a canonical map")
    Uw = U[window, :]
    Vw = V[window, :]
    Nw = np.conj(Vw) @ Vw.T
    Mw = Uw @ Vw.T
    n = len(window)
    B = np.real(Nw) + 0.5 * np.eye(n)
    Vq = np.block(
        [
            [B + np.real(Mw), np.imag(Mw) + np.imag(Nw)],
            [np.imag(Mw) - np.imag(Nw), B - np.real(Mw)],
        ]
    )
    return (Vq + Vq.T) / 2.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """
    Return normal, boundary and edge-case specifications.
    """
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "U = np.eye(2, dtype=complex)\n"
                "V = np.zeros((2, 2), dtype=complex)\n"
                "window = np.array([], dtype=int)\n"
            ),
            "call": ("qssf_second_moments(U.copy(), V.copy(), window.copy())"),
            "gold_call": (
                "_oracle_qssf_second_moments(\n"
                "    U.copy(), V.copy(), window.copy()\n"
                ")"
            ),
            "tol": 1e-12,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def invalid(fn):\n"
                "    try:\n"
                "        fn(\n"
                "            np.eye(2),\n"
                "            np.zeros((2, 2)),\n"
                "            np.array([0, 0]),\n"
                "        )\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": ("invalid(qssf_second_moments)"),
            "gold_call": ("invalid(_oracle_qssf_second_moments)"),
            "tol": 0.0,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 64\n"
                "window = np.arange(10, 20)\n"
                "U = np.eye(Nt, dtype=complex)\n"
                "V = np.zeros((Nt, Nt), dtype=complex)\n"
            ),
            "call": ("qssf_second_moments(U, V, window)"),
            "gold_call": ("_oracle_qssf_second_moments(U, V, window)"),
            "tol": 1e-12,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 32\n"
                "window = np.array([5, 6, 7, 8, 9])\n"
                "theta = 0.3\n"
                "U = np.cosh(theta) * np.eye(Nt, dtype=complex)\n"
                "V = np.sinh(theta) * np.eye(Nt, dtype=complex)\n"
            ),
            "call": ("qssf_second_moments(U, V, window)"),
            "gold_call": ("_oracle_qssf_second_moments(U, V, window)"),
            "tol": 1e-12,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 16\n"
                "window = np.array([2, 4, 6])\n"
                "rng = np.random.default_rng(777)\n"
                "X = rng.standard_normal(\n"
                "    (Nt, Nt)\n"
                ") + 1j * rng.standard_normal((Nt, Nt))\n"
                "Q, _ = np.linalg.qr(X)\n"
                "r = np.linspace(0.05, 0.45, Nt)\n"
                "U = Q * np.cosh(r)\n"
                "V = Q * np.sinh(r)\n"
            ),
            "call": ("qssf_second_moments(U, V, window)"),
            "gold_call": ("_oracle_qssf_second_moments(U, V, window)"),
            "tol": 1e-12,
        },
    ]
