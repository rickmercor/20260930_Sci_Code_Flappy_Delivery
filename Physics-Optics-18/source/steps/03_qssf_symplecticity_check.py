"""
Compute both normalized Frobenius residuals of the bosonic canonical relations. Return diagnostic errors for canonical or noncanonical finite matrix pairs without imposing a physicality threshold in this diagnostic routine.

Evaluate the preservation of canonical bosonic commutation relations and



symplectic invariants for the quantum Bogoliubov matrices U and V.







The canonical commutation relations [a_j(z), a_k^dagger(z)] = delta_jk and



[a_j(z), a_k(z)] = 0 impose exact symplectic identities on the Bogoliubov



transformation:



    U * U^dagger - V * V^dagger = I



    U * V^T - V * U^T = 0







Numerical drift and symplecticity loss are quantified by the normalized



Frobenius norms:



    epsilon_1 = (1 / Nt) * || U * U^dagger - V * V^dagger - I ||_F



    epsilon_2 = (1 / Nt) * || U * V^T - V * U^T ||_F







At the benchmark, both errors are expected below 1e-11. This function



reports errors without imposing that bound: noncanonical matrices are valid



inputs. U,V must be finite matching nonempty square matrices; invalid data



raises ValueError. No caller inputs are modified.







Parameters



----------



U : np.ndarray



    Two-dimensional complex array of shape (Nt, Nt) representing the normal



    Bogoliubov transformation matrix.



V : np.ndarray



    Two-dimensional complex array of shape (Nt, Nt) representing the



    anomalous Bogoliubov transformation matrix.







Returns



-------



np.ndarray



    One-dimensional float array of shape (2,) containing:



    - Index 0: Normalized Frobenius residual epsilon_1 of the first identity.



    - Index 1: Normalized Frobenius residual epsilon_2 of the second identity.

Returns
-------
float ndarray (2,), normalized commutator errors err1, err2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_symplecticity_check(U: "np.ndarray", V: "np.ndarray") -> "np.ndarray":
    """
    Calculate the normalized Frobenius errors for symplectic commutator
    relations.

    Parameters
    ----------
    U : "np.ndarray"
        Normal Bogoliubov matrix, shape (Nt, Nt).
    V : "np.ndarray"
        Anomalous Bogoliubov matrix, shape (Nt, Nt).

    Returns
    -------
    np.ndarray
        One-dimensional float array of shape (2,) containing [err1, err2].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qssf_symplecticity_check(
    U: "np.ndarray", V: "np.ndarray"
) -> "np.ndarray":
    U, V = _qssf_pair(U, V)
    Nt = U.shape[0]
    identity = np.eye(Nt, dtype=complex)
    comm1 = U @ U.conj().T - V @ V.conj().T - identity
    comm2 = U @ V.T - V @ U.T
    err1 = float(np.linalg.norm(comm1) / Nt)
    err2 = float(np.linalg.norm(comm2) / Nt)
    return np.array([err1, err2], dtype=float)

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
                "\n"
                "def invalid(fn):\n"
                "    try:\n"
                "        fn(np.zeros((0, 0)), np.zeros((0, 0)))\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": ("invalid(qssf_symplecticity_check)"),
            "gold_call": ("invalid(_oracle_qssf_symplecticity_check)"),
            "tol": 0.0,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 16\n"
                "U = np.eye(Nt, dtype=complex)\n"
                "V = np.zeros((Nt, Nt), dtype=complex)\n"
            ),
            "call": ("qssf_symplecticity_check(U, V)"),
            "gold_call": ("_oracle_qssf_symplecticity_check(U, V)"),
            "tol": 1e-12,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 16\n"
                "U = 1.2 * np.eye(Nt, dtype=complex)\n"
                "V = 0.1 * np.eye(Nt, dtype=complex)\n"
            ),
            "call": ("qssf_symplecticity_check(U, V)"),
            "gold_call": ("_oracle_qssf_symplecticity_check(U, V)"),
            "tol": 1e-12,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 16\n"
                "rng = np.random.default_rng(42)\n"
                "U = rng.standard_normal(\n"
                "    (Nt, Nt)\n"
                ") + 1j * rng.standard_normal((Nt, Nt))\n"
                "V = 0.5 * (\n"
                "    rng.standard_normal((Nt, Nt))\n"
                "    + 1j * rng.standard_normal((Nt, Nt))\n"
                ")\n"
            ),
            "call": ("qssf_symplecticity_check(U, V)"),
            "gold_call": ("_oracle_qssf_symplecticity_check(U, V)"),
            "tol": 1e-12,
        },
    ]
