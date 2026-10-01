"""
Construct the ordered Pauli decomposition of an open transverse-field Ising chain.

The open-chain Hamiltonian contains nearest-neighbour ZZ interactions and single-site transverse X fields.  Encoding these operators as integer Pauli strings supplies the deterministic generator order used by all later imaginary-time updates.

Returns
-------
tuple[np.ndarray, np.ndarray] containing an integer code array of shape (2*n_qubits - 1, n_qubits) and an aligned float coefficient array of shape (2*n_qubits - 1,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real

import numpy as np


def build_open_tfim_pauli_terms(
    n_qubits: int,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the ordered open-chain transverse-field Ising Hamiltonian.

    The Pauli encoding is 0=I, 1=X, 2=Y, and 3=Z.  Bond
    strings are returned first in increasing left-site order, followed by
    single-site field strings in increasing site order.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Must be a positive integer and cannot be bool.
    coupling : float
        Finite real nearest-neighbour coupling J in
        -J * sum_i Z_i Z_{i+1}.
    field : float
        Finite real transverse-field strength h in
        -h * sum_i X_i.

    Returns
    -------
    pauli_codes : np.ndarray
        Integer array of shape (2*n_qubits - 1, n_qubits) containing the
        ordered Pauli strings.
    coefficients : np.ndarray
        Float array of shape (2*n_qubits - 1,) containing the Hamiltonian
        coefficients in the same order.

    Raises
    ------
    ValueError
        If n_qubits is not a positive integer, or if either scalar is not
        a finite real number.
    """
    return (
        np.empty((2 * int(n_qubits) - 1, int(n_qubits)), dtype=np.int8),
        np.empty(2 * int(n_qubits) - 1, dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_open_tfim_pauli_terms(
    n_qubits: int,
    coupling: float,
    field: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation."""
    from numbers import Integral, Real

    import numpy as np

    if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral):
        raise ValueError("n_qubits must be a positive integer")
    n = int(n_qubits)
    if n < 1:
        raise ValueError("n_qubits must be a positive integer")
    if isinstance(coupling, bool) or not isinstance(coupling, Real):
        raise ValueError("coupling must be a finite real scalar")
    if isinstance(field, bool) or not isinstance(field, Real):
        raise ValueError("field must be a finite real scalar")
    j = float(coupling)
    h = float(field)
    if not np.isfinite(j) or not np.isfinite(h):
        raise ValueError("coupling and field must be finite")

    codes = np.zeros((2 * n - 1, n), dtype=np.int8)
    coeffs = np.empty(2 * n - 1, dtype=float)

    row = 0
    for site in range(n - 1):
        codes[row, site] = 3
        codes[row, site + 1] = 3
        coeffs[row] = -j
        row += 1
    for site in range(n):
        codes[row, site] = 1
        coeffs[row] = -h
        row += 1

    return codes, coeffs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return a list of test case specifications."""
    return [
        # Case 1: normal open chain.
        {
            "setup": (
                "import numpy as np\n"
                "n_qubits = 4\n"
                "coupling = 1.0\n"
                "field = 0.73\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(build_open_tfim_pauli_terms(n_qubits, coupling, field))",
            "gold_call": "_pack_pair(_oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field))",
        },
        # Case 2: single-qubit boundary.
        {
            "setup": (
                "import numpy as np\n"
                "n_qubits = 1\n"
                "coupling = 2.0\n"
                "field = -0.25\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(build_open_tfim_pauli_terms(n_qubits, coupling, field))",
            "gold_call": "_pack_pair(_oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field))",
        },
        # Case 3: zero-coupling/field edge.
        {
            "setup": (
                "import numpy as np\n"
                "n_qubits = 3\n"
                "coupling = 0.0\n"
                "field = 0.0\n"
                "def _pack_pair(result):\n"
                "    codes, coeffs = result\n"
                "    codes = np.asarray(codes, dtype=float)\n"
                "    coeffs = np.asarray(coeffs, dtype=float)\n"
                "    header = np.asarray(\n"
                "        [codes.ndim, *codes.shape, coeffs.ndim, *coeffs.shape],\n"
                "        dtype=float,\n"
                "    )\n"
                "    return np.concatenate((header, codes.ravel(), coeffs.ravel()))"
            ),
            "call": "_pack_pair(build_open_tfim_pauli_terms(n_qubits, coupling, field))",
            "gold_call": "_pack_pair(_oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field))",
        },
        # Case 4: invalid qubit count.
        {
            "setup": (
                "n_qubits = 0\n"
                "coupling = 1.0\n"
                "field = 0.5\n"
                "def run_model():\n"
                "    try:\n"
                "        build_open_tfim_pauli_terms(n_qubits, coupling, field)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Case 5: non-finite scalar.
        {
            "setup": (
                "import numpy as np\n"
                "n_qubits = 2\n"
                "coupling = np.inf\n"
                "field = 0.5\n"
                "def run_model():\n"
                "    try:\n"
                "        build_open_tfim_pauli_terms(n_qubits, coupling, field)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_build_open_tfim_pauli_terms(n_qubits, coupling, field)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
