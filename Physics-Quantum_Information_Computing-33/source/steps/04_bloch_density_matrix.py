"""
Single-qubit density matrix from a Bloch vector.

Every single-qubit state is rho = (I + r_x X + r_y Y + r_z Z)/2 with a real Bloch vector r of Euclidean length at most 1; length 1 is a pure state, length below 1 a mixed state. The Pauli coefficients of rho are exactly the entries of its signed Pauli spectrum after the identity, Tr(rho sigma_i) = r_i, so the Bloch vector and the Pauli spectrum carry the same information.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bloch_density_matrix(r: np.ndarray) -> np.ndarray:
    """Single-qubit density matrix with Bloch vector ``r``.

    Return rho = (I + r_x X + r_y Y + r_z Z)/2 using the phase-free
    Paulis of ``single_qubit_pauli``. Supported domain: ``r`` is a real
    array of shape (3,) with finite entries and Euclidean norm at most
    1 (a tolerance of 1e-9 above 1 is accepted). Raise ValueError
    otherwise.

    Parameters
    ----------
    r : np.ndarray
        Real Bloch vector of shape (3,).

    Returns
    -------
    rho : np.ndarray
        Complex Hermitian array of shape (2, 2) with unit trace.
    """
    return rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bloch_density_matrix(r: np.ndarray) -> np.ndarray:
    """Reference oracle: rho = (I + r . sigma)/2 built from step 01."""
    import numpy as np

    vec = np.asarray(r, dtype=float).reshape(-1)
    if vec.size != 3 or not np.all(np.isfinite(vec)):
        raise ValueError("r must be a finite real vector of shape (3,).")
    if float(np.linalg.norm(vec)) > 1.0 + 1e-9:
        raise ValueError("Bloch vector must have Euclidean norm at most 1.")

    # Step 01 is available in the concatenated Studio namespace; the inline table is a
    # fallback only for isolated execution of this step.
    _pauli = globals().get("_oracle_single_qubit_pauli")
    if not callable(_pauli):
        tables = (
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
            np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
            np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
        )

        def _pauli(kind: int) -> np.ndarray:
            return tables[int(kind)].copy()

    rho = _pauli(0).astype(complex)
    for i in range(3):
        rho = rho + vec[i] * _pauli(i + 1)
    return 0.5 * rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = (
        "[np.round(np.asarray({fn}(r)).real, 12).tolist(), "
        "np.round(np.asarray({fn}(r)).imag, 12).tolist()]"
    )
    return [
        {
            # Maximally mixed state.
            "setup": "import numpy as np\nr = np.array([0.0, 0.0, 0.0])\n",
            "call": pair.format(fn="bloch_density_matrix"),
            "gold_call": pair.format(fn="_oracle_bloch_density_matrix"),
        },
        {
            # Pure stabilizer state |0>.
            "setup": "import numpy as np\nr = np.array([0.0, 0.0, 1.0])\n",
            "call": pair.format(fn="bloch_density_matrix"),
            "gold_call": pair.format(fn="_oracle_bloch_density_matrix"),
        },
        {
            # Pure T state (I + (X + Y)/sqrt 2)/2.
            "setup": "import numpy as np\nr = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)\n",
            "call": pair.format(fn="bloch_density_matrix"),
            "gold_call": pair.format(fn="_oracle_bloch_density_matrix"),
        },
        {
            # Mixed magic qubit of the task.
            "setup": "import numpy as np\nr = np.array([0.6, 0.5, 0.3])\n",
            "call": pair.format(fn="bloch_density_matrix"),
            "gold_call": pair.format(fn="_oracle_bloch_density_matrix"),
        },
        {
            # Mixed state with negative components.
            "setup": "import numpy as np\nr = np.array([-0.2, 0.7, -0.4])\n",
            "call": pair.format(fn="bloch_density_matrix"),
            "gold_call": pair.format(fn="_oracle_bloch_density_matrix"),
        },
    ]
