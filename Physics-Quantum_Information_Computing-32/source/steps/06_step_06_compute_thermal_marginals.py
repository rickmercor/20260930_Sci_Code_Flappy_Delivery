"""
Compute exact Pauli expectations in a finite thermal quantum state.

For the dimensionless Hamiltonian $h=H/E_*=\sum_jc_jP_j$, the Gibbs state is $\rho=e^{-bh}/\operatorname{tr}e^{-bh}$ with $b=\beta E_*\ge0$. Diagonalize the Hermitian $h=U\operatorname{diag}(E)U^\dagger$ and evaluate

$$

w_k=\frac{e^{-b(E_k-E_{\min})}}{\sum_\ell e^{-b(E_\ell-E_{\min})}},\qquad y_j=\sum_kw_k(U^\dagger P_jU)_{kk}.

$$

The energy shift cancels in normalization and prevents overflow. This expression is independent of the choice of basis in degenerate eigenspaces. At $b=0$, the state is maximally mixed. Tensor factors follow the label order, with the leftmost factor corresponding to the most significant computational basis bit.

Returns
-------
Real dimensionless array $(m,)$ with entries $\operatorname{tr}(\rho P_j)$ in label order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_thermal_marginals(
    labels: tuple[str, ...], coefficients: "np.ndarray", inverse_temperature: float
) -> "np.ndarray":
    r"""Compute exact Pauli expectations in a finite thermal quantum state.

    Parameters
    ----------
    labels : tuple[str, ...]
        Distinct nonidentity unsigned Pauli words on $1\le n\le6$ qubits.
    coefficients : np.ndarray
        Finite real vector $(m,)$ of dimensionless Hamiltonian coefficients in label order.
    inverse_temperature : float
        Finite dimensionless inverse temperature $b\ge0$.

    Returns
    -------
    expectations : np.ndarray
        Real dimensionless array $(m,)$ with entries $\operatorname{tr}(\rho P_j)$ in label order.

    Raises
    ------
    ValueError
        If labels are invalid, coefficients have the wrong shape or are complex or nonfinite, or inverse temperature is negative or nonfinite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pauli_matrices(labels):
    single = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    matrices = []
    for word in labels:
        value = np.ones((1, 1), dtype=complex)
        for letter in word:
            value = np.kron(value, single[letter])
        matrices.append(value)
    return np.array(matrices)


def _oracle_compute_thermal_marginals(
    labels: tuple[str, ...], coefficients: "np.ndarray", inverse_temperature: float
) -> "np.ndarray":
    _oracle_encode_pauli_words(labels)
    raw = np.asarray(coefficients)
    if raw.shape != (len(labels),) or np.iscomplexobj(raw):
        raise ValueError("coefficients must be a real vector in label order")
    coefficients = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(coefficients))
        or not np.isfinite(inverse_temperature)
        or inverse_temperature < 0
    ):
        raise ValueError(
            "coefficients and nonnegative temperature parameter must be finite"
        )
    operators = _pauli_matrices(labels)
    hamiltonian = np.einsum("j,jab->ab", coefficients, operators)
    energies, vectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-float(inverse_temperature) * (energies - energies[0]))
    weights /= weights.sum()
    diagonals = np.einsum("ak,jab,bk->jk", vectors.conj(), operators, vectors)
    return np.real(diagonals @ weights)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
labels = ('X','Y','Z')
coefficients = np.array([-.7,.6,-.9])
b = 2.3
""",
            "call": "compute_thermal_marginals(labels, coefficients.copy(), b)",
            "gold_call": "_oracle_compute_thermal_marginals(labels, coefficients.copy(), b)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
labels = ('XI','YY','ZZ')
coefficients = np.array([.2,-.5,.9])
b = 0.0
""",
            "call": "compute_thermal_marginals(labels, coefficients.copy(), b)",
            "gold_call": "_oracle_compute_thermal_marginals(labels, coefficients.copy(), b)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
labels = ('ZI','IZ','ZZ')
coefficients = np.array([0.,0.,-1.])
b = 800.0
""",
            "call": "compute_thermal_marginals(labels, coefficients.copy(), b)",
            "gold_call": "_oracle_compute_thermal_marginals(labels, coefficients.copy(), b)",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np
labels = ('X',)
coefficients = np.array([1.])
b = -1.0

def _raises_value_error(fn):
    try:
        fn(labels, coefficients.copy(), b)
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(compute_thermal_marginals)",
            "gold_call": "_raises_value_error(_oracle_compute_thermal_marginals)",
            "tol": 0.0,
        },
    ]
