"""
Compute bath-weighted operator moments through the requested order in the normalized Pauli basis (I, X, Y, Z)/sqrt(2). Use system-first tensor-product ordering.

The bath state is rho_b = exp(-beta*Hb)/Tr[exp(-beta*Hb)]. The moment matrix is Omega[n,i,j] = Tr[D^n(phi_i tensor I_b)(phi_j tensor rho_b)], where D(A) = i[H,A]. The thermal state uses Hb; the commutator uses the total interacting Hamiltonian H.

Returns
-------
A real NumPy array of shape (order + 1, 4, 4), containing Ω₀ through Ωorder.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_moments(h, hb, beta, order):
    """Compute bath-weighted operator moments.

    Parameters
    ----------
    h : array_like, complex, shape (2*b, 2*b)
        Hermitian total Hamiltonian, system-first tensor ordering.
    hb : array_like, complex, shape (b, b)
        Hermitian bath Hamiltonian; b >= 1.
    beta : float
        Finite nonnegative inverse temperature.
    order : int
        Highest moment order, from 0 through 12.

    Returns
    -------
    ndarray, real, shape (order + 1, 4, 4)
        Moments in basis (I, X, Y, Z)/sqrt(2), with
        Omega[n,i,j] = Tr[D**n(phi_i tensor I)(phi_j tensor rho_b)]
        and D(A) = 1j*(h@A - A@h).

    Raises
    ------
    ValueError
        If shapes are incompatible, a Hamiltonian is nonfinite or
        non-Hermitian at absolute tolerance 1e-12, beta is nonfinite
        or negative, or order is not an integer in [0, 12].
    """
    return np.zeros((order + 1, 4, 4))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_moments(h, hb, beta, order):
    import numpy as np

    h = np.asarray(h, dtype=complex)
    hb = np.asarray(hb, dtype=complex)

    if (
        hb.ndim != 2
        or hb.shape[0] != hb.shape[1]
        or hb.shape[0] == 0
        or h.shape != (2 * hb.shape[0], 2 * hb.shape[0])
        or not np.isfinite(h).all()
        or not np.isfinite(hb).all()
        or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0)
        or not np.allclose(hb, hb.conj().T, atol=1e-12, rtol=0)
        or not np.isfinite(beta)
        or beta < 0
        or not isinstance(order, (int, np.integer))
        or not 0 <= order <= 12
    ):
        raise ValueError(
            "Invalid Hamiltonian, temperature, or moment order."
        )

    energy, vectors = np.linalg.eigh(hb)
    weights = np.exp(-beta * (energy - energy.min()))
    weights = weights / weights.sum()
    rho = (vectors * weights) @ vectors.conj().T

    pauli = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    ) / np.sqrt(2.0)

    bath_identity = np.eye(hb.shape[0])

    operators = np.array(
        [np.kron(a, bath_identity) for a in pauli]
    )
    dual = np.array(
        [np.kron(a, rho) for a in pauli]
    )

    result = np.empty((order + 1, 4, 4), dtype=float)

    for n in range(order + 1):
        result[n] = np.einsum(
            "iab,jba->ij", operators, dual
        ).real

        operators = 1j * (
            h @ operators - operators @ h
        )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
x = np.array([[0., 1.], [1., 0.]])
z = np.diag([1., -1.])
hb = np.diag([0., .7, 1.3])
bx = np.array([[0., .23, .11], [.23, 0., .17], [.11, .17, 0.]])
bz = np.array([[0., .13, -.19], [.13, 0., .07], [-.19, .07, 0.]])
h = np.kron(np.eye(2), hb) + np.kron(x, bx) + np.kron(z, bz)
""",
            "call": "compute_moments(h, hb, 1.4, 12)",
            "gold_call": "_oracle_compute_moments(h, hb, 1.4, 12)",
        },
        {
            "setup": """import numpy as np
h = np.zeros((2, 2))
hb = np.zeros((1, 1))
""",
            "call": "compute_moments(h, hb, 0., 0)",
            "gold_call": "_oracle_compute_moments(h, hb, 0., 0)",
        },
        {
            "setup": """import numpy as np
hb = np.array([[0., .3j], [-.3j, .8]])
h = np.kron(np.eye(2), hb)
""",
            "call": "compute_moments(h, hb, 80., 4)",
            "gold_call": "_oracle_compute_moments(h, hb, 80., 4)",
        },
    ]
