"""
Build the finite many-body representation of a two-component impurity coupled to matrix-valued bath sites.

Finite impurity models provide controlled real-frequency benchmarks for correlated fermions.  When the local and bath one-body matrices do not commute, orbital coherences and complex phases remain active throughout the many-body calculation instead of reducing to independent scalar channels.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], the full Hamiltonian, interaction Hamiltonian, annihilation operators, and occupation labels
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_noncommuting_impurity_model(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return the Hamiltonians, annihilators, and particle-number labels.

    Parameters
    ----------
    impurity_energy : np.ndarray
        Finite Hermitian array of shape ``(2, 2)``.
    bath_energies : np.ndarray
        Finite Hermitian bath blocks of shape ``(M, 2, 2)``; ``M`` may be zero.
    bath_couplings : np.ndarray
        Finite impurity-to-bath blocks of shape ``(M, 2, 2)``.  With impurity
        column ``a0`` and bath column ``ai``, block ``i`` contributes
        ``a0^H bath_couplings[i]^H ai + ai^H bath_couplings[i] a0``.
    interaction_u : float
        Finite real coefficient of the impurity interaction ``n_0 n_1``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The full Hamiltonian and interaction Hamiltonian, each of shape
        ``(2**L, 2**L)``, all annihilation matrices with shape
        ``(L, 2**L, 2**L)``, and the integer particle number of every basis
        state with shape ``(2**L,)``.  Here ``L = 2*(M+1)``.  The occupation-bit
        basis uses ``(state >> p) & 1`` for mode ``p``; impurity modes are 0 and
        1, bath modes are site-major, and fermionic signs count occupied modes
        with lower indices.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, or the reality of ``interaction_u``
        violate the contract.

    Notes
    -----
    Input arrays are not required to be preserved by implementations.
    """
    return hamiltonian, interaction_hamiltonian, annihilators, occupations

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_noncommuting_impurity_model(
    impurity_energy: "np.ndarray",
    bath_energies: "np.ndarray",
    bath_couplings: "np.ndarray",
    interaction_u: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    impurity_energy = np.asarray(impurity_energy, dtype=np.complex128)
    bath_energies = np.asarray(bath_energies, dtype=np.complex128)
    bath_couplings = np.asarray(bath_couplings, dtype=np.complex128)
    if impurity_energy.shape != (2, 2):
        raise ValueError("impurity_energy must have shape (2,2)")
    if bath_energies.ndim != 3 or bath_energies.shape[1:] != (2, 2):
        raise ValueError("bath_energies must have shape (M,2,2)")
    if bath_couplings.shape != bath_energies.shape:
        raise ValueError("bath_couplings must match bath_energies")
    if any(not np.all(np.isfinite(x)) for x in
           (impurity_energy, bath_energies, bath_couplings)):
        raise ValueError("all one-body arrays must be finite")
    if not np.allclose(impurity_energy, impurity_energy.conj().T,
                       rtol=0.0, atol=2e-12):
        raise ValueError("impurity_energy must be Hermitian")
    if any(not np.allclose(block, block.conj().T, rtol=0.0, atol=2e-12)
           for block in bath_energies):
        raise ValueError("each bath-energy block must be Hermitian")
    if (not np.isscalar(interaction_u)
            or not np.isfinite(interaction_u)
            or abs(complex(interaction_u).imag) > 0.0):
        raise ValueError("interaction_u must be finite and real")

    site_count = int(bath_energies.shape[0])
    mode_count = 2 * (site_count + 1)
    dimension = 1 << mode_count
    one_body = np.zeros((mode_count, mode_count), dtype=np.complex128)
    one_body[:2, :2] = impurity_energy
    for site in range(site_count):
        block = slice(2 * (site + 1), 2 * (site + 2))
        one_body[block, block] = bath_energies[site]
        one_body[:2, block] = bath_couplings[site].conj().T
        one_body[block, :2] = bath_couplings[site]

    annihilators = np.zeros(
        (mode_count, dimension, dimension), dtype=np.complex128
    )
    for mode in range(mode_count):
        lower_mask = (1 << mode) - 1
        for state in range(dimension):
            if (state >> mode) & 1:
                target = state ^ (1 << mode)
                parity = (state & lower_mask).bit_count() & 1
                annihilators[mode, target, state] = -1.0 if parity else 1.0

    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for p in range(mode_count):
        creation = annihilators[p].conj().T
        for q in range(mode_count):
            if one_body[p, q] != 0.0:
                hamiltonian += one_body[p, q] * creation @ annihilators[q]
    number_0 = annihilators[0].conj().T @ annihilators[0]
    number_1 = annihilators[1].conj().T @ annihilators[1]
    interaction_hamiltonian = float(np.real(interaction_u)) * number_0 @ number_1
    hamiltonian += interaction_hamiltonian
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    occupations = np.asarray(
        [state.bit_count() for state in range(dimension)], dtype=np.int64
    )
    return hamiltonian, interaction_hamiltonian, annihilators, occupations

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid model-construction cases."""
    common = """import numpy as np
def pack(fn, impurity, bath, coupling, interaction):
    h, hint, annihilators, occupations = fn(
        impurity.copy(), bath.copy(), coupling.copy(), interaction
    )
    modes = 2 * (bath.shape[0] + 1)
    dimension = 1 << modes
    if h.shape != (dimension, dimension):
        raise AssertionError("full Hamiltonian shape is incorrect")
    if hint.shape != h.shape or annihilators.shape != (modes, dimension, dimension):
        raise AssertionError("operator shapes are incorrect")
    if occupations.shape != (dimension,):
        raise AssertionError("occupation-label shape is incorrect")
    return np.concatenate((h.ravel(), hint.ravel(), annihilators.ravel(), occupations.astype(complex)))
"""
    return [
        {
            "setup": common + """
impurity = np.array([[-0.973779, 0.112032-0.097339j],
                     [0.112032+0.097339j, -1.300553]], complex)
bath = np.array([[[-1.084493, -0.148375-0.048758j],
                  [-0.148375+0.048758j, 1.029384]]], complex)
coupling = np.array([[[-0.049439-0.310296j, -0.118014+0.213365j],
                      [0.009953+0.122430j, -0.019628+0.285367j]]], complex)
interaction = 2.367874
""",
            "call": "pack(build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "gold_call": "pack(_oracle_build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "tol": 2e-12,
        },
        {
            "setup": common + """
impurity = np.array([[-0.4, 0.07j], [-0.07j, 0.2]], complex)
bath = np.empty((0, 2, 2), complex)
coupling = np.empty((0, 2, 2), complex)
interaction = 0.0
""",
            "call": "pack(build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "gold_call": "pack(_oracle_build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "tol": 2e-12,
        },
        {
            "setup": common + """
impurity = np.array([[-0.72, 0.16+0.09j], [0.16-0.09j, -0.31]], complex)
bath = np.array([[[-1.1, 0.12-0.03j], [0.12+0.03j, 0.45]],
                 [[-0.25, -0.08+0.11j], [-0.08-0.11j, 1.2]]], complex)
coupling = np.array([[[0.21+0.04j, -0.07+0.13j], [0.09-0.02j, 0.18+0.06j]],
                     [[-0.11+0.08j, 0.15-0.05j], [0.04+0.12j, -0.19+0.03j]]], complex)
interaction = 1.35
""",
            "call": "pack(build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "gold_call": "pack(_oracle_build_noncommuting_impurity_model, impurity, bath, coupling, interaction)",
            "tol": 3e-12,
        },
        {
            "setup": """import numpy as np
impurity = np.array([[0.0, 0.2], [0.0, 0.1]], complex)
bath = np.empty((0, 2, 2), complex)
coupling = np.empty((0, 2, 2), complex)
def candidate():
    try:
        build_noncommuting_impurity_model(impurity.copy(), bath.copy(), coupling.copy(), 1.0)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
def reference():
    try:
        _oracle_build_noncommuting_impurity_model(impurity.copy(), bath.copy(), coupling.copy(), 1.0)
        return np.array([0.0])
    except ValueError:
        return np.array([1.0])
""",
            "call": "candidate()",
            "gold_call": "reference()",
        },
    ]
