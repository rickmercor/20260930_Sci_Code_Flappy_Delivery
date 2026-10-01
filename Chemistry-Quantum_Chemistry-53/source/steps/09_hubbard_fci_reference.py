"""
Construct the exact closed-shell FCI benchmark and return its energy, full spin-summed one-particle density matrix, site-resolved double occupations, and exact static density-response matrix.

The exact benchmark is evaluated in the fixed N_up = N_down sector. Fermionic phases must be retained for all creation-annihilation pairs. The full excited-state spectrum determines the static response d<n_i>/d<v_j>, whose symmetry, negative semidefiniteness and uniform-potential null mode provide independent checks of the density-potential mapping.

Returns
-------
np.ndarray of float with shape (1 + 2*L*L + L,): the exact energy, row-major spin-summed one-particle density matrix, local double occupations, and row-major static density-response matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hubbard_fci_reference(h, U, n_elec):
    '''Exact closed-shell reference for a finite Hubbard Hamiltonian.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Finite real symmetric one-electron matrix in the site basis.
    U : float
        On-site interaction in U n_{i up} n_{i down}.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2. The calculation
        uses N_up = N_down = n_elec / 2 and supports 2 <= L <= 6.

    Returns
    -------
    result : np.ndarray of float, shape (1 + 2*L*L + L,)
        result[0] is the exact ground-state energy. The next L*L entries are
        the row-major spin-summed one-particle density matrix gamma[p,q] =
        sum_sigma <a^dagger_(p,sigma) a_(q,sigma)>. The following L entries
        are <n_(i,up) n_(i,down)>. The final L*L entries are the row-major
        static response d<n_i>/d<v_j>. Raises ValueError for invalid inputs
        or a degenerate ground state with a gap below 1e-10.
    '''
    return np.zeros(
        1 + 2 * np.asarray(h).shape[0] ** 2 + np.asarray(h).shape[0],
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from itertools import combinations


def _oracle_hubbard_fci_reference(h, U, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")

    L = h.shape[0]
    if L < 2 or L > 6:
        raise ValueError(
            "the exact reference supports 2 <= L <= 6"
        )
    if (
        not np.all(np.isfinite(h))
        or not np.allclose(
            h, h.T, rtol=0.0, atol=1e-10
        )
    ):
        raise ValueError("h must be finite and symmetric")

    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(n_elec, bool) or not isinstance(
        n_elec, (int, np.integer)
    ):
        raise ValueError("n_elec must be an integer")
    if (
        n_elec % 2 != 0
        or n_elec < 2
        or n_elec > 2 * L - 2
    ):
        raise ValueError(
            "n_elec must be even and satisfy "
            "2 <= n_elec <= 2L - 2"
        )

    n_spin = int(n_elec) // 2
    states = [
        sum(1 << p for p in occupied)
        for occupied in combinations(
            range(L), n_spin
        )
    ]
    index = {
        state: position
        for position, state in enumerate(states)
    }
    ncfg = len(states)

    h_spin = np.zeros(
        (ncfg, ncfg), dtype=float
    )

    for column, state in enumerate(states):
        for q in range(L):
            if not (state >> q) & 1:
                continue

            h_spin[column, column] += h[q, q]
            without_q = state ^ (1 << q)

            sign_q = (
                -1.0
                if (
                    state & ((1 << q) - 1)
                ).bit_count() % 2
                else 1.0
            )

            for p in range(L):
                if (
                    p == q
                    or (without_q >> p) & 1
                    or h[p, q] == 0.0
                ):
                    continue

                sign_p = (
                    -1.0
                    if (
                        without_q & ((1 << p) - 1)
                    ).bit_count() % 2
                    else 1.0
                )

                row_state = without_q | (1 << p)
                row = index[row_state]

                h_spin[row, column] += (
                    sign_q * sign_p * h[p, q]
                )

    identity = np.eye(ncfg)
    h_fci = (
        np.kron(h_spin, identity)
        + np.kron(identity, h_spin)
    )

    interaction = np.empty(
        ncfg * ncfg, dtype=float
    )
    basis_occupations = np.empty(
        (ncfg * ncfg, L), dtype=float
    )

    for ia, alpha in enumerate(states):
        for ib, beta in enumerate(states):
            position = ia * ncfg + ib

            interaction[position] = (
                U * (alpha & beta).bit_count()
            )

            for site in range(L):
                basis_occupations[
                    position, site
                ] = (
                    ((alpha >> site) & 1)
                    + ((beta >> site) & 1)
                )

    h_fci[np.diag_indices_from(h_fci)] += (
        interaction
    )

    levels, vectors = np.linalg.eigh(h_fci)
    if levels[1] - levels[0] < 1e-10:
        raise ValueError(
            "degenerate FCI ground state: "
            "density undefined"
        )

    coefficients = vectors[:, 0].reshape(
        ncfg, ncfg
    )
    probabilities = coefficients ** 2
    rdm = np.zeros((L, L), dtype=float)

    spin_density_matrices = (
        coefficients @ coefficients.T,
        coefficients.T @ coefficients,
    )

    for rho in spin_density_matrices:
        for column, state in enumerate(states):
            for q in range(L):
                if not (state >> q) & 1:
                    continue

                rdm[q, q] += rho[column, column]
                without_q = state ^ (1 << q)

                sign_q = (
                    -1.0
                    if (
                        state & ((1 << q) - 1)
                    ).bit_count() % 2
                    else 1.0
                )

                for p in range(L):
                    if (
                        p == q
                        or (without_q >> p) & 1
                    ):
                        continue

                    sign_p = (
                        -1.0
                        if (
                            without_q
                            & ((1 << p) - 1)
                        ).bit_count() % 2
                        else 1.0
                    )

                    row_state = (
                        without_q | (1 << p)
                    )
                    row = index[row_state]

                    rdm[p, q] += (
                        sign_q
                        * sign_p
                        * rho[row, column]
                    )

    double_occupations = np.zeros(
        L, dtype=float
    )

    for ia, alpha in enumerate(states):
        for ib, beta in enumerate(states):
            shared = alpha & beta
            weight = probabilities[ia, ib]

            for site in range(L):
                if (shared >> site) & 1:
                    double_occupations[site] += (
                        weight
                    )

    energy = float(levels[0])
    reconstructed_energy = float(
        np.sum(h * rdm.T)
        + U * np.sum(double_occupations)
    )

    if not np.isclose(
        reconstructed_energy,
        energy,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            "the reduced densities do not "
            "reconstruct the FCI energy"
        )

    ground = vectors[:, 0]
    transition_density = (
        vectors[:, 1:].T
        @ (
            basis_occupations
            * ground[:, None]
        )
    )

    inverse_gaps = (
        1.0 / (levels[0] - levels[1:])
    )
    response = (
        2.0
        * transition_density.T
        @ (
            transition_density
            * inverse_gaps[:, None]
        )
    )
    response = 0.5 * (
        response + response.T
    )

    return np.concatenate(
        (
            [energy],
            rdm.ravel(),
            double_occupations,
            response.ravel(),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _ring(L, t, v):
    h = np.diag(np.asarray(v, dtype=float))
    for i in range(L):
        h[i, (i + 1) % L] -= t
        h[(i + 1) % L, i] -= t
    return h
h6 = _ring(6, 1.0, [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0])
dense5 = np.array([
    [ 0.30, -0.80,  0.21,  0.00, -0.17],
    [-0.80, -0.40, -0.63,  0.19,  0.08],
    [ 0.21, -0.63,  0.80, -0.51,  0.14],
    [ 0.00,  0.19, -0.51, -0.20, -0.72],
    [-0.17,  0.08,  0.14, -0.72,  0.10],
])
dense6 = np.array([
    [-1.10, -0.70,  0.16, -0.09,  0.04, -0.31],
    [-0.70,  0.45, -0.58,  0.12, -0.07,  0.11],
    [ 0.16, -0.58, -0.35, -0.66,  0.18, -0.05],
    [-0.09,  0.12, -0.66,  0.90, -0.49,  0.13],
    [ 0.04, -0.07,  0.18, -0.49, -0.75, -0.61],
    [-0.31,  0.11, -0.05,  0.13, -0.61,  0.25],
])
"""
    return [
        # Normal: the exact benchmark for the task configuration.
        {
            "setup": setup,
            "call": "np.round(hubbard_fci_reference(h6, 7.0, 6), 8)",
            "gold_call": "np.round(_oracle_hubbard_fci_reference(h6, 7.0, 6), 8)",
        },
        # Hard: dense long-range hopping with one hole in each spin sector.
        {
            "setup": setup,
            "call": "np.round(hubbard_fci_reference(dense5, 3.7, 8), 9)",
            "gold_call": "np.round(_oracle_hubbard_fci_reference(dense5, 3.7, 8), 9)",
        },
        # Hard boundary: largest supported lattice at half and high filling.
        {
            "setup": setup,
            "call": "np.round(np.concatenate([hubbard_fci_reference(dense6, 5.2, 6), hubbard_fci_reference(dense6, 5.2, 10)]), 9)",
            "gold_call": "np.round(np.concatenate([_oracle_hubbard_fci_reference(dense6, 5.2, 6), _oracle_hubbard_fci_reference(dense6, 5.2, 10)]), 9)",
        },
        # Covariance: a site permutation must transform every returned block.
        {
            "setup": setup + "perm = np.array([3, 0, 4, 1, 2])\nhp = dense5[np.ix_(perm, perm)]\n",
            "call": "np.round(np.concatenate([hubbard_fci_reference(dense5, 3.7, 8), hubbard_fci_reference(hp, 3.7, 8)]), 9)",
            "gold_call": "np.round(np.concatenate([_oracle_hubbard_fci_reference(dense5, 3.7, 8), _oracle_hubbard_fci_reference(hp, 3.7, 8)]), 9)",
        },
        # Gauge covariance: a uniform site shift changes only the energy.
        {
            "setup": setup + "shifted6 = dense6 + 2.75 * np.eye(6)\n",
            "call": "np.round(np.concatenate([hubbard_fci_reference(dense6, 5.2, 6), hubbard_fci_reference(shifted6, 5.2, 6)]), 9)",
            "gold_call": "np.round(np.concatenate([_oracle_hubbard_fci_reference(dense6, 5.2, 6), _oracle_hubbard_fci_reference(shifted6, 5.2, 6)]), 9)",
        },
        # Response audit: compare the analytic matrix with finite differences.
        {
            "setup": setup + """
h4r = np.array([
    [-0.7, -0.8,  0.2,  0.0],
    [-0.8,  0.4, -0.6,  0.1],
    [ 0.2, -0.6,  0.9, -0.5],
    [ 0.0,  0.1, -0.5, -0.3],
])
def _response_audit(fn):
    eps = 2e-5
    base = fn(h4r, 3.2, 4)
    response = base[21:].reshape(4, 4)
    finite = np.zeros((4, 4))
    for j in range(4):
        hp = h4r.copy()
        hm = h4r.copy()
        hp[j, j] += eps
        hm[j, j] -= eps
        gp = fn(hp, 3.2, 4)[1:17].reshape(4, 4)
        gm = fn(hm, 3.2, 4)[1:17].reshape(4, 4)
        finite[:, j] = (np.diag(gp) - np.diag(gm)) / (2.0 * eps)
    return np.array([
        np.max(np.abs(response - finite)),
        np.max(np.abs(response - response.T)),
        np.max(np.abs(np.sum(response, axis=0))),
        np.max(np.linalg.eigvalsh(response)),
    ])
""",
            "call": "np.round(_response_audit(hubbard_fci_reference), 8)",
            "gold_call": "np.round(_response_audit(_oracle_hubbard_fci_reference), 8)",
        },
        # Boundary: the interacting term vanishes.
        {
            "setup": setup + "h5 = _ring(5, 0.8, [0.3, -0.4, 0.8, -0.2, 0.1])\n",
            "call": "np.round(hubbard_fci_reference(h5, 0.0, 4), 8)",
            "gold_call": "np.round(_oracle_hubbard_fci_reference(h5, 0.0, 4), 8)",
        },
        # Edge: one electron of each spin in a four-site interacting ring.
        {
            "setup": setup + "h4 = _ring(4, 1.0, [0.5, -0.5, 1.5, -1.5])\n",
            "call": "np.round(hubbard_fci_reference(h4, 3.0, 2), 8)",
            "gold_call": "np.round(_oracle_hubbard_fci_reference(h4, 3.0, 2), 8)",
        },
        # Invalid: an odd electron number cannot define the requested sector.
        {
            "setup": setup + """
def run_model():
    try:
        hubbard_fci_reference(h6, 7.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_hubbard_fci_reference(h6, 7.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
