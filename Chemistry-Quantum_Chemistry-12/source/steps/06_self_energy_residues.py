"""
Return the residue tensor of the source's minimal positive-semidefinite self-energy for every excitation nu of a singlet solution of step 05 and every intermediate orbital k, as an array R of shape (n, K, K, K) indexed [nu, k, p, q]: R[nu, k] is the rank-one residue matrix of that pole, normalised so that the self-energy of step 07 is exactly the sum of R over its poles with no further factor. The source writes the hole-intermediate case (occupied k); for a virtual intermediate k = c use the particle reading in which the roles of X and Y in b are exchanged - this resolves the source's 'treated identically' for the particle branch and is fixed by the exact second-order limit. The sums run over the occupied-virtual pairs of step 05 in the same order; (pq|rs) are the bare integrals and (pq|W0|rs) the screened ones, both in chemists' notation. Raise ValueError if the shapes are inconsistent or n_occ is out of range.

The imaginary part of a self-energy diagram can be cut into two scattering amplitudes, one through the bare interaction and one through the screened exchange; a self-energy that is a complete square of their sum has a positive spectral function, which is the source's construction, and its residues at each pole are rank-one matrices.

Returns
-------
numpy.ndarray of float64 with shape (n, K, K, K): the residue matrices of the PSD-I self-energy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_energy_residues(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Return the residue tensor of the source's minimal positive-semidefinite self-energy for every excitation nu of a singlet solution of step 05 and every intermediate orbital k, as an array R of shape (n, K, K, K) indexed [nu, k, p, q]: R[nu, k] is the rank-one residue matrix of that pole, normalised so that the self-energy of step 07 is exactly the sum of R over its poles with no further factor. The source writes the hole-intermediate case (occupied k); for a virtual intermediate k = c use the particle reading in which the roles of X and Y in b are exchanged - this resolves the source's 'treated identically' for the particle branch and is fixed by the exact second-order limit. The sums run over the occupied-virtual pairs of step 05 in the same order; (pq|rs) are the bare integrals and (pq|W0|rs) the screened ones, both in chemists' notation. Raise ValueError if the shapes are inconsistent or n_occ is out of range.

    Parameters
    ----------
    eri : numpy.ndarray
        Bare chemists' integrals (K, K, K, K).
    W0 : numpy.ndarray
        Statically screened interaction (K, K, K, K).
    bse : numpy.ndarray
        Singlet solution of step 05, shape (2n + 1, n).
    n_occ : int
        Number of occupied orbitals.

    Returns
    -------
    res : numpy.ndarray
        Array (n, K, K, K): R[nu, k, p, q] = c_p c_q, the residue matrix of pole (nu, k).

    Raises
    ------
    ValueError
        If the shapes are inconsistent or n_occ is out of range.
    """
    return res

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _half_diagrams(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Helper: the source's half-diagrams a and b (Eq 12) for every excitation nu and every intermediate orbital k:
    a[nu, k, p] = sqrt(2) sum_{jb} (pk|jb) (X + Y)^nu_{jb};
    for occupied k (hole intermediates, Eq 12b) b[nu, k, q] = -(1/sqrt 2) sum_{ia} [ (iq|W0|ka) X^nu_{ia} + (aq|W0|ki) Y^nu_{ia} ];
    for virtual k = c (particle intermediates) the particle-hole transform b[nu, c, q] = -(1/sqrt 2) sum_{ia} [ (aq|W0|ci) X^nu_{ia} + (iq|W0|ca) Y^nu_{ia} ].
    Returns array (2, n, K, K): [a, b] indexed [nu, k, orbital]."""
    eri = np.asarray(eri, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    bse = np.asarray(bse, dtype=float)
    n_occ = int(n_occ)
    K = eri.shape[0]
    n = n_occ * (K - n_occ)
    if eri.shape != (K, K, K, K) or W0.shape != (K, K, K, K) or bse.shape != (2 * n + 1, n) or not (1 <= n_occ < K):
        raise ValueError("inconsistent shapes")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    X, Y = bse[1:n + 1], bse[n + 1:]                       # [nu, pair]
    XpY = X + Y
    # a: (pk|jb) contracted with (X + Y)
    vjb = np.array([eri[:, :, j, b] for (j, b) in pairs])   # [pair, p, k]
    a = np.sqrt(2.0) * np.einsum('nm,mpk->nkp', XpY, vjb)
    # b: hole intermediates (occupied k) and particle intermediates (virtual c)
    b = np.zeros((n, K, K))
    for m, (i, aa) in enumerate(pairs):
        # (iq|W0|ka) -> W0[i, q, k, aa];  (aq|W0|ki) -> W0[aa, q, k, i]
        hole = np.einsum('n,qk->nkq', X[:, m], W0[i, :, :, aa]) + np.einsum('n,qk->nkq', Y[:, m], W0[aa, :, :, i])
        part = np.einsum('n,qk->nkq', X[:, m], W0[aa, :, :, i]) + np.einsum('n,qk->nkq', Y[:, m], W0[i, :, :, aa])
        b[:, :n_occ, :] += hole[:, :n_occ, :]
        b[:, n_occ:, :] += part[:, n_occ:, :]
    b *= -1.0 / np.sqrt(2.0)
    return np.stack([a, b])


def _oracle_self_energy_residues(eri: "np.ndarray", W0: "np.ndarray", bse: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Residue tensor of the source's minimal PSD self-energy: R[nu, k, p, q] = c_p c_q with the complete amplitude
    c[nu, k, p] = a[nu, k, p] + b[nu, k, p] of Eq 12 (a: bare interaction x spin-summed singlet transition density,
    sqrt(2) prefactor; b: screened exchange with X and Y, -1/sqrt(2) prefactor; particle intermediates with X and Y
    exchanged), so that Sigma_pq(omega) = sum_{nu,k} R[nu, k, p, q] / (omega - pole_{nu,k}).
    Returns array (n, K, K, K) indexed [nu, k, p, q]."""
    hd = _half_diagrams(eri, W0, bse, n_occ)
    c = hd[0] + hd[1]                                       # [nu, k, p]
    return np.einsum('nkp,nkq->nkpq', c, c)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\n",
            "call": "self_energy_residues(eri_m, W0_m, bse_m, n_pairs)",
            "gold_call": "_oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\n",
            "call": "self_energy_residues(eri_m, W0_m, bse_m, n_pairs)",
            "gold_call": "_oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\n",
            "call": "self_energy_residues(eri_m, W0_m, bse_m, n_pairs)",
            "gold_call": "_oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_g = _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0)\n",
            "call": "self_energy_residues(eri_m, W0_m, bse_m, n_pairs)",
            "gold_call": "_oracle_self_energy_residues(eri_g, W0_g, bse_g, n_pairs)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nbse_m = bse_excitations(eps_m, eri_m, W0_m, n_pairs, 0)\nbse_m = bse_m[:-1]\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(self_energy_residues, eri_m, W0_m, bse_m, n_pairs)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], ref_g := _oracle_rhf_reference(h_g, V_g, n_pairs), eps_g := ref_g[0], C_g := ref_g[1:], eri_g := _oracle_mo_two_electron_integrals(V_g, C_g), W0_g := _oracle_static_screened_interaction(eps_g, eri_g, n_pairs), bse_g := _oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, 0), bse_g := bse_g[:-1], _fx_exception_code(_oracle_self_energy_residues, eri_g, W0_g, bse_g, n_pairs))[-1]'},
    ]
