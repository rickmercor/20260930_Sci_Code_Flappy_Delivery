"""
Solve the source's spin-adapted Bethe-Salpeter excitation problem for the singlet manifold (triplet = 0) or the triplet manifold (triplet = 1). The occupied-virtual pairs (i, a), i = 0..n_occ-1, a = n_occ..K-1, are ordered i-major and a-minor; eri and W0 are chemists'-notation arrays with the layout of steps 03 and 04. Keep the n = n_occ (K - n_occ) positive roots Omega_nu in ascending order; normalise each eigenvector to sum_{ia} (X^2 - Y^2) = 1 and fix its sign so that the component of X + Y of largest magnitude is positive. Return an array of shape (2n + 1, n): row 0 = Omega, rows 1..n = X (row nu, column = pair), rows n+1..2n = Y. Raise ValueError if the shapes are inconsistent, n_occ is out of range, triplet is not 0 or 1, the reference is unstable, or a squared excitation energy is not positive.

The Bethe-Salpeter equation with a statically screened exchange kernel gives the excited states whose energies and transition amplitudes replace the RPA screening of GW inside the source's self-energy; singlets and triplets decouple for a closed-shell reference and enter the self-energies with different weights.

Returns
-------
numpy.ndarray of float64 with shape (2n + 1, n): excitation energies, X and Y.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bse_excitations(eps: "np.ndarray", eri: "np.ndarray", W0: "np.ndarray", n_occ: int, triplet: int) -> "np.ndarray":
    """Solve the source's spin-adapted Bethe-Salpeter excitation problem for the singlet manifold (triplet = 0) or the triplet manifold (triplet = 1). The occupied-virtual pairs (i, a), i = 0..n_occ-1, a = n_occ..K-1, are ordered i-major and a-minor; eri and W0 are chemists'-notation arrays with the layout of steps 03 and 04. Keep the n = n_occ (K - n_occ) positive roots Omega_nu in ascending order; normalise each eigenvector to sum_{ia} (X^2 - Y^2) = 1 and fix its sign so that the component of X + Y of largest magnitude is positive. Return an array of shape (2n + 1, n): row 0 = Omega, rows 1..n = X (row nu, column = pair), rows n+1..2n = Y. Raise ValueError if the shapes are inconsistent, n_occ is out of range, triplet is not 0 or 1, the reference is unstable, or a squared excitation energy is not positive.

    Parameters
    ----------
    eps : numpy.ndarray
        Reference orbital energies, length K.
    eri : numpy.ndarray
        Bare chemists' integrals (K, K, K, K).
    W0 : numpy.ndarray
        Statically screened interaction (K, K, K, K).
    n_occ : int
        Number of occupied orbitals.
    triplet : int
        0 for the singlet manifold, 1 for the triplet manifold.

    Returns
    -------
    bse : numpy.ndarray
        Array (2n + 1, n): Omega, then X, then Y.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ or triplet is out of range, or the reference is unstable.
    """
    return bse

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bse_excitations(eps: "np.ndarray", eri: "np.ndarray", W0: "np.ndarray", n_occ: int, triplet: int) -> "np.ndarray":
    """Casida problem [[A, B], [-B, -A]] (X, Y)^T = Omega (X, Y)^T with the source's (v, W0) kernel (Eq 21-22):
    singlet A = D + 2J - K, B = 2J - K'; triplet A = D - K, B = -K', where D_{ia,jb} = (e_a - e_i) delta delta,
    J_{ia,jb} = (ia|jb), K_{ia,jb} = (ij|W0|ab), K'_{ia,jb} = (ib|W0|aj); pairs (i, a) ordered i-major, a-minor.
    Positive roots Omega_nu ascending; each (X, Y) normalised to sum(X^2 - Y^2) = 1 with the sign fixed so that
    the component of X + Y of largest magnitude is positive.  Returns array (2 n + 1, n), n = n_occ (K - n_occ):
    row 0 = Omega, rows 1..n = X (row = excitation nu, column = pair), rows n+1..2n = Y."""
    eps = np.asarray(eps, dtype=float).ravel()
    eri = np.asarray(eri, dtype=float)
    W0 = np.asarray(W0, dtype=float)
    n_occ, triplet = int(n_occ), int(triplet)
    K = eps.size
    if eri.shape != (K, K, K, K) or W0.shape != (K, K, K, K) or not (1 <= n_occ < K) or triplet not in (0, 1):
        raise ValueError("inconsistent shapes, n_occ out of range or triplet not in {0, 1}")
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, K)]
    n = len(pairs)
    D = np.diag([eps[a] - eps[i] for i, a in pairs])
    J = np.array([[eri[i, a, j, b] for (j, b) in pairs] for (i, a) in pairs])
    Kx = np.array([[W0[i, j, a, b] for (j, b) in pairs] for (i, a) in pairs])
    Kp = np.array([[W0[i, b, a, j] for (j, b) in pairs] for (i, a) in pairs])
    if triplet:
        A, B = D - Kx, -Kp
    else:
        A, B = D + 2.0 * J - Kx, 2.0 * J - Kp
    w, Uv = np.linalg.eigh(A - B)
    if w.min() <= 0.0:
        raise ValueError("A - B is not positive definite (reference instability)")
    S = Uv @ np.diag(np.sqrt(w)) @ Uv.T
    O2, T = np.linalg.eigh(S @ (A + B) @ S)
    if O2.min() <= 0.0:
        raise ValueError("imaginary excitation energy")
    Om = np.sqrt(O2)
    XpY = S @ T / np.sqrt(Om)[None, :]
    XmY = np.linalg.solve(S, T) * np.sqrt(Om)[None, :]
    X, Y = 0.5 * (XpY + XmY), 0.5 * (XpY - XmY)
    for nu in range(n):
        k = int(np.argmax(np.abs(XpY[:, nu])))
        if XpY[k, nu] < 0.0:
            X[:, nu] = -X[:, nu]
            Y[:, nu] = -Y[:, nu]
    return np.vstack([Om[None, :], X.T, Y.T])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "eps_site = np.array([0.4, -0.3, 0.2, -0.1, 0.3, -0.5, 0.1, -0.2]) - 1.5\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 4\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\ntriplet = 0\n",
            "call": "bse_excitations(eps_m, eri_m, W0_m, n_pairs, triplet)",
            "gold_call": "_oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, triplet)",
        },
        {
            "setup": "eps_site = np.array([0.2, -0.4, 0.1, 0.3, -0.2, -0.1]) - 1.0\nt, delta, U, kappa, n_pairs = 1.0, 0.1, 3.0, 2.0, 3\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\ntriplet = 1\n",
            "call": "bse_excitations(eps_m, eri_m, W0_m, n_pairs, triplet)",
            "gold_call": "_oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, triplet)",
        },
        {
            "setup": "eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\ntriplet = 0\n",
            "call": "bse_excitations(eps_m, eri_m, W0_m, n_pairs, triplet)",
            "gold_call": "_oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, triplet)",
        },
        {
            "setup": "eps_site = np.array([-0.5, -0.9])\nt, delta, U, kappa, n_pairs = 1.0, 0.15, 4.0, 1.5, 1\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nhV_g = _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nh_g, V_g = hV_g[0], hV_g[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\nref_g = _oracle_rhf_reference(h_g, V_g, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neps_g, C_g = ref_g[0], ref_g[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\neri_g = _oracle_mo_two_electron_integrals(V_g, C_g)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\nW0_g = _oracle_static_screened_interaction(eps_g, eri_g, n_pairs)\ntriplet = 0\n",
            "call": "bse_excitations(eps_m, eri_m, W0_m, n_pairs, triplet)",
            "gold_call": "_oracle_bse_excitations(eps_g, eri_g, W0_g, n_pairs, triplet)",
        },
        {'setup': 'eps_site = np.array([0.3, -0.1, 0.2, -0.4]) - 0.8\nt, delta, U, kappa, n_pairs = 1.0, 0.2, 2.5, 1.0, 2\nhV_m = ppp_hamiltonian(t, delta, U, kappa, eps_site)\nh_m, V_m = hV_m[0], hV_m[1]\nref_m = rhf_reference(h_m, V_m, n_pairs)\neps_m, C_m = ref_m[0], ref_m[1:]\neri_m = mo_two_electron_integrals(V_m, C_m)\nW0_m = static_screened_interaction(eps_m, eri_m, n_pairs)\ntriplet = 2\ndef _fx_exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n', 'call': '_fx_exception_code(bse_excitations, eps_m, eri_m, W0_m, n_pairs, triplet)', 'gold_call': '(hV_g := _oracle_ppp_hamiltonian(t, delta, U, kappa, eps_site), h_g := hV_g[0], V_g := hV_g[1], ref_g := _oracle_rhf_reference(h_g, V_g, n_pairs), eps_g := ref_g[0], C_g := ref_g[1:], eri_g := _oracle_mo_two_electron_integrals(V_g, C_g), W0_g := _oracle_static_screened_interaction(eps_g, eri_g, n_pairs), _fx_exception_code(_oracle_bse_excitations, eps_g, eri_g, W0_g, n_pairs, triplet))[-1]'},
    ]
