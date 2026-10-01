"""
Return the orbitals C (K, K) that minimise the pCCD energy E(C) = <Phi0| exp(-T) H exp(T) |Phi0> (amplitudes re-solved from the projected equations at every orbital set; the first P columns doubly occupied in the reference) over all real orthogonal rotations of the starting orbitals C0, for the Hamiltonian with site-basis one-body matrix h and density-density integrals (pq|rs) = delta_pq delta_rs V_pr. Occupied-occupied, occupied-virtual and virtual-virtual rotations are all non-redundant. Converge until every component of the orbital gradient of the energy functional (dE/dkappa_pq for the rotation generator kappa, obtained from the generalised Fock matrix of the preceding step) is below 1e-10 in magnitude. Return the pCCD natural orbitals of the minimum: the columns ordered by descending per-spin occupation number of the response 1-RDM (the 1-RDM is diagonal in this basis). Column phase convention: in every column the first component with |c| > 1e-8 is positive. Raise ValueError if the shapes are inconsistent, n_pairs is outside [1, K - 1], or the optimisation does not converge.

The pCCD energy depends on the orbitals, and in its orbital-optimised form the orbitals are chosen variationally; the optimised orbitals are the natural orbitals of the wavefunction and typically localise on bonds, which is where the pair ansatz recovers most of the static correlation.

Returns
-------
float array (K, K), natural orbitals in columns (descending occupation), phase-fixed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def oo_pccd_orbitals(h: "np.ndarray", V: "np.ndarray", C0: "np.ndarray", n_pairs: int) -> "np.ndarray":
    '''Variationally orbital-optimised pCCD: the natural orbitals that minimise the pCCD energy.

    Parameters
    ----------
    h : np.ndarray
        Site-basis one-body matrix (K, K).
    V : np.ndarray
        Site-basis interaction matrix (K, K).
    C0 : np.ndarray
        Starting orbitals (K, K), e.g. the RHF orbitals.
    n_pairs : int
        Number of electron pairs P.

    Returns
    -------
    C : np.ndarray
        Optimised natural orbitals (K, K), columns sorted by descending occupation number, phase-fixed.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_pairs is outside [1, K - 1], or the optimisation does not converge.
    '''
    return C

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _fix_phase(C):
    """Column sign convention: first component with |c| > 1e-8 positive."""
    C = np.array(C, dtype=float)
    for k in range(C.shape[1]):
        col = C[:, k]
        nz = np.where(np.abs(col) > 1e-8)[0]
        if nz.size and col[nz[0]] < 0.0:
            C[:, k] = -col
    return C


def _pair_quantities(hm, Vm):
    """eps_p = 2 h_pp + (pp|pp); W_pq = 2(pp|qq) - (pq|pq) (p != q, W_pp = 0); K_pq = (pq|pq)."""
    J = np.einsum('ppqq->pq', Vm); Kx = np.einsum('pqpq->pq', Vm)
    eps = 2.0 * np.diag(hm) + np.diag(J)
    W = 2.0 * J - Kx
    np.fill_diagonal(W, 0.0)
    return eps, W, Kx


def _pccd_gfm_at(h, V, C, P):
    hm = C.T @ h @ C
    Vm = _oracle_mo_two_electron_integrals(V, C)
    c = _oracle_pccd_amplitudes(hm, Vm, P)
    lam = _oracle_pccd_lambda_amplitudes(hm, Vm, c)
    rd = _oracle_pccd_response_rdms(c, lam)
    F = _oracle_generalized_fock(hm, Vm, rd)
    eps, W, Kx = _pair_quantities(hm, Vm)
    E = eps[:P].sum() + W[:P, :P].sum() + (c * Kx[:P, P:]).sum()
    return E, F, np.diag(rd[0]), C


def _rotate(C, kv, iu):
    K = C.shape[0]; kap = np.zeros((K, K)); kap[iu] = kv; kap = kap - kap.T
    w, U = np.linalg.eigh(1j * kap)                          # exp(kap) via Hermitian i*kap
    R = (U * np.exp(1j * w)) @ U.conj().T                   # exp(-kap): kap = -i U diag(w) U^dag
    return C @ np.real(R)


def _oracle_oo_pccd_orbitals(h: "np.ndarray", V: "np.ndarray", C0: "np.ndarray", n_pairs: int) -> "np.ndarray":
    """Variational orbital optimisation of pCCD: minimise the pCCD energy over all orbital rotations
    C -> C exp(-kappa); gradient 4(F^T - F) from the generalised Fock matrix, Newton steps with a
    finite-difference Hessian of that gradient, converged to max|g| < 1e-10. Returns the natural
    orbitals (columns sorted by occupation number descending, phase-fixed)."""
    h = np.asarray(h, dtype=float); V = np.asarray(V, dtype=float); C = np.array(C0, dtype=float); P = int(n_pairs)
    K = h.shape[0]
    if C.shape != (K, K) or V.shape != (K, K) or not (1 <= P < K):
        raise ValueError("bad shapes or n_pairs")
    iu = np.triu_indices(K, 1); n = iu[0].size

    def _grad_at(Cx):
        E, F, gam, _ = _pccd_gfm_at(h, V, Cx, P)
        return E, (4.0 * (F.T - F))[iu], gam

    E, g, gam = _grad_at(C)
    for it in range(100):
        if np.max(np.abs(g)) < 1e-10:
            break
        Hs = np.zeros((n, n)); dk = 1e-4
        for m in range(n):
            e = np.zeros(n); e[m] = dk
            Hs[:, m] = (_grad_at(_rotate(C, e, iu))[1] - _grad_at(_rotate(C, -e, iu))[1]) / (2.0 * dk)
        Hs = 0.5 * (Hs + Hs.T)
        w, Uh = np.linalg.eigh(Hs)
        w = np.where(w > 1e-6, w, np.maximum(np.abs(w), 1e-6))    # positive-definite shift for descent
        step = -Uh @ ((Uh.T @ g) / w)
        smax = np.max(np.abs(step))
        if smax > 0.5:                                        # trust region on the rotation angle
            step *= 0.5 / smax
        alpha = 1.0
        for _ in range(30):
            Cn = _rotate(C, alpha * step, iu)
            try:
                En, gn, gamn = _grad_at(Cn)
            except ValueError:
                alpha *= 0.5
                continue
            if En <= E + 1e-12:
                break
            alpha *= 0.5
        else:
            raise ValueError("oo-pCCD line search failed")
        C, E, g, gam = Cn, En, gn, gamn
    else:
        raise ValueError("oo-pCCD did not converge")
    order = np.argsort(-gam)
    return _fix_phase(C[:, order])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nh, V, n_pairs = hV[0], hV[1], 4\nC0 = _oracle_rhf_orbitals(h, V, n_pairs)",
            "call": "oo_pccd_orbitals(h, V, C0, n_pairs)",
            "gold_call": "_oracle_oo_pccd_orbitals(h, V, C0, n_pairs)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nh, V, n_pairs = hV[0], hV[1], 3\nC0 = _oracle_rhf_orbitals(h, V, n_pairs)",
            "call": "oo_pccd_orbitals(h, V, C0, n_pairs)",
            "gold_call": "_oracle_oo_pccd_orbitals(h, V, C0, n_pairs)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(0.8, 0.0, 5.0, 2.5, np.array([-3.0, -3.2, -2.75, -3.15]))\nh, V, n_pairs = hV[0], hV[1], 2\nC0 = _oracle_rhf_orbitals(h, V, n_pairs)",
            "call": "oo_pccd_orbitals(h, V, C0, n_pairs)",
            "gold_call": "_oracle_oo_pccd_orbitals(h, V, C0, n_pairs)",
            "tol": 1e-08,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        oo_pccd_orbitals(np.zeros((4, 4)), np.eye(4), np.eye(4), 0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_oo_pccd_orbitals(np.zeros((4, 4)), np.eye(4), np.eye(4), 0)\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
