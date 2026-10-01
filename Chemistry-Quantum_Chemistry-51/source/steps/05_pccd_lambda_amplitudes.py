"""
Return the Lagrange multipliers lambda (P, K - P) of the pCCD energy functional L = <Phi0| exp(-T) H exp(T) |Phi0> + sum_{ia} lambda_ia <Phi_i^a| exp(-T) H exp(T) |Phi0>, i.e. the solution of dL/dc_ia = 0 at the converged amplitudes c (the pCCD Lambda equations, a linear system for lambda). pCCD (pair coupled cluster doubles) wavefunction |Psi> = exp(T)|Phi0> with T = sum_{i occ, a vir} c_ia P+_a P_i, P+_p = a+_{p alpha} a+_{p beta} the pair creation operator, |Phi0> the closed-shell determinant of the first P orbitals of the given basis; amplitudes c (P, K - P) solve the projected equations <Phi_i^a| exp(-T) H exp(T) |Phi0> = 0 for every pair-excited determinant |Phi_i^a> = P+_a P_i |Phi0>. Raise ValueError if the shapes are inconsistent or c does not satisfy the pCCD equations to 1e-8.

Coupled cluster energies are not variational in the amplitudes, so response properties and density matrices need the left-hand (de-excitation) amplitudes that make the energy functional stationary; for pair coupled cluster these multipliers pair one-to-one with the amplitudes.

Returns
-------
float array of shape (P, K - P).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pccd_lambda_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    '''Lagrange multipliers (de-excitation amplitudes) of the pCCD energy functional.

    Parameters
    ----------
    hm : np.ndarray
        One-body matrix (K, K) in the orbital basis.
    Vm : np.ndarray
        Chemists' integrals (pq|rs), array (K, K, K, K).
    c : np.ndarray
        Converged pCCD amplitudes (P, K - P).

    Returns
    -------
    lam : np.ndarray
        Multipliers lambda_ia, array (P, K - P).

    Raises
    ------
    ValueError
        If the shapes are inconsistent or c does not satisfy the pCCD equations (residual above 1e-8).
    '''
    return lam

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _pair_quantities(hm, Vm):
    """eps_p = 2 h_pp + (pp|pp); W_pq = 2(pp|qq) - (pq|pq) (p != q, W_pp = 0); K_pq = (pq|pq)."""
    J = np.einsum('ppqq->pq', Vm); Kx = np.einsum('pqpq->pq', Vm)
    eps = 2.0 * np.diag(hm) + np.diag(J)
    W = 2.0 * J - Kx
    np.fill_diagonal(W, 0.0)
    return eps, W, Kx


def _pccd_residual_jacobian(c, eps, W, Kx, P):
    """pCCD projected equations R_ia = <Phi_i^a| e^{-T} H e^{T} |Phi_0> (pair Hamiltonian form) and dR/dc."""
    K = eps.size; o = slice(0, P); v = slice(P, K); nv = K - P
    Wo = W[:, o].sum(axis=1)
    D = (eps[v][None, :] - eps[o][:, None]) + 2.0 * ((Wo[v][None, :] - W[v, o].T) - Wo[o][:, None])
    Kov = Kx[o, v]; Kvv = Kx[v, v].copy(); np.fill_diagonal(Kvv, 0.0); Koo = Kx[o, o].copy(); np.fill_diagonal(Koo, 0.0)
    q = (c * Kov).sum(axis=1)[:, None] + (c * Kov).sum(axis=0)[None, :] - c * Kov
    R = Kov + c * D + c @ Kvv + Koo @ c + c @ Kov.T @ c - 2.0 * c * q
    # Jacobian dR_ia/dc_jb
    Jm = np.zeros((P, nv, P, nv))
    I_o = np.eye(P); I_v = np.eye(nv)
    Jm += np.einsum('ij,ab,ia->iajb', I_o, I_v, D - 2.0 * q)
    Jm += np.einsum('ij,ba->iajb', I_o, Kvv)
    Jm += np.einsum('ab,ij->iajb', I_v, Koo)
    Jm += np.einsum('ij,kb,ka->iajb', I_o, Kov, c)          # d/dc_jb of sum_kc c_ic K_kc c_ka, c index = b, i = j
    Jm += np.einsum('ab,ic,jc->iajb', I_v, c, Kov)          # k = j, a = b
    # -2 c_ia dq_ia/dc_jb, dq_ia/dc_jb = delta_ij K_ib + delta_ab K_ja - delta_ij delta_ab K_ia
    Jm -= 2.0 * np.einsum('ia,ij,ib->iajb', c, I_o, Kov)
    Jm -= 2.0 * np.einsum('ia,ab,ja->iajb', c, I_v, Kov)
    Jm += 2.0 * np.einsum('ia,ij,ab,ia->iajb', c, I_o, I_v, Kov)
    return R, Jm.reshape(P * nv, P * nv)


def _oracle_pccd_lambda_amplitudes(hm: "np.ndarray", Vm: "np.ndarray", c: "np.ndarray") -> "np.ndarray":
    """Lambda equations dL/dc_ia = 0 with L = E(c) + sum lambda_jb R_jb(c): J^T lambda = -K_ov. Returns lambda (P, K-P)."""
    hm = np.asarray(hm, dtype=float); Vm = np.asarray(Vm, dtype=float); c = np.asarray(c, dtype=float)
    K = hm.shape[0]; P = c.shape[0]
    if c.shape != (P, K - P) or hm.shape != (K, K) or Vm.shape != (K, K, K, K):
        raise ValueError("bad shapes")
    eps, W, Kx = _pair_quantities(hm, Vm)
    R, Jm = _pccd_residual_jacobian(c, eps, W, Kx, P)
    if np.max(np.abs(R)) > 1e-8:
        raise ValueError("amplitudes do not satisfy the pCCD equations")
    lam = np.linalg.solve(Jm.T, -Kx[:P, P:].ravel())
    return lam.reshape(P, K - P)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 4)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 4)",
            "call": "pccd_lambda_amplitudes(hm, Vm, c)",
            "gold_call": "_oracle_pccd_lambda_amplitudes(hm, Vm, c)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 3)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 3)",
            "call": "pccd_lambda_amplitudes(hm, Vm, c)",
            "gold_call": "_oracle_pccd_lambda_amplitudes(hm, Vm, c)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(0.8, 0.0, 5.0, 2.5, np.array([-3.0, -3.2, -2.75, -3.15]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 2)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 2)",
            "call": "pccd_lambda_amplitudes(hm, Vm, c)",
            "gold_call": "_oracle_pccd_lambda_amplitudes(hm, Vm, c)",
            "tol": 1e-10,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        pccd_lambda_amplitudes(np.zeros((4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((3, 2)))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_pccd_lambda_amplitudes(np.zeros((4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((3, 2)))\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
