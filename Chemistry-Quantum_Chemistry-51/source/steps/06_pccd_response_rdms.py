"""
Return the response reduced density matrices of the pCCD wavefunction with amplitudes c and multipliers lam, per spin (alpha electrons), as one array (3, K, K) in the orbital basis (orbitals 0..P-1 occupied in the reference): rdms[0] the diagonal 1-RDM diag(gamma_p); rdms[1][p, q] = Gamma^{p qbar}_{p qbar} = <a+_{p alpha} a+_{q beta} a_{q beta} a_{p alpha}> (equal to the same-spin element Gamma^{pq}_{pq} for p != q, with the diagonal set to gamma_p); rdms[2][p, q] = Gamma^{p pbar}_{q qbar} = <a+_{p alpha} a+_{p beta} a_{q beta} a_{q alpha}> (pair transfer q -> p). Response (Lagrangian) density matrices: gamma_pq = <Phi0| (1 + Lambda) exp(-T) a+_{p alpha} a_{q alpha} exp(T) |Phi0> and Gamma^{pq}_{rs} = <Phi0| (1 + Lambda) exp(-T) a+_p a+_q a_s a_r exp(T) |Phi0> (spin labels as indicated), with the de-excitation operator Lambda = sum_{ia} lambda_ia P+_i P_a; the 1-RDM is diagonal and the only non-zero 2-RDM blocks are Gamma^{pq}_{pq} = Gamma^{p qbar}_{p qbar} (same-orbital-pair, p != q) and the pair-transfer block Gamma^{p pbar}_{q qbar} (Gamma^{p pbar}_{p pbar} = gamma_p). Raise ValueError if c and lam are not 2-D arrays of the same shape.

Because pCCD is a product of natural geminals (a seniority-zero wavefunction), its response 1-RDM is diagonal in the orbital basis and the 2-RDM has only three non-zero index patterns; all of them are low-order polynomials in the amplitudes and multipliers, so the densities come at no extra cost after the Lambda equations.

Returns
-------
float array of shape (3, K, K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pccd_response_rdms(c: "np.ndarray", lam: "np.ndarray") -> "np.ndarray":
    '''Response one- and two-particle reduced density matrices of pCCD (per spin).

    Parameters
    ----------
    c : np.ndarray
        pCCD amplitudes (P, K - P).
    lam : np.ndarray
        Lambda amplitudes (P, K - P).

    Returns
    -------
    rdms : np.ndarray
        Array (3, K, K): rdms[0] = diag(gamma_p), rdms[1][p, q] = Gamma^{p qbar}_{p qbar} (= Gamma^{pq}_{pq}; diagonal gamma_p), rdms[2][p, q] = Gamma^{p pbar}_{q qbar}.

    Raises
    ------
    ValueError
        If c and lam are not 2-D arrays of the same shape.
    '''
    return rdms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pccd_response_rdms(c: "np.ndarray", lam: "np.ndarray") -> "np.ndarray":
    """Response 1- and 2-RDMs of pCCD (per spin), Eqs 13-19 of the source with Eq 17 corrected.
    Returns (3, K, K): [0] diag(gamma_p); [1] Gamma^{p qbar}_{p qbar} (= Gamma^{pq}_{pq}; diagonal = gamma_p);
    [2] Gamma^{p pbar}_{q qbar} (pair-transfer block)."""
    c = np.asarray(c, dtype=float); lam = np.asarray(lam, dtype=float)
    if c.ndim != 2 or c.shape != lam.shape:
        raise ValueError("c and lambda must be 2-D arrays of the same shape")
    P, nv = c.shape; K = P + nv
    s_i = (c * lam).sum(axis=1); s_a = (c * lam).sum(axis=0)
    gam = np.concatenate([1.0 - s_i, s_a])
    G = np.zeros((K, K)); Gp = np.zeros((K, K))
    o = slice(0, P); v = slice(P, K)
    G[o, o] = 1.0 - s_i[:, None] - s_i[None, :]
    G[o, v] = s_a[None, :] - lam * c
    G[v, o] = G[o, v].T
    G[v, v] = 0.0
    np.fill_diagonal(G, gam)
    Gp[o, o] = c @ lam.T                                     # sum_c lambda_jc c_ic  -> [i, j]
    Gp[o, o] += np.diag(1.0 - 2.0 * s_i)
    Gp[o, v] = (c + 2.0 * lam * c * c - 2.0 * s_a[None, :] * c - 2.0 * s_i[:, None] * c + c @ lam.T @ c)
    Gp[v, o] = lam.T
    Gp[v, v] = lam.T @ c
    return np.stack([np.diag(gam), G, Gp])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 4)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 4)\nlam = _oracle_pccd_lambda_amplitudes(hm, Vm, c)",
            "call": "pccd_response_rdms(c, lam)",
            "gold_call": "_oracle_pccd_response_rdms(c, lam)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 3)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 3)\nlam = _oracle_pccd_lambda_amplitudes(hm, Vm, c)",
            "call": "pccd_response_rdms(c, lam)",
            "gold_call": "_oracle_pccd_response_rdms(c, lam)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nc = np.array([[0.1, -0.05], [0.02, 0.2]])\nlam = np.array([[0.12, -0.04], [0.03, 0.18]])",
            "call": "pccd_response_rdms(c, lam)",
            "gold_call": "_oracle_pccd_response_rdms(c, lam)",
            "tol": 1e-10,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        pccd_response_rdms(np.zeros((2, 2)), np.zeros((2, 3)))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_pccd_response_rdms(np.zeros((2, 2)), np.zeros((2, 3)))\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
