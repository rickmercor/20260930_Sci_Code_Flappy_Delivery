"""
Return the generalised Fock matrix (orbital Lagrangian) of pCCD, F[p, q] = -<Phi0| (1 + Lambda) exp(-T) a+_{q alpha} [H, a_{p alpha}] exp(T) |Phi0> (alpha electron removed; H = sum h_pq E_pq + (1/2) sum (pq|rs) (E_pq E_rs - delta_qr E_ps)), evaluated from the per-spin response density matrices rdms of the preceding step, with the pair-transfer block entering only through its symmetrised form (Gamma^{p pbar}_{q qbar} + Gamma^{q qbar}_{p pbar}) / 2. Raise ValueError if the shapes are inconsistent.

The generalised Fock matrix contracts the one- and two-electron integrals with the response densities; its antisymmetric part is the orbital gradient of the energy functional and its symmetric part, at the variational orbitals, is the matrix of the extended Koopmans' eigenvalue problem.

Returns
-------
float array of shape (K, K).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generalized_fock(hm: "np.ndarray", Vm: "np.ndarray", rdms: "np.ndarray") -> "np.ndarray":
    '''Generalised Fock matrix of pCCD from the response density matrices.

    Parameters
    ----------
    hm : np.ndarray
        One-body matrix (K, K) in the orbital basis.
    Vm : np.ndarray
        Chemists' integrals (pq|rs), array (K, K, K, K).
    rdms : np.ndarray
        Response density matrices (3, K, K) in the layout of the preceding step.

    Returns
    -------
    F : np.ndarray
        Generalised Fock matrix F (K, K), element F[p, q].

    Raises
    ------
    ValueError
        If the shapes are inconsistent.
    '''
    return F

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_generalized_fock(hm: "np.ndarray", Vm: "np.ndarray", rdms: "np.ndarray") -> "np.ndarray":
    """Eq 22: F_pq = h_pq gamma_q + sum_r [ <qr||pr> G_qr + <q rbar|p rbar> G_qr + <p qbar|r rbar> Gt_qr ],
    physicists' <pq|rs> = (pr|qs), Gt = (Gp + Gp^T)/2. Returns F (K, K)."""
    hm = np.asarray(hm, dtype=float); Vm = np.asarray(Vm, dtype=float); rdms = np.asarray(rdms, dtype=float)
    K = hm.shape[0]
    if rdms.shape != (3, K, K) or Vm.shape != (K, K, K, K):
        raise ValueError("bad shapes")
    gam = np.diag(rdms[0]); G = rdms[1].copy(); Gp = rdms[2]
    np.fill_diagonal(G, 0.0)
    Gt = 0.5 * (Gp + Gp.T)
    phys = np.einsum('prqs->pqrs', Vm)
    anti = phys - np.einsum('pqrs->pqsr', phys)
    F = hm * gam[None, :]
    F += np.einsum('qrpr,qr->pq', anti, G) + np.einsum('qrpr,qr->pq', phys, G)
    F += np.einsum('pqrr,qr->pq', phys, Gt)
    return F

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of step test cases (normal, boundary and edge inputs, plus one invalid input)."""
    return [
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.15, 6.0, 4.0, -4.0 + np.array([0.0, 0.3, -0.2, 0.1, -0.3, 0.2, 0.15, -0.25]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 4)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 4)\nrdms = _oracle_pccd_response_rdms(c, _oracle_pccd_lambda_amplitudes(hm, Vm, c))",
            "call": "generalized_fock(hm, Vm, rdms)",
            "gold_call": "_oracle_generalized_fock(hm, Vm, rdms)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(1.0, 0.1, 4.0, 3.0, -3.5 + np.array([0.0, 0.2, -0.1, 0.3, -0.2, 0.1]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 3)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 3)\nrdms = _oracle_pccd_response_rdms(c, _oracle_pccd_lambda_amplitudes(hm, Vm, c))",
            "call": "generalized_fock(hm, Vm, rdms)",
            "gold_call": "_oracle_generalized_fock(hm, Vm, rdms)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nhV = _oracle_ppp_hamiltonian(0.8, 0.0, 5.0, 2.5, np.array([-3.0, -3.2, -2.75, -3.15]))\nC = _oracle_rhf_orbitals(hV[0], hV[1], 2)\nhm = C.T @ hV[0] @ C\nVm = _oracle_mo_two_electron_integrals(hV[1], C)\nc = _oracle_pccd_amplitudes(hm, Vm, 2)\nrdms = _oracle_pccd_response_rdms(c, _oracle_pccd_lambda_amplitudes(hm, Vm, c))",
            "call": "generalized_fock(hm, Vm, rdms)",
            "gold_call": "_oracle_generalized_fock(hm, Vm, rdms)",
            "tol": 1e-10,
        },
        {  # invalid input: the documented ValueError contract
            "setup": "import numpy as np\ndef run_model():\n    try:\n        generalized_fock(np.zeros((4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((2, 4, 4)))\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_generalized_fock(np.zeros((4, 4)), np.zeros((4, 4, 4, 4)), np.zeros((2, 4, 4)))\n        return 0\n    except ValueError:\n        return 1",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
