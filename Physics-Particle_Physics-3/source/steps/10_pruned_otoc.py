"""
Orchestrator. For the N-site longitudinal-field XXZ chain with couplings J, Jz, hz at inverse temperature beta, with W = Z on the 1-indexed site w_site and V = X on v_site, compute the thermal out-of-time-order correlator of the source's Eq (16) at t_final with the source's Methods B shadow-picture protocol: build the chain terms and the doubled terms, the root observable V^dagger (x) V^T, i.e. the string with X on site v_site of both copies, and the initial state W |TFD>, the thermofield double state with Z applied on site w_site of the first copy (check with the Liouvillian that the root is not conserved), prune the operator-growth graph with threshold eps1, build the hybrid Krylov basis inside the pruned set with threshold eps2, remainder tolerance tol and cap kmax, form the shadow matrix, initialise the shadow vector from the Pauli expectation values of the initial state, propagate it in the Heisenberg picture on the grid t_k = k dt for k = 0..t_final/dt, and also evaluate the exact series on the same grid through the earlier steps. Return the pruned correlator at t_final. Call the earlier step functions rather than reimplementing them. Raise ValueError if N is not an integer of at least 2, if w_site or v_site is not between 1 and N, or if 0 < dt <= t_final fails. Pauli strings are encoded as in step 1: code = sum_s f_s 4^s with f_s = 0, 1, 2, 3 for I, X, Z, Y on site s = 0..M-1, site 1 being the least significant digit s = 0.

The source demonstrates its pruning schemes on the thermal OTOC of the longitudinal-field XXZ chain, where the Krylov and hybrid schemes save register qubits even for small systems. At fixed thresholds the pruned dynamics is a property of the selection rules, not of a converged limit.

Returns
-------
float, the pruned OTOC at t_final.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt):
    """float, the pruned OTOC at t_final."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt):
    """ORCHESTRATOR. Thermal OTOC of Eq (16) with W = Z on w_site and V = X on v_site of the
    N-site chain at inverse temperature beta, computed with the source's Methods B protocol in
    the shadow picture: doubled Hamiltonian, root observable V^dagger (x) V^T, initial state
    W |TFD>; hybrid pruning (Alg 1 with eps1, Alg 2 with eps2, tol, kmax); shadow-matrix
    propagation on the grid t_k = k dt up to t_final; exact series for reference. Returns the
    pruned OTOC at t_final."""
    if int(N) != N or N < 2 or int(w_site) != w_site or int(v_site) != v_site:
        raise ValueError("N, w_site and v_site must be integers with N >= 2")
    if not (1 <= w_site <= N) or not (1 <= v_site <= N):
        raise ValueError("w_site and v_site must lie between 1 and N")
    if not (t_final > 0.0) or not (dt > 0.0) or dt > t_final:
        raise ValueError("need 0 < dt <= t_final")
    N = int(N)
    M = 2 * N
    terms = _oracle_lfxxz_terms(N, J, Jz, hz)
    termsD = _oracle_doubled_terms(terms, N)
    root = 1 * 4 ** (int(v_site) - 1) + 1 * 4 ** (N + int(v_site) - 1)     # V^dagger (x) V^T, V = X
    e_root = np.zeros(4 ** M)
    e_root[root] = 1.0
    l_root = _oracle_liouvillian_apply(termsD, M, e_root)
    if float(l_root @ l_root) == 0.0:
        raise ValueError("the root observable commutes with H_D, its dynamics is trivial")
    S = _oracle_graph_pruning(termsD, M, root, eps1)
    K = _oracle_hybrid_krylov(termsD, M, root, S[0], eps2, tol, kmax)
    A = _oracle_shadow_matrix(termsD, M, K)
    tfd = _oracle_thermofield_state(terms, N, beta)
    # |Psi_0> = (W (x) 1) |TFD>, W = Z on w_site of the first copy: a sign on the amplitudes
    # whose bit for that site (site 1 the most significant bit) is 1
    idx = np.arange(4 ** N)
    bit = (idx >> (M - int(w_site))) & 1
    psi0 = tfd * np.where(bit == 1, -1.0, 1.0)
    ev = _oracle_pauli_expectations(psi0, M)
    v0 = K @ ev
    nsteps = int(round(t_final / dt))
    ts = dt * np.arange(nsteps + 1)
    # shadow propagation v(t) = exp(-A t) v0 through the Hermitian matrix i A
    B = 0.5j * (A - A.T)
    lam, Wm = np.linalg.eigh(B)
    c = Wm.conj().T @ v0.astype(complex)
    series = np.array([float(np.real((Wm @ (np.exp(1j * lam * t) * c))[0])) for t in ts])
    exact = _oracle_exact_observable_series(termsD, M, psi0, root, ts)
    if not np.all(np.isfinite(series)) or not np.all(np.isfinite(exact)):
        raise ValueError("non-finite series")
    return float(series[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "N, J, Jz, hz, beta, w_site, v_site = 4, 1.0, 0.3, 1.0, 2.0, 1, 1\neps1, eps2, tol, kmax, t_final, dt = 0.02, 1e-6, 1e-12, 100, 2.5, 0.1",
            "call": "pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "gold_call": "_oracle_pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "tol": 1e-06,
        },
        {
            "setup": "N, J, Jz, hz, beta, w_site, v_site = 4, 1.0, 0.3, 1.0, 2.0, 1, 1\neps1, eps2, tol, kmax, t_final, dt = 0.02, 1e-6, 1e-12, 100, 2.5, 0.1\nt_final = 1.5",
            "call": "pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "gold_call": "_oracle_pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "tol": 1e-06,
        },
        {
            "setup": "N, J, Jz, hz, beta, w_site, v_site = 4, 1.0, 0.3, 1.0, 2.0, 1, 1\neps1, eps2, tol, kmax, t_final, dt = 0.02, 1e-6, 1e-12, 100, 2.5, 0.1\neps2 = 1e-3\nbeta = 1.0",
            "call": "pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "gold_call": "_oracle_pruned_otoc(N, J, Jz, hz, beta, w_site, v_site, eps1, eps2, tol, kmax, t_final, dt)",
            "tol": 1e-06,
        },
    ]
