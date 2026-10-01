"""
Build a completely positive trace preserving quantum channel from a fitted Hamiltonian and jump-operator scale pair, by exponentiating the resulting Lindblad generator over a fixed time and extracting Kraus operators from the resulting Choi matrix.



This turns a statistically fitted open-system generator into an object, a quantum channel, that can be composed with other channels, for example placed in a coherent switch with a second, independently specified channel.

Returns
-------
return Ks
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_from_lindbladian(seed_gen: int, alpha_hat: float, beta_hat: float, tau: float) -> "np.ndarray":
    """Build channel Kraus operators from a fitted Lindbladian by exponentiation.

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The estimated Hamiltonian and jump operator
        exponentiated here are ``H_hat = alpha_hat * H0`` and
        ``L_hat = beta_hat * L0``.
    alpha_hat : float
        Fitted Hamiltonian scale.
    beta_hat : float
        Fitted jump-operator scale.
    tau : float
        Fixed exponentiation time, tau > 0.

    Returns
    -------
    Ks : np.ndarray
        (m, d, d) complex array of the resulting channel's Kraus operators, one per
        Choi eigenvalue above 1e-8, ordered by increasing eigenvalue index.

    Raises
    ------
    ValueError
        If tau <= 0.
    """
    return Ks

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _build_qutrit_generators(seed: int) -> "tuple":
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    H0 = (A + A.conj().T) / 2
    L0 = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    return H0, L0


def _lindblad_superop(H: "np.ndarray", Ls: "list") -> "np.ndarray":
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(), L) - 0.5 * (np.kron(I, L.conj().T @ L) + np.kron((L.conj().T @ L).T, I))
    return Lsup


def _apply_superop(Esup: "np.ndarray", rho: "np.ndarray") -> "np.ndarray":
    d = rho.shape[0]
    vecrho = rho.reshape(-1, 1, order="F")
    out = Esup @ vecrho
    return out.reshape(d, d, order="F")


def _oracle_channel_from_lindbladian(seed_gen: int, alpha_hat: float, beta_hat: float, tau: float) -> "np.ndarray":
    if tau <= 0:
        raise ValueError("tau must be positive")
    H0, L0 = _build_qutrit_generators(seed_gen)
    H_hat = alpha_hat * H0
    L_hat = beta_hat * L0
    d = 3
    Lsup = _lindblad_superop(H_hat, [L_hat])
    Esup = expm(tau * Lsup)

    J = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            out = _apply_superop(Esup, Eij)
            J[i * d:(i + 1) * d, j * d:(j + 1) * d] = out
    J = (J + J.conj().T) / 2
    w, v = np.linalg.eigh(J)
    w_clipped = np.clip(w, 0, None)

    Ks = []
    for idx in range(len(w_clipped)):
        if w_clipped[idx] > 1e-8:
            vec = v[:, idx]
            K = np.sqrt(w_clipped[idx]) * vec.reshape(d, d, order="F")
            Ks.append(K)
    return np.array(Ks)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: generic seed, small tau. Checks completeness AND the channel's
        #     actual action on several representative input states, phase-invariant
        #     (comparing sum_i K_i rho K_i^dagger, not the raw Kraus operators, so a
        #     unitarily-equivalent but differently-phased Kraus set still matches) ---
        {
            "setup": """
import numpy as np
seed_gen = 9
alpha_hat = 1.0
beta_hat = 1.0
tau = 0.1
def summarize(fn):
    Ks = fn(seed_gen, alpha_hat, beta_hat, tau)
    S = sum(K.conj().T @ K for K in Ks)
    rho_a = np.array([[1.0,0,0],[0,0,0],[0,0,0]], dtype=complex)
    rho_b = np.array([[0,0,0],[0,1.0,0],[0,0,0]], dtype=complex)
    psi_c = np.array([1,1,0], dtype=complex) / np.sqrt(2)
    rho_c = np.outer(psi_c, psi_c.conj())
    action_a = sum(K @ rho_a @ K.conj().T for K in Ks)
    action_b = sum(K @ rho_b @ K.conj().T for K in Ks)
    action_c = sum(K @ rho_c @ K.conj().T for K in Ks)
    return np.round(np.concatenate([S.ravel(), action_a.ravel(), action_b.ravel(), action_c.ravel()]), 6)
""",
            "call": "summarize(channel_from_lindbladian)",
            "gold_call": "summarize(_oracle_channel_from_lindbladian)",
        },
        # --- Boundary: alpha_hat=beta_hat=0, channel should reduce to the identity map ---
        {
            "setup": """
import numpy as np
seed_gen = 11
alpha_hat = 0.0
beta_hat = 0.0
tau = 0.5
def summarize(fn):
    Ks = fn(seed_gen, alpha_hat, beta_hat, tau)
    total = sum(K @ np.array([[1.0,0,0],[0,0,0],[0,0,0]], dtype=complex) @ K.conj().T for K in Ks)
    return np.round(total, 6)
""",
            "call": "summarize(channel_from_lindbladian)",
            "gold_call": "summarize(_oracle_channel_from_lindbladian)",
        },
        # --- Edge: the actual task instance's fitted parameters and tau, checked
        #     against the oracle's actual action on representative states as well
        #     as completeness and Kraus count ---
        {
            "setup": """
import numpy as np
seed_gen = 101
alpha_hat = 1.89302208806816
beta_hat = 0.7232072266302545
tau = 0.3
def summarize(fn):
    Ks = fn(seed_gen, alpha_hat, beta_hat, tau)
    S = sum(K.conj().T @ K for K in Ks)
    count_row = np.full(S.shape[0], float(len(Ks)))
    rho_a = np.array([[1.0,0,0],[0,0,0],[0,0,0]], dtype=complex)
    rho_b = np.array([[0,0,0],[0,1.0,0],[0,0,0]], dtype=complex)
    psi_c = np.array([1,1,0], dtype=complex) / np.sqrt(2)
    rho_c = np.outer(psi_c, psi_c.conj())
    action_a = sum(K @ rho_a @ K.conj().T for K in Ks)
    action_b = sum(K @ rho_b @ K.conj().T for K in Ks)
    action_c = sum(K @ rho_c @ K.conj().T for K in Ks)
    return np.round(np.concatenate([count_row, S.ravel(), action_a.ravel(), action_b.ravel(), action_c.ravel()]), 5)
""",
            "call": "summarize(channel_from_lindbladian)",
            "gold_call": "summarize(_oracle_channel_from_lindbladian)",
        },
    ]
