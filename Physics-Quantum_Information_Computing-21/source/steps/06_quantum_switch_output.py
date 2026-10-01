"""
Place an estimated single-qutrit channel and a given two-qutrit Weyl-noise depolarizing channel in a quantum switch with a fixed inserted local unitary, and return the postselected output branch with the larger entanglement negativity, normalized to a valid density matrix.



A quantum switch coherently controls the order in which two channels act, rather than applying them in a fixed sequence, and postselecting on the control outcome can leave more entanglement in the output than either fixed order or any classical mixture of the two fixed orders.

Returns
-------
return rho_f
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def quantum_switch_output(Ks_est: "np.ndarray", lam2: float, dloc: int) -> "np.ndarray":
    """Run the two channels through a quantum switch and return the best branch.

    Parameters
    ----------
    Ks_est : np.ndarray
        (m, dloc, dloc) complex array, the Kraus operators of the estimated channel
        acting on a single qutrit, to be applied locally to the second of two qutrits
        (the first qutrit carries the identity): the embedded Kraus operators are
        ``np.kron(np.eye(dloc), K)`` for each ``K`` in ``Ks_est``, and the returned
        state's tensor factors follow this same first-qutrit-then-second-qutrit
        ordering throughout.
    lam2 : float
        Depolarizing parameter of the given two-qutrit Weyl-noise channel, 0 <= lam2 <= 1,
        in the convention where the channel acts as rho -> lam2*rho + (1-lam2)*I/dloc**2
        (lam2 = 1 is the identity channel), realized with Kraus operators proportional to
        the dloc**4 products of the local Weyl operators.
    dloc : int
        Local dimension, dloc >= 3. The inserted local unitary D_{2,0} (x) D_{2,0}
        is fixed in terms of the Weyl index 2, which requires a local dimension of
        at least 3; this function is defined for qutrits and qudits of dimension 3
        or higher, not for qubits (dloc = 2) or a trivial single level (dloc = 1).

    Returns
    -------
    rho_f : np.ndarray
        (dloc*dloc, dloc*dloc) complex array, the normalized output state of the
        postselected switch branch with the larger negativity, for the maximally
        entangled two-qutrit input, the inserted local unitary D_{2,0} (x) D_{2,0}, and
        control phase 0. If one control outcome has exactly zero postselection
        probability, the other (valid) branch is returned directly without a
        negativity comparison.

    Raises
    ------
    ValueError
        If lam2 is not in [0, 1], if dloc < 3, or if both control outcomes have
        zero postselection probability.
    """
    return rho_f

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _qutrit_weyl_ops(dloc: int) -> "dict":
    ops = {}
    om = np.exp(2j * np.pi / dloc)
    X = np.zeros((dloc, dloc))
    for k in range(dloc):
        X[(k + 1) % dloc, k] = 1.0
    Z = np.diag([om ** k for k in range(dloc)])
    for m in range(dloc):
        for n in range(dloc):
            ops[(m, n)] = np.linalg.matrix_power(X, m) @ np.linalg.matrix_power(Z, n)
    return ops


def _build_conditional(pm: str, phi: float, Ks1: "list", Ks2: "list", U: "np.ndarray", rho_AB: "np.ndarray", dAB: int) -> "np.ndarray":
    total = np.zeros((dAB, dAB), dtype=complex)
    for Ki in Ks1:
        for Kj in Ks2:
            Hfwd = Kj @ U @ Ki
            Hbwd = Ki @ U @ Kj
            if pm == "+":
                M = 0.5 * (Hfwd + np.exp(-1j * phi) * Hbwd)
            else:
                M = 0.5 * (Hfwd - np.exp(-1j * phi) * Hbwd)
            total = total + M @ rho_AB @ M.conj().T
    return total


def _oracle_quantum_switch_output(Ks_est: "np.ndarray", lam2: float, dloc: int) -> "np.ndarray":
    if not (0 <= lam2 <= 1):
        raise ValueError("lam2 must be in [0, 1]")
    if dloc < 3:
        raise ValueError("dloc must be at least 3 (the inserted unitary needs Weyl index 2)")
    dA, dB = dloc, dloc
    dAB = dA * dB

    Ks1 = [np.kron(np.eye(dA), K) for K in Ks_est]

    Dops = _qutrit_weyl_ops(dloc)
    p2 = ((dloc ** 4 - 1) * lam2 + 1) / dloc ** 4
    Ks2 = [np.sqrt(p2) * np.eye(dAB)]
    for m in range(dloc):
        for n in range(dloc):
            for mp in range(dloc):
                for npp in range(dloc):
                    if m == 0 and n == 0 and mp == 0 and npp == 0:
                        continue
                    coeff = np.sqrt((1 - p2) / (dloc ** 4 - 1))
                    Ks2.append(coeff * np.kron(Dops[(m, n)], Dops[(mp, npp)]))

    psi = np.zeros(dAB, dtype=complex)
    for i in range(dloc):
        psi[i * dloc + i] = 1 / np.sqrt(dloc)
    rho_AB = np.outer(psi, psi.conj())

    U = np.kron(Dops[(2, 0)], Dops[(2, 0)])

    rho_plus = _build_conditional("+", 0.0, Ks1, Ks2, U, rho_AB, dAB)
    rho_minus = _build_conditional("-", 0.0, Ks1, Ks2, U, rho_AB, dAB)
    P_plus = np.trace(rho_plus).real
    P_minus = np.trace(rho_minus).real

    # A control outcome can have exactly zero postselection probability (e.g. at
    # lam2 = 1 with an identity estimated channel); normalizing and scoring that
    # branch is meaningless, so skip it and return the other, valid branch directly.
    branches = [(P_plus, rho_plus), (P_minus, rho_minus)]
    valid = [(p, r) for p, r in branches if p > 1e-12]
    if not valid:
        raise ValueError("both control outcomes have zero postselection probability")
    if len(valid) == 1:
        p_only, rho_only = valid[0]
        return rho_only / p_only

    rho_plus_norm = rho_plus / P_plus
    rho_minus_norm = rho_minus / P_minus

    N_plus = _oracle_entanglement_negativity(rho_plus_norm, dA, dB)
    N_minus = _oracle_entanglement_negativity(rho_minus_norm, dA, dB)

    if N_plus >= N_minus:
        return rho_plus_norm
    return rho_minus_norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: estimated channel is the identity, moderate depolarizing noise ---
        {
            "setup": """
import numpy as np
Ks_est = np.array([np.eye(3, dtype=complex)])
lam2 = 0.2
dloc = 3
def summarize(fn):
    rho_f = fn(Ks_est.copy(), lam2, dloc)
    trace_row = np.full(rho_f.shape[0], np.trace(rho_f).real)
    eigs = np.linalg.eigvalsh((rho_f+rho_f.conj().T)/2)
    return np.round(np.vstack([trace_row, eigs]), 6)
""",
            "call": "summarize(quantum_switch_output)",
            "gold_call": "summarize(_oracle_quantum_switch_output)",
        },
        # --- Boundary: a configuration where the minus-outcome branch has the larger
        #     negativity (the opposite of the actual task instance), exercising the
        #     oracle's other selection branch ---
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

def _build_generators(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    H0 = (A + A.conj().T)/2
    L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    return H0, L0

def _lindblad_superop(H, Ls):
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j*(np.kron(I,H) - np.kron(H.T,I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(),L) - 0.5*(np.kron(I,L.conj().T@L) + np.kron((L.conj().T@L).T,I))
    return Lsup

def _apply_superop(Esup, rho):
    d = rho.shape[0]
    return (Esup @ rho.reshape(-1,1,order="F")).reshape(d,d,order="F")

H0, L0 = _build_generators(98083)
alpha_hat, beta_hat = 0.35668335954258135, 1.4708278463687552
H_hat, L_hat = alpha_hat*H0, beta_hat*L0
tau = 0.3389448773616492
d = 3
Esup = expm(tau*_lindblad_superop(H_hat, [L_hat]))
J = np.zeros((d*d,d*d), dtype=complex)
for i in range(d):
    for j in range(d):
        Eij = np.zeros((d,d), dtype=complex); Eij[i,j]=1.0
        out = _apply_superop(Esup, Eij)
        J[i*d:(i+1)*d, j*d:(j+1)*d] = out
J = (J+J.conj().T)/2
w,v = np.linalg.eigh(J)
w_clipped = np.clip(w,0,None)
Ks_list = []
for idx in range(len(w_clipped)):
    if w_clipped[idx] > 1e-8:
        Ks_list.append(np.sqrt(w_clipped[idx])*v[:,idx].reshape(d,d,order="F"))
Ks_est = np.array(Ks_list)
lam2 = 0.32921768800306006
dloc = 3
def summarize(fn):
    rho_f = fn(Ks_est.copy(), lam2, dloc)
    trace_row = np.full(rho_f.shape[0], np.trace(rho_f).real)
    eigs = np.linalg.eigvalsh((rho_f+rho_f.conj().T)/2)
    return np.round(np.vstack([trace_row, eigs]), 6)
""",
            "call": "summarize(quantum_switch_output)",
            "gold_call": "summarize(_oracle_quantum_switch_output)",
        },
        # --- Edge: the actual task instance's estimated channel Kraus operators ---
        {
            "setup": """
import numpy as np
from scipy.linalg import expm

def _build_generators(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    H0 = (A + A.conj().T)/2
    L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))
    return H0, L0

def _lindblad_superop(H, Ls):
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j*(np.kron(I,H) - np.kron(H.T,I))
    for L in Ls:
        Lsup = Lsup + np.kron(L.conj(),L) - 0.5*(np.kron(I,L.conj().T@L) + np.kron((L.conj().T@L).T,I))
    return Lsup

def _apply_superop(Esup, rho):
    d = rho.shape[0]
    return (Esup @ rho.reshape(-1,1,order="F")).reshape(d,d,order="F")

H0, L0 = _build_generators(101)
alpha_hat, beta_hat = 1.89302208806816, 0.7232072266302545
H_hat, L_hat = alpha_hat*H0, beta_hat*L0
tau = 0.3
d = 3
Esup = expm(tau*_lindblad_superop(H_hat, [L_hat]))
J = np.zeros((d*d,d*d), dtype=complex)
for i in range(d):
    for j in range(d):
        Eij = np.zeros((d,d), dtype=complex); Eij[i,j]=1.0
        out = _apply_superop(Esup, Eij)
        J[i*d:(i+1)*d, j*d:(j+1)*d] = out
J = (J+J.conj().T)/2
w,v = np.linalg.eigh(J)
w_clipped = np.clip(w,0,None)
Ks_list = []
for idx in range(len(w_clipped)):
    if w_clipped[idx] > 1e-8:
        Ks_list.append(np.sqrt(w_clipped[idx])*v[:,idx].reshape(d,d,order="F"))
Ks_est = np.array(Ks_list)
lam2 = 0.35
dloc = 3
def summarize(fn):
    rho_f = fn(Ks_est.copy(), lam2, dloc)
    trace_row = np.full(rho_f.shape[0], np.trace(rho_f).real)
    return np.round(np.vstack([trace_row, rho_f.real, rho_f.imag]), 5)
""",
            "call": "summarize(quantum_switch_output)",
            "gold_call": "summarize(_oracle_quantum_switch_output)",
        },
        # --- Boundary: lam2=1 (identity second channel) with an identity estimated
        #     channel makes one control outcome's postselection probability exactly
        #     zero (P_minus = 0.0); the oracle must skip that branch rather than
        #     dividing by zero, returning the other, valid, normalized branch ---
        {
            "setup": """
import numpy as np
Ks_est = np.array([np.eye(3, dtype=complex)])
lam2 = 1.0
dloc = 3
def summarize(fn):
    rho_f = fn(Ks_est.copy(), lam2, dloc)
    return np.round(np.array([np.trace(rho_f).real, np.max(np.abs(rho_f.imag - rho_f.imag))]), 6)
""",
            "call": "summarize(quantum_switch_output)",
            "gold_call": "summarize(_oracle_quantum_switch_output)",
            "tol": 1e-6,
        },
        # --- Edge: dloc below the qutrit minimum the fixed unitary requires ---
        {
            "setup": """
import numpy as np
Ks_est = np.array([np.eye(2, dtype=complex)])
lam2 = 0.5
dloc = 2
def run_model():
    try:
        quantum_switch_output(Ks_est.copy(), lam2, dloc)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def run_gold():
    try:
        _oracle_quantum_switch_output(Ks_est.copy(), lam2, dloc)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
