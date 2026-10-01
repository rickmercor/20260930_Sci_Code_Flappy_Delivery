"""
Step 3 — Generalized Fock matrix of the paired-bond reference.

For a seniority-zero state the generalized Fock matrix reduces to

    f_pq = h_pq n_p + (1/2) sum_r (2 V_rrpq - V_rqpr) D_rp + sum_r V_rprq P_rp,

built from the reference's seniority-conserving two-body data:

  - direct elements D_rp = n_r n_p for orbitals in different bond units,

    D_rp = 0 when r and p share a unit, and D_pp = 0 on the diagonal;

  - pair-transfer elements P_pp = n_p on the diagonal,

    P_{a0,a1} = P_{a1,a0} = -1/eta_a inside each unit, and zero otherwise.

The intra-unit pair element -1/eta_a = -sqrt(n_{a0} n_{a1}) is the coherence

of the electron pair shared by the bonding and antibonding orbitals; omitting

it gives a plain occupation-weighted Fock matrix and wrong single-excitation

couplings downstream. f is NOT symmetric unless the orbitals are variationally

optimal; here the integrals are fixed, so the asymmetry f_pq - f_qp is

meaningful data. Occupations follow from omega as in Step 2.



Inputs: h, V, omega. Returns the full matrix f, shape (2M, 2M).



Raises

------

ValueError is raised, and only ValueError, when any of the following holds:

  - h must be a square matrix

  - h must have even dimension 2M with M >= 1

  - V must have shape (2M, 2M, 2M, 2M) matching h

  - integrals must be finite

  - h must be symmetric

  - V must have the 8-fold real-orbital permutation symmetry

  - omega must be a length-M vector of bond-unit gaps

  - omega must be finite

Returns
-------
np.ndarray of shape (2M, 2M), the generalized Fock matrix as floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_generalized_fock(h, V, omega):
    '''Generalized Fock matrix f_pq of the paired-bond reference.

    Parameters
    ----------
    h : np.ndarray
        (2M, 2M) symmetric one-electron integrals.
    V : np.ndarray
        (2M, 2M, 2M, 2M) two-electron integrals (pq|rs).
    omega : np.ndarray
        (M,) bond-unit gap parameters.

    Returns
    -------
    f : np.ndarray
        (2M, 2M) generalized Fock matrix (generally non-symmetric).
    '''
    return f

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _validate_system(h, V):
    """Validate the integral arrays; return (h, V, M) with M bond units."""
    h = np.asarray(h, dtype=float)
    V = np.asarray(V, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("h must be a square matrix")
    n_orb = h.shape[0]
    if n_orb < 2 or n_orb % 2 != 0:
        raise ValueError("h must have even dimension 2M with M >= 1")
    if V.shape != (n_orb, n_orb, n_orb, n_orb):
        raise ValueError("V must have shape (2M, 2M, 2M, 2M) matching h")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(V))):
        raise ValueError("integrals must be finite")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be symmetric")
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not np.allclose(V, np.transpose(V, perm), rtol=0.0, atol=1e-10):
            raise ValueError("V must have the 8-fold real-orbital permutation symmetry")
    return h, V, n_orb // 2

def _validate_gaps(omega, M):
    omega = np.asarray(omega, dtype=float)
    if omega.shape != (M,):
        raise ValueError("omega must be a length-M vector of bond-unit gaps")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    return omega

def _pair_integrals(V):
    """Direct J_pq = V_ppqq, exchange K_pq = V_pqqp, pair-transfer L_pq = V_pqpq,
    and the combination G = 2J - K."""
    idx = np.arange(V.shape[0])
    J = V[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    K = V[idx[:, None], idx[None, :], idx[None, :], idx[:, None]]
    L = V[idx[:, None], idx[None, :], idx[:, None], idx[None, :]]
    return J, K, L, 2.0 * J - K

def _occupations(omega):
    """Occupations n_{a,mu} = 1 + (-1)^mu * w_a/eta_a, eta_a = sqrt(w_a^2 + 1)."""
    omega = np.asarray(omega, dtype=float)
    eta = np.sqrt(omega * omega + 1.0)
    n = np.empty(2 * omega.size)
    n[0::2] = 1.0 + omega / eta
    n[1::2] = 1.0 - omega / eta
    return n, eta

def _orbital_energies(h, L, G, n):
    """eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's bond unit} G_pq n_q."""
    n_orb = h.shape[0]
    eps = np.empty(n_orb)
    for p in range(n_orb):
        s = h[p, p] + 0.5 * L[p, p]
        for q in range(n_orb):
            if q // 2 != p // 2:
                s += 0.5 * G[p, q] * n[q]
        eps[p] = s
    return eps

def _generalized_fock(h, V, omega):
    """f_pq = h_pq n_p + (1/2) sum_r (2 V_rrpq - V_rqpr) D_rp + sum_r V_rprq P_rp,
    with the reference two-body data D_pp = 0, D within a unit = 0, D_rp = n_r n_p
    otherwise, P_pp = n_p, and P_{a0,a1} = P_{a1,a0} = -1/eta_a."""
    n_orb = h.shape[0]
    M = n_orb // 2
    n, eta = _occupations(omega)
    D = np.outer(n, n)
    for a in range(M):
        D[2 * a, 2 * a + 1] = D[2 * a + 1, 2 * a] = 0.0
    np.fill_diagonal(D, 0.0)
    P = np.zeros((n_orb, n_orb))
    np.fill_diagonal(P, n)
    for a in range(M):
        P[2 * a, 2 * a + 1] = P[2 * a + 1, 2 * a] = -1.0 / eta[a]
    f = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            s = h[p, q] * n[p]
            for r in range(n_orb):
                s += 0.5 * (2.0 * V[r, r, p, q] - V[r, q, p, r]) * D[r, p]
                s += V[r, p, r, q] * P[r, p]
            f[p, q] = s
    return f

def _oracle_pair_generalized_fock(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _generalized_fock(h, V, omega)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(17, 3)\nomega = np.array([0.55, 0.90, 0.40])\n",
            "call": "pair_generalized_fock(h, V, omega)",
            "gold_call": "_oracle_pair_generalized_fock(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(3, 2)\nomega = np.array([0.60, 0.90])\n",
            "call": "pair_generalized_fock(h, V, omega)",
            "gold_call": "_oracle_pair_generalized_fock(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(29, 1)\nomega = np.array([0.75])\n",
            "call": "pair_generalized_fock(h, V, omega)",
            "gold_call": "_oracle_pair_generalized_fock(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(17, 3)\n\ndef run_model():\n    try:\n        pair_generalized_fock(h, V, np.array([0.5, np.nan, 0.7]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_pair_generalized_fock(h, V, np.array([0.5, np.nan, 0.7]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
