"""
Step 2 — Energy of the paired-bond reference state.

With stationary (or given) gap parameters w_a, occupations

n_{a,mu} = 1 + (-1)^mu w_a/eta_a and eta_a = sqrt(w_a^2 + 1), the reference

energy is a pure occupation-number functional,

    E_ref = sum_p eps_p n_p - sum_a L_{a0,a1} * sqrt(n_{a0} n_{a1})

            - (1/2) sum'_{p<q} G_pq n_p n_q,

where eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's unit} G_pq n_q, the

primed sum excludes orbital pairs belonging to the same bond unit, and

sqrt(n_{a0} n_{a1}) = 1/eta_a. The middle term is the intra-unit pair-transfer

(pairing) stabilization; the last term removes the double counting introduced

by summing occupation-weighted orbital energies, exactly as in mean-field

theory. No nuclear-repulsion constant is added.



Inputs: h, V as in Step 1, plus the gap vector omega of shape (M,).

Returns E_ref as a float (Hartree).



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
float, the reference energy E_ref in Hartree as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reference_pair_energy(h, V, omega):
    '''Energy of the paired-bond reference for gap parameters omega.

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
    energy : float
        The reference energy E_ref in Hartree.
    '''
    return energy

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

def _reference_energy(h, V, omega):
    """E_ref = sum_p eps_p n_p - sum_a L_{a0,a1}/eta_a
    - (1/2) sum'_{p<q} G_pq n_p n_q  (' excludes same-unit pairs)."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    energy = float(np.dot(eps, n))
    for a in range(M):
        energy -= L[2 * a, 2 * a + 1] / eta[a]
    for p in range(2 * M):
        for q in range(p + 1, 2 * M):
            if q // 2 != p // 2:
                energy -= 0.5 * G[p, q] * n[p] * n[q]
    return energy

def _oracle_reference_pair_energy(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return float(_reference_energy(h, V, omega))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(7, 3)\nomega = np.array([0.45, 0.80, 1.10])\n",
            "call": "reference_pair_energy(h, V, omega)",
            "gold_call": "_oracle_reference_pair_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(5, 2)\nomega = np.array([0.0, 2.5])\n",
            "call": "reference_pair_energy(h, V, omega)",
            "gold_call": "_oracle_reference_pair_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(13, 3)\nomega = np.array([8.0, 0.30, 1.20])\n",
            "call": "reference_pair_energy(h, V, omega)",
            "gold_call": "_oracle_reference_pair_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(7, 3)\n\ndef run_model():\n    try:\n        reference_pair_energy(h, V, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_reference_pair_energy(h, V, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
