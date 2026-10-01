"""
Step 7 — Variational dressing of the reference by the intruder states.

Each doubly-split configuration over units a < b supports two orthogonal

seniority-four states. The complementary double splits (the partners omitted

from Step 5) can fall energetically close to the reference and behave as

intruders, so they are treated variationally: build the (1 + M(M-1)/2)-

dimensional CI matrix over {reference} + {all complementary double splits}

and take its LOWEST eigenvalue.



Matrix elements (intermediates as in Steps 4-5):

  - top-left: E_ref (Step 2);

  - diagonal, pair a<b: E_ref + [double-split energy eta_a L_{a0,a1}

    + eta_b L_{b0,b1} + t_{a0,a1} + t_{b0,b1} + g--_{a,b}]

    + K_{a0,b0} + K_{a0,b1} + K_{a1,b0} + K_{a1,b1} - 2 K_{a0,a1}

    - 2 K_{b0,b1};

  - first row/column, pair a<b:

    -(sqrt(3)/2)(sqrt(n_{a0} n_{b0}) + sqrt(n_{a1} n_{b1})) V_{a0,b1,b0,a1}

    -(sqrt(3)/2)(sqrt(n_{a1} n_{b0}) + sqrt(n_{a0} n_{b1})) V_{a1,b1,b0,a0};

  - between two complementary double splits: nonzero ONLY when the two unit

    pairs share exactly one unit; writing the shared unit first, for pairs

    (s,b) and (s,g) the element is

    -(1/2)(sqrt(n_{b0} n_{g1}) + sqrt(n_{b1} n_{g0})) V_{g0,b1,b0,g1}

    -(1/2)(sqrt(n_{b0} n_{g0}) + sqrt(n_{b1} n_{g1})) V_{g0,b0,b1,g1};

    disjoint pairs give a strictly zero element.

Diagonalize (symmetric eigensolver) and return the lowest eigenvalue: the

intruder-dressed reference energy. For M = 1 the matrix is 1x1 and the result

is E_ref itself.



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
float, the lowest eigenvalue of the dressed CI matrix in Hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dressed_reference_energy(h, V, omega):
    '''Lowest eigenvalue of the reference + complementary-double-split CI.

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
        The intruder-dressed reference energy (lowest CI eigenvalue).
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools

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

def _t_open_shell(J, K, L, G, p, q):
    """Blocked-singlet placement energy t_pq: J_pq + K_pq - L_pp/2 - L_qq/2,
    with an extra -G_pq/2 when p and q sit in different bond units."""
    val = J[p, q] + K[p, q] - 0.5 * L[p, p] - 0.5 * L[q, q]
    if p // 2 != q // 2:
        val -= 0.5 * G[p, q]
    return val

def _g_minus(G, omega, eta, a, p):
    """g-_{a,p} = (w_a / (2 eta_a)) (G_{a0,p} - G_{a1,p})."""
    return 0.5 * (omega[a] / eta[a]) * (G[2 * a, p] - G[2 * a + 1, p])

def _g_mm(G, omega, eta, a, b):
    """g--_{a,b} = (w_a w_b / (2 eta_a eta_b))
    (G_{a0,b0} - G_{a0,b1} - G_{a1,b0} + G_{a1,b1})."""
    return 0.5 * (omega[a] * omega[b] / (eta[a] * eta[b])) * (
        G[2 * a, 2 * b] - G[2 * a, 2 * b + 1]
        - G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _g_sym(G, a, b):
    """g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1}) / 2."""
    return 0.5 * (G[2 * a, 2 * b] + G[2 * a, 2 * b + 1]
                  + G[2 * a + 1, 2 * b] + G[2 * a + 1, 2 * b + 1])

def _dressed_matrix(h, V, omega):
    """CI matrix over {reference} + all complementary double splits."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    sn = np.sqrt(n)
    e_ref = _reference_energy(h, V, omega)
    pairs = list(itertools.combinations(range(M), 2))
    dim = 1 + len(pairs)
    mat = np.zeros((dim, dim))
    mat[0, 0] = e_ref
    for i, (a, b) in enumerate(pairs, start=1):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1)
               + _g_mm(G, omega, eta, a, b)
               + K[a0, b0] + K[a0, b1] + K[a1, b0] + K[a1, b1]
               - 2.0 * K[a0, a1] - 2.0 * K[b0, b1])
        mat[i, i] = e_ref + d_e
        coup = (-0.5 * np.sqrt(3.0) * (sn[a0] * sn[b0] + sn[a1] * sn[b1])
                * V[a0, b1, b0, a1]
                - 0.5 * np.sqrt(3.0) * (sn[a1] * sn[b0] + sn[a0] * sn[b1])
                * V[a1, b1, b0, a0])
        mat[0, i] = mat[i, 0] = coup
    for i, pi in enumerate(pairs, start=1):
        for j, pj in enumerate(pairs, start=1):
            if j <= i:
                continue
            shared = set(pi) & set(pj)
            if len(shared) != 1:
                continue
            be = (set(pi) - shared).pop()
            ga = (set(pj) - shared).pop()
            b0, b1 = 2 * be, 2 * be + 1
            g0, g1 = 2 * ga, 2 * ga + 1
            val = (-0.5 * (sn[b0] * sn[g1] + sn[b1] * sn[g0]) * V[g0, b1, b0, g1]
                   - 0.5 * (sn[b0] * sn[g0] + sn[b1] * sn[g1]) * V[g0, b0, b1, g1])
            mat[i, j] = mat[j, i] = val
    return mat

def _oracle_dressed_reference_energy(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    mat = _dressed_matrix(h, V, omega)
    return float(np.linalg.eigvalsh(mat)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(7, 3)\nomega = np.array([0.418, 0.666, 0.958])\n",
            "call": "dressed_reference_energy(h, V, omega)",
            "gold_call": "_oracle_dressed_reference_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(3, 1)\nomega = np.array([0.618])\n",
            "call": "dressed_reference_energy(h, V, omega)",
            "gold_call": "_oracle_dressed_reference_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(29, 4)\nomega = np.array([0.955, 1.298, 0.744, 0.528])\n",
            "call": "dressed_reference_energy(h, V, omega)",
            "gold_call": "_oracle_dressed_reference_energy(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(7, 3)\nomega = np.array([0.5, 0.6, 0.7])\n_, Vsmall = _mk_system(7, 2)\n\ndef run_model():\n    try:\n        dressed_reference_energy(h, Vsmall, omega)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_dressed_reference_energy(h, Vsmall, omega)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
