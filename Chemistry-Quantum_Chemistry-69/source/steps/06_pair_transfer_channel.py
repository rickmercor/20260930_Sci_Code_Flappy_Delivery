"""
Step 6 — Second-order sum over the correlated pair-transfer channel.

Pair transfers replace the double electron-transfers of determinant-based

theories: a whole electron pair migrates between bond units as a correlated

object. Intermediates as in Step 4; x marks unit partner orbitals. Families:

 (i) seniority-conserving transfer a -> b (ordered a != b; unit a emptied,

     unit b filled): energy eta_a L_{a0,a1} + eta_b L_{b0,b1} + eps_{b0}

     + eps_{b1} - eps_{a0} - eps_{a1} + 2 G_{b0,b1} - g_{a,b} + g--_{a,b}

     - g-_{a,b0} - g-_{a,b1} + g-_{b,a0} + g-_{b,a1}, where

     g_{a,b} = (G_{a0,b0} + G_{a0,b1} + G_{a1,b0} + G_{a1,b1})/2; coupling

     (1/2) sum_{mu,nu} (-1)^(mu+nu+1) L_{a_mu,b_nu}

     sqrt(n_{a_mu} n_{b_(1-nu)});

 (ii) open-shell pair (a_mu, b_nu) scattered into unit g (g filled; partner

     orbitals a_x, b_x emptied; a < b, g distinct): with

     B = eta_a L_{a0,a1} + eta_b L_{b0,b1} + eta_g L_{g0,g1} + t_{a_mu,b_nu}

       + g--_{a,b} + g--_{a,g} + g--_{b,g} + G_{a_x,b_x}/2 - G_{a_x,g0}/2

       - G_{a_x,g1}/2 - G_{b_x,g0}/2 - G_{b_x,g1}/2 and

     C = eps_{g0} + eps_{g1} - eps_{a_x} - eps_{b_x} - g-_{a,g0} - g-_{a,g1}

       - g-_{b,g0} - g-_{b,g1} + g-_{b,a_x} + g-_{g,a_x} + g-_{a,b_x}

       + g-_{g,b_x},

     the energy is B + C + 2 G_{g0,g1} and the coupling

     (1/2)(-1)^(mu+nu+1) sqrt(n_{a_mu} n_{b_nu}) (V_{a_mu,g1,b_nu,g1}

     sqrt(n_{g0}) - V_{a_mu,g0,b_nu,g0} sqrt(n_{g1}));

 (iii) the reverse: unit g emptied into the open-shell pair (a_mu, b_nu) with

     partners a_x, b_x filled: energy B - C + G_{a0,a1} + G_{b0,b1}; coupling

     (1/2)(-1)^(mu+nu) sqrt(n_{a_x} n_{b_x}) (V_{g0,a_mu,g0,b_nu} sqrt(n_{g0})

     - V_{g1,a_mu,g1,b_nu} sqrt(n_{g1}));

 (iv) seniority-four transfers (a_mu, b_nu) -> (g_lam, d_kap) over four

     distinct units (a < b source, g < d target): energy

       sum of the four eta L terms + t_{a_mu,b_nu} + t_{g_lam,d_kap}

       + eps_{g_x} + eps_{d_x} - eps_{a_x} - eps_{b_x} + G_{g0,g1} + G_{d0,d1}

       + all six g-- terms + G_{a_x,b_x}/2 + G_{g_x,d_x}/2

       - g-_{a,g_x} - g-_{b,g_x} - g-_{d,g_x} - g-_{a,d_x} - g-_{b,d_x}

       - g-_{g,d_x} + g-_{b,a_x} + g-_{g,a_x} + g-_{d,a_x} + g-_{a,b_x}

       + g-_{g,b_x} + g-_{d,b_x} - G_{a_x,g_x}/2 - G_{a_x,d_x}/2

       - G_{b_x,g_x}/2 - G_{b_x,d_x}/2;

     coupling (1/4)(-1)^(mu+nu+lam+kap+1)

     sqrt(n_{a_mu} n_{b_nu} n_{g_x} n_{d_x}) (V_{a_mu,g_lam,b_nu,d_kap}

     + V_{a_mu,d_kap,b_nu,g_lam});

 (v) the complementary seniority-four partner of (iv): energy of (iv) plus

     K_{a_mu,g_lam} + K_{a_mu,d_kap} + K_{b_nu,g_lam} + K_{b_nu,d_kap}

     - 2 K_{a_mu,b_nu} - 2 K_{g_lam,d_kap}; coupling

     (sqrt(3)/4)(-1)^(mu+nu+lam+kap) sqrt(n_{a_mu} n_{b_nu} n_{g_x} n_{d_x})

     (V_{a_mu,g_lam,b_nu,d_kap} - V_{a_mu,d_kap,b_nu,g_lam}).

Sum -|coupling|^2/energy over all of (i)-(v); ValueError on a vanishing

energy. Families (ii)-(iii) need M >= 3; (iv)-(v) need M >= 4.



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

  - vanishing excitation energy in a perturbative denominator

Returns
-------
float, the pair-transfer-channel second-order energy sum in Hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_transfer_correction(h, V, omega):
    '''Second-order sum over all correlated pair-transfer excitations.

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
    e2 : float
        Second-order energy sum of the pair-transfer channel.
    '''
    return e2

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

def _pair_transfer_terms(h, V, omega):
    """(excitation energy, coupling) for every correlated pair-transfer state."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.permutations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        d_e = (eta[a] * L[a0, a1] + eta[b] * L[b0, b1]
               + eps[b0] + eps[b1] - eps[a0] - eps[a1] + 2.0 * G[b0, b1]
               - _g_sym(G, a, b) + _g_mm(G, omega, eta, a, b)
               - _g_minus(G, omega, eta, a, b0) - _g_minus(G, omega, eta, a, b1)
               + _g_minus(G, omega, eta, b, a0) + _g_minus(G, omega, eta, b, a1))
        coup = 0.5 * sum((-1.0) ** (mu + nu + 1) * L[2 * a + mu, 2 * b + nu]
                         * sn[2 * a + mu] * sn[2 * b + 1 - nu]
                         for mu in (0, 1) for nu in (0, 1))
        terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                for g in range(M):
                    if g in (a, b):
                        continue
                    g0, g1 = 2 * g, 2 * g + 1
                    part_b = (eta[a] * L[2 * a, 2 * a + 1]
                              + eta[b] * L[2 * b, 2 * b + 1] + eta[g] * L[g0, g1]
                              + _t_open_shell(J, K, L, G, am, bn)
                              + _g_mm(G, omega, eta, a, b)
                              + _g_mm(G, omega, eta, a, g)
                              + _g_mm(G, omega, eta, b, g)
                              + 0.5 * G[ax, bx]
                              - 0.5 * G[ax, g0] - 0.5 * G[ax, g1]
                              - 0.5 * G[bx, g0] - 0.5 * G[bx, g1])
                    part_c = (eps[g0] + eps[g1] - eps[ax] - eps[bx]
                              - _g_minus(G, omega, eta, a, g0)
                              - _g_minus(G, omega, eta, a, g1)
                              - _g_minus(G, omega, eta, b, g0)
                              - _g_minus(G, omega, eta, b, g1)
                              + _g_minus(G, omega, eta, b, ax)
                              + _g_minus(G, omega, eta, g, ax)
                              + _g_minus(G, omega, eta, a, bx)
                              + _g_minus(G, omega, eta, g, bx))
                    # open-shell pair (a_mu, b_nu) scattered into a filled unit g
                    d_e = part_b + part_c + 2.0 * G[g0, g1]
                    coup = (0.5 * (-1.0) ** (mu + nu + 1) * sn[am] * sn[bn]
                            * (V[am, g1, bn, g1] * sn[g0] - V[am, g0, bn, g0] * sn[g1]))
                    terms.append((d_e, coup))
                    # unit g emptied into an open-shell pair (a_mu, b_nu)
                    d_e = (part_b - part_c + G[2 * a, 2 * a + 1]
                           + G[2 * b, 2 * b + 1])
                    coup = (0.5 * (-1.0) ** (mu + nu) * sn[ax] * sn[bx]
                            * (V[g0, am, g0, bn] * sn[g0] - V[g1, am, g1, bn] * sn[g1]))
                    terms.append((d_e, coup))
    for a, b in itertools.combinations(range(M), 2):
        rest = [u for u in range(M) if u not in (a, b)]
        for g, d in itertools.combinations(rest, 2):
            for mu, nu, lam, kap in itertools.product((0, 1), repeat=4):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                gl, gx = 2 * g + lam, 2 * g + 1 - lam
                dk, dx = 2 * d + kap, 2 * d + 1 - kap
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + eta[g] * L[2 * g, 2 * g + 1] + eta[d] * L[2 * d, 2 * d + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + _t_open_shell(J, K, L, G, gl, dk)
                       + eps[gx] + eps[dx] - eps[ax] - eps[bx]
                       + G[2 * g, 2 * g + 1] + G[2 * d, 2 * d + 1]
                       + _g_mm(G, omega, eta, a, b) + _g_mm(G, omega, eta, a, g)
                       + _g_mm(G, omega, eta, a, d) + _g_mm(G, omega, eta, b, g)
                       + _g_mm(G, omega, eta, b, d) + _g_mm(G, omega, eta, g, d)
                       + 0.5 * G[ax, bx] + 0.5 * G[gx, dx]
                       - _g_minus(G, omega, eta, a, gx) - _g_minus(G, omega, eta, b, gx)
                       - _g_minus(G, omega, eta, d, gx) - _g_minus(G, omega, eta, a, dx)
                       - _g_minus(G, omega, eta, b, dx) - _g_minus(G, omega, eta, g, dx)
                       + _g_minus(G, omega, eta, b, ax) + _g_minus(G, omega, eta, g, ax)
                       + _g_minus(G, omega, eta, d, ax) + _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, g, bx) + _g_minus(G, omega, eta, d, bx)
                       - 0.5 * G[ax, gx] - 0.5 * G[ax, dx]
                       - 0.5 * G[bx, gx] - 0.5 * G[bx, dx])
                root = sn[am] * sn[bn] * sn[gx] * sn[dx]
                phase = (-1.0) ** (mu + nu + lam + kap)
                coup = 0.25 * (-phase) * root * (V[am, gl, bn, dk] + V[am, dk, bn, gl])
                terms.append((d_e, coup))
                d_e5 = d_e + (K[am, gl] + K[am, dk] + K[bn, gl] + K[bn, dk]
                              - 2.0 * K[am, bn] - 2.0 * K[gl, dk])
                coup5 = (np.sqrt(3.0) / 4.0) * phase * root * (
                    V[am, gl, bn, dk] - V[am, dk, bn, gl])
                terms.append((d_e5, coup5))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def _oracle_pair_transfer_correction(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_pair_transfer_terms(h, V, omega))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(23, 4)\nomega = np.array([1.146, 0.459, 0.744, 0.693])\n",
            "call": "pair_transfer_correction(h, V, omega)",
            "gold_call": "_oracle_pair_transfer_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(5, 2)\nomega = np.array([1.106, 0.846])\n",
            "call": "pair_transfer_correction(h, V, omega)",
            "gold_call": "_oracle_pair_transfer_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(13, 3)\nomega = np.array([0.722, 1.139, 0.405])\n",
            "call": "pair_transfer_correction(h, V, omega)",
            "gold_call": "_oracle_pair_transfer_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(5, 2)\n\ndef run_model():\n    try:\n        pair_transfer_correction(h, V, np.array([[0.5], [0.5]]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_pair_transfer_correction(h, V, np.array([[0.5], [0.5]]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
