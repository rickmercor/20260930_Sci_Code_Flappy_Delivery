"""
Step 5 — Second-order sum over the double-excitation channel.

Doubles are simultaneous pairs of the Step-4 singles, with correlated energy

updates. Intermediates as in Step 4. The families and their data (unit

indices a, b, g all distinct; x marks the unit partner orbital):

 (i) double swaps in units a<b: energy 2 eta_a L_{a0,a1} + 2 eta_b L_{b0,b1}

     + 4 g--_{a,b}; coupling

     (1/(2 eta_a eta_b)) sum_{mu,nu} (-1)^(mu+nu) G_{a_mu,b_nu};

 (ii) swap in a + split in b (ordered a != b): energy 2 eta_a L_{a0,a1}

     + eta_b L_{b0,b1} + t_{b0,b1} + 2 g--_{a,b}; coupling

     ((sqrt(n_{b0}) - sqrt(n_{b1}))/(2 eta_a)) sum_mu (-1)^mu

     (2 V_{a_mu,a_mu,b0,b1} - V_{a_mu,b1,b0,a_mu});

 (iii) double splits in a<b: energy eta_a L_{a0,a1} + eta_b L_{b0,b1}

     + t_{a0,a1} + t_{b0,b1} + g--_{a,b}; coupling

     (sqrt(n_{a0})-sqrt(n_{a1}))(sqrt(n_{b0})-sqrt(n_{b1})) V_{a0,a1,b0,b1}

     - (1/2)(sqrt(n_{a0} n_{b0}) + sqrt(n_{a1} n_{b1})) V_{a0,b1,b0,a1}

     + (1/2)(sqrt(n_{a1} n_{b0}) + sqrt(n_{a0} n_{b1})) V_{a1,b1,b0,a0}.

     Each doubly-split configuration also supports a second, orthogonal

     seniority-four state (the complementary double split). Those states are

     intruder-prone and are EXCLUDED from this perturbative sum: they are

     treated variationally in Step 7.

 (iv) swap in a + transfer b_nu -> g_lam: energy 2 eta_a L_{a0,a1} + [transfer

     energy core: eta_b L_{b0,b1} + eta_g L_{g0,g1} + t_{b_nu,g_lam} +

     eps_{g_x} - eps_{b_x} + G_{g0,g1} - G_{b_x,g_x}/2] + 2 g--_{a,b}

     + 2 g--_{a,g} + g--_{b,g} - 2 g-_{a,g_x} - g-_{b,g_x} + 2 g-_{a,b_x}

     + g-_{g,b_x}; coupling ((-1)^(nu+lam+1)/(2 sqrt(2) eta_a))

     sqrt(n_{b_nu} n_{g_x}) sum_mu (-1)^mu (2 V_{b_nu,g_lam,a_mu,a_mu}

     - V_{a_mu,g_lam,b_nu,a_mu});

 (v) split in a + transfer b_nu -> g_lam: energy eta_a L_{a0,a1} + t_{a0,a1}

     + [transfer core] + g--_{a,b} + g--_{a,g} + g--_{b,g} - g-_{a,g_x}

     - g-_{b,g_x} + g-_{a,b_x} + g-_{g,b_x}; coupling

     ((-1)^(nu+lam+1)/(2 sqrt(2))) sqrt(n_{b_nu} n_{g_x}) sum_mu (-1)^mu

     sqrt(n_{a_mu}) (2 V_{b_nu,g_lam,a_mu,a_(1-mu)} -

     V_{a_mu,g_lam,b_nu,a_(1-mu)});

 (vi) the complementary partner of (v): energy of (v) plus

     K_{a0,b_nu} + K_{a0,g_lam} + K_{a1,b_nu} + K_{a1,g_lam} - 2 K_{a0,a1}

     - 2 K_{b_nu,g_lam}; coupling ((-1)^(nu+lam)/2) sqrt(3/2)

     sqrt(n_{b_nu} n_{g_x}) sum_mu sqrt(n_{a_mu}) V_{a_mu,g_lam,b_nu,a_(1-mu)}.

Sum -|coupling|^2/energy over families (i)-(vi); ValueError on a vanishing

energy. Families (iv)-(vi) need M >= 3 and are absent for M = 2.



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
float, the double-channel second-order energy sum in Hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def double_excitation_correction(h, V, omega):
    '''Second-order sum over doubles, excluding complementary double splits.

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
        Second-order energy sum of the retained double excitations.
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

def _double_terms(h, V, omega):
    """(excitation energy, coupling) for every double excitation EXCEPT the
    complementary double splits (those are dressed variationally instead)."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    sn = np.sqrt(n)
    terms = []
    for a, b in itertools.combinations(range(M), 2):
        a0, a1, b0, b1 = 2 * a, 2 * a + 1, 2 * b, 2 * b + 1
        eta_la = eta[a] * L[a0, a1]
        eta_lb = eta[b] * L[b0, b1]
        gmm_ab = _g_mm(G, omega, eta, a, b)
        # double swap
        coup = sum((-1.0) ** (mu + nu) * G[2 * a + mu, 2 * b + nu]
                   for mu in (0, 1) for nu in (0, 1)) / (2.0 * eta[a] * eta[b])
        terms.append((2.0 * eta_la + 2.0 * eta_lb + 4.0 * gmm_ab, coup))
        # double split (its complementary partner is dressed, not summed)
        d_e = (eta_la + eta_lb + _t_open_shell(J, K, L, G, a0, a1)
               + _t_open_shell(J, K, L, G, b0, b1) + gmm_ab)
        coup = ((sn[a0] - sn[a1]) * (sn[b0] - sn[b1]) * V[a0, a1, b0, b1]
                - 0.5 * (sn[a0] * sn[b0] + sn[a1] * sn[b1]) * V[a0, b1, b0, a1]
                + 0.5 * (sn[a1] * sn[b0] + sn[a0] * sn[b1]) * V[a1, b1, b0, a0])
        terms.append((d_e, coup))
    for a, b in itertools.permutations(range(M), 2):
        b0, b1 = 2 * b, 2 * b + 1
        # swap in unit a + split in unit b
        d_e = (2.0 * eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[b0, b1]
               + _t_open_shell(J, K, L, G, b0, b1)
               + 2.0 * _g_mm(G, omega, eta, a, b))
        coup = (sn[b0] - sn[b1]) * sum(
            (-1.0) ** mu * (2.0 * V[2 * a + mu, 2 * a + mu, b0, b1]
                            - V[2 * a + mu, b1, b0, 2 * a + mu])
            for mu in (0, 1)) / (2.0 * eta[a])
        terms.append((d_e, coup))
    for a in range(M):
        others = [u for u in range(M) if u != a]
        a0, a1 = 2 * a, 2 * a + 1
        for b, g in itertools.permutations(others, 2):
            for nu in (0, 1):
                for lam in (0, 1):
                    bn, bx = 2 * b + nu, 2 * b + 1 - nu
                    gl, gx = 2 * g + lam, 2 * g + 1 - lam
                    common = (eta[b] * L[2 * b, 2 * b + 1]
                              + eta[g] * L[2 * g, 2 * g + 1]
                              + _t_open_shell(J, K, L, G, bn, gl)
                              + eps[gx] - eps[bx] + G[2 * g, 2 * g + 1]
                              - 0.5 * G[bx, gx])
                    gmm_ab = _g_mm(G, omega, eta, a, b)
                    gmm_ag = _g_mm(G, omega, eta, a, g)
                    gmm_bg = _g_mm(G, omega, eta, b, g)
                    gm_agx = _g_minus(G, omega, eta, a, gx)
                    gm_bgx = _g_minus(G, omega, eta, b, gx)
                    gm_abx = _g_minus(G, omega, eta, a, bx)
                    gm_gbx = _g_minus(G, omega, eta, g, bx)
                    # swap in a + transfer b_nu -> g_lam
                    d_e = (2.0 * eta[a] * L[a0, a1] + common
                           + 2.0 * gmm_ab + 2.0 * gmm_ag + gmm_bg
                           - 2.0 * gm_agx - gm_bgx + 2.0 * gm_abx + gm_gbx)
                    ssum = sum((-1.0) ** mu * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + mu]
                                               - V[2 * a + mu, gl, bn, 2 * a + mu])
                               for mu in (0, 1))
                    coup = ((-1.0) ** (nu + lam + 1)
                            / (2.0 * np.sqrt(2.0) * eta[a])
                            * np.sqrt(n[bn] * n[gx]) * ssum)
                    terms.append((d_e, coup))
                    # split in a + transfer b_nu -> g_lam
                    d_e6 = (eta[a] * L[a0, a1] + _t_open_shell(J, K, L, G, a0, a1)
                            + common + gmm_ab + gmm_ag + gmm_bg
                            - gm_agx - gm_bgx + gm_abx + gm_gbx)
                    ssum6 = sum((-1.0) ** mu * sn[2 * a + mu]
                                * (2.0 * V[bn, gl, 2 * a + mu, 2 * a + 1 - mu]
                                   - V[2 * a + mu, gl, bn, 2 * a + 1 - mu])
                                for mu in (0, 1))
                    coup6 = ((-1.0) ** (nu + lam + 1) / (2.0 * np.sqrt(2.0))
                             * np.sqrt(n[bn] * n[gx]) * ssum6)
                    terms.append((d_e6, coup6))
                    # complementary split + transfer
                    d_e7 = d_e6 + (K[a0, bn] + K[a0, gl] + K[a1, bn] + K[a1, gl]
                                   - 2.0 * K[a0, a1] - 2.0 * K[bn, gl])
                    ssum7 = sum(sn[2 * a + mu] * V[2 * a + mu, gl, bn, 2 * a + 1 - mu]
                                for mu in (0, 1))
                    coup7 = ((-1.0) ** (nu + lam) * 0.5 * np.sqrt(1.5)
                             * np.sqrt(n[bn] * n[gx]) * ssum7)
                    terms.append((d_e7, coup7))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def _oracle_double_excitation_correction(h, V, omega):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_double_terms(h, V, omega))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(17, 3)\nomega = np.array([1.007, 0.493, 0.980])\n",
            "call": "double_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_double_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(3, 2)\nomega = np.array([0.396, 0.690])\n",
            "call": "double_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_double_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(29, 4)\nomega = np.array([0.955, 1.298, 0.744, 0.528])\n",
            "call": "double_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_double_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(3, 2)\nhbad = h.copy()\nhbad[0, 1] += 0.02\n\ndef run_model():\n    try:\n        double_excitation_correction(hbad, V, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_double_excitation_correction(hbad, V, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
