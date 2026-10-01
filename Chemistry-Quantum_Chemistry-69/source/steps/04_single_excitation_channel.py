"""
Step 4 — Second-order sum over the single-excitation channel.

Second-order perturbation theory with diagonal (state-resolved) partitioning:

each excited configuration Psi contributes -|<Psi|H|ref>|^2 / (E[Psi] - E_ref).

This step sums the SINGLE excitations. Notation: unit indices a != b; mu, nu

in {0,1}; x marks the partner orbital (index 1-mu resp. 1-nu of the same

unit); t_pq = J_pq + K_pq - L_pp/2 - L_qq/2, minus G_pq/2 extra when p, q sit

in different units; g-_{a,p} = (w_a/(2 eta_a))(G_{a0,p} - G_{a1,p});

g--_{a,b} = (w_a w_b/(2 eta_a eta_b))(G_{a0,b0} - G_{a0,b1} - G_{a1,b0}

+ G_{a1,b1}); f is the generalized Fock matrix of Step 3.



Three families of singles exist:

 (i) occupation swaps within unit a: excitation energy 2 eta_a L_{a0,a1};

     coupling (eps_{a0} - eps_{a1} + w_a L_{a0,a1}) / eta_a, which vanishes

     only at exact gap stationarity (the pairing analogue of the Brillouin

     condition) and is nonzero at a general omega such as the frozen

     fixed-budget profile of Step 1 -- evaluate it, never assume it away;

 (ii) pair splits within unit a (both orbitals singly occupied): energy

     eta_a L_{a0,a1} + t_{a0,a1}; coupling

     (f_{a0,a1} - f_{a1,a0}) / (sqrt(n_{a0}) + sqrt(n_{a1})) -- nonzero here

     because the fixed orbitals are not exactly variationally optimal;

 (iii) electron transfers a_mu -> b_nu between different units (2M(M-1)*2

     states): energy

       eta_a L_{a0,a1} + eta_b L_{b0,b1} + t_{a_mu,b_nu} + eps_{b_x}

       - eps_{a_x} + G_{b0,b1} + g--_{a,b} - g-_{a,b_x} + g-_{b,a_x}

       - G_{a_x,b_x}/2,

     coupling

       (-1)^(mu+nu+1) sqrt(n_{b_x}/(2 n_{a_mu})) * [ f_{a_mu,b_nu}

       + n_{a_mu} sqrt(n_{b_nu}/n_{b_x}) V_{b_nu,b_x,a_mu,b_x}

       + (1/2)(2 V_{b_x,b_x,a_mu,b_nu} - V_{b_x,b_nu,a_mu,b_x}

       - V_{b_nu,b_nu,a_mu,b_nu}) n_{a_mu} n_{b_nu} ].

Raise ValueError on a vanishing excitation energy. Returns the channel's

second-order sum (a non-positive float when all energies are positive).



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
float, the single-channel second-order energy sum in Hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_excitation_correction(h, V, omega, fock=None):
    '''Second-order energy sum over swaps, splits, and electron transfers.

    Parameters
    ----------
    h : np.ndarray
        (2M, 2M) symmetric one-electron integrals.
    V : np.ndarray
        (2M, 2M, 2M, 2M) two-electron integrals (pq|rs).
    omega : np.ndarray
        (M,) bond-unit gap parameters.
    fock : np.ndarray or None
        Optional (2M, 2M) generalized Fock matrix. None recomputes it.

    Returns
    -------
    e2 : float
        Sum of -|coupling|^2 / (excitation energy) over all single excitations.
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

def _single_terms(h, V, omega, fock=None):
    """(excitation energy, coupling) for every single excitation."""
    J, K, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    n, eta = _occupations(omega)
    eps = _orbital_energies(h, L, G, n)
    f = _generalized_fock(h, V, omega) if fock is None else np.asarray(fock, dtype=float)
    sn = np.sqrt(n)
    terms = []
    for a in range(M):
        a0, a1 = 2 * a, 2 * a + 1
        eta_l = eta[a] * L[a0, a1]
        # occupation swap within unit a
        terms.append((2.0 * eta_l,
                      (eps[a0] - eps[a1] + omega[a] * L[a0, a1]) / eta[a]))
        # pair split within unit a
        terms.append((eta_l + _t_open_shell(J, K, L, G, a0, a1),
                      (f[a0, a1] - f[a1, a0]) / (sn[a0] + sn[a1])))
    for a, b in itertools.permutations(range(M), 2):
        for mu in (0, 1):
            for nu in (0, 1):
                am, ax = 2 * a + mu, 2 * a + 1 - mu
                bn, bx = 2 * b + nu, 2 * b + 1 - nu
                d_e = (eta[a] * L[2 * a, 2 * a + 1] + eta[b] * L[2 * b, 2 * b + 1]
                       + _t_open_shell(J, K, L, G, am, bn)
                       + eps[bx] - eps[ax] + G[2 * b, 2 * b + 1]
                       + _g_mm(G, omega, eta, a, b)
                       - _g_minus(G, omega, eta, a, bx)
                       + _g_minus(G, omega, eta, b, ax)
                       - 0.5 * G[ax, bx])
                rhs = (f[am, bn]
                       + n[am] * np.sqrt(n[bn] / n[bx]) * V[bn, bx, am, bx]
                       + 0.5 * (2.0 * V[bx, bx, am, bn] - V[bx, bn, am, bx]
                                - V[bn, bn, am, bn]) * n[am] * n[bn])
                coup = (-1.0) ** (mu + nu + 1) * np.sqrt(n[bx] / (2.0 * n[am])) * rhs
                terms.append((d_e, coup))
    return terms

def _en2_sum(terms):
    total = 0.0
    for d_e, coup in terms:
        if d_e == 0.0:
            raise ValueError("vanishing excitation energy in a perturbative denominator")
        total -= coup * coup / d_e
    return float(total)

def _oracle_single_excitation_correction(h, V, omega, fock=None):
    """Reference implementation. Deterministic."""
    h, V, M = _validate_system(h, V)
    omega = _validate_gaps(omega, M)
    return _en2_sum(_single_terms(h, V, omega, fock))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(11, 3)\nomega = np.array([0.198, 1.272, 0.700])\n",
            "call": "single_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_single_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(13, 2)\nomega = np.array([1.115, 0.431])\n",
            "call": "single_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_single_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(7, 3)\nomega = np.array([0.0, 0.70, 1.30])\n",
            "call": "single_excitation_correction(h, V, omega)",
            "gold_call": "_oracle_single_excitation_correction(h, V, omega)",
        },
        {
            "setup": "import numpy as np\n\ndef _mk_system(seed, M):\n    rng = np.random.default_rng(seed)\n    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])\n    dx = 0.10 + 0.05 * rng.random(M)\n    x = np.empty(2 * M)\n    x[0::2] = centers\n    x[1::2] = centers + dx\n    tau = 0.95 + 0.10 * rng.random(2 * M)\n    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)\n    m = 0.5 * (x[:, None] + x[None, :])\n    V = (A[:, :, None, None] * A[None, None, :, :]\n         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))\n    h = np.zeros((2 * M, 2 * M))\n    diag_b = -1.25 + 0.08 * rng.random(M)\n    gap = 0.45 + 0.55 * rng.random(M)\n    for a in range(M):\n        h[2 * a, 2 * a] = diag_b[a]\n        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]\n    off = 0.02 * rng.standard_normal((2 * M, 2 * M))\n    off = 0.5 * (off + off.T)\n    np.fill_diagonal(off, 0.0)\n    h = h + off\n    return h, V\n\nh, V = _mk_system(13, 2)\nVbad = V.copy()\nVbad[1, 0, 3, 2] -= 0.03\n\ndef run_model():\n    try:\n        single_excitation_correction(h, Vbad, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n\ndef run_gold():\n    try:\n        _oracle_single_excitation_correction(h, Vbad, np.array([0.5, 0.5]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
