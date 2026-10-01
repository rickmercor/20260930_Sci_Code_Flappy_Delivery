"""
Step 1 — Damped relaxation of the bond-unit gap profile (fixed sweep budget).

The system is a valence-only pairing model: 2M spatial orbitals grouped into M

bond units, unit a holding a bonding orbital (index 2a) and an antibonding

orbital (index 2a+1). Real one- and two-electron integrals h_pq and

V_pqrs = (pq|rs) (chemists' notation) are given. Only the reduced two-electron

elements J_pq = V_ppqq, K_pq = V_pqqp, L_pq = V_pqpq and G = 2J - K enter this

step.



The reference state places one electron pair in each bond unit, delocalized

over the unit's two orbitals with occupations

    n_{a,0} = 1 + w_a/eta_a,   n_{a,1} = 1 - w_a/eta_a,

    eta_a = sqrt(w_a^2 + 1),

where w_a is the unit's gap parameter. With effective orbital energies

    eps_p = h_pp + L_pp/2 + (1/2) sum_{q outside p's unit} G_pq n_q,

the exact stationarity solve for unit a alone, all other units' occupations

frozen, is single-shot — eps_{a0} and eps_{a1} do not depend on unit a's own

occupations —

    w_a = (eps_{a1} - eps_{a0}) / L_{a0,a1}.

The instance does NOT converge this coupled problem. It prescribes a frozen,

path-dependent profile: starting from w = 0 (all n = 1), run exactly n_sweeps

damped Jacobi sweeps in which every unit's single-unit solution is evaluated

at the current profile and every gap then moves the fraction alpha of the way

toward it,

    w_a  <-  w_a + alpha * ((eps_{a1} - eps_{a0}) / L_{a0,a1} - w_a),

all units updated simultaneously from the same current profile. Return the

profile after exactly n_sweeps sweeps — no convergence test, no early exit,

no further refinement. Raise ValueError if any intra-unit pair-transfer

integral L_{a0,a1} vanishes (|L| < 1e-12), if alpha lies outside (0, 1], or

if n_sweeps is not a positive integer.



Inputs: h (2M, 2M) symmetric; V (2M, 2M, 2M, 2M) with 8-fold real-orbital

symmetry; alpha; n_sweeps. Returns the frozen gap vector, shape (M,).



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

  - an intra-unit pair-transfer integral L_{a0,a1} vanishes

  - alpha must lie in (0, 1]

  - n_sweeps must be a positive integer

Returns
-------
np.ndarray of shape (M,), the frozen bond-unit gap parameters after exactly n_sweeps damped sweeps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    '''Run the fixed-budget damped relaxation of the gap profile.

    Parameters
    ----------
    h : np.ndarray
        (2M, 2M) symmetric one-electron integral matrix.
    V : np.ndarray
        (2M, 2M, 2M, 2M) two-electron integrals (pq|rs), 8-fold symmetric.
    alpha : float
        Damping factor in (0, 1]: the fraction of the single-unit stationarity
        update applied per sweep.
    n_sweeps : int
        Exact number of simultaneous damped sweeps to run from w = 0; the
        budget is unconditional (no convergence test, no early exit).

    Returns
    -------
    omega : np.ndarray
        (M,) frozen bond-unit gap parameters after exactly n_sweeps sweeps.
    '''
    return omega

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_MK_SYSTEM = """import numpy as np

def _mk_system(seed, M):
    rng = np.random.default_rng(seed)
    centers = np.concatenate([[0.0], np.cumsum(1.75 + 0.35 * rng.random(M - 1))])
    dx = 0.10 + 0.05 * rng.random(M)
    x = np.empty(2 * M)
    x[0::2] = centers
    x[1::2] = centers + dx
    tau = 0.95 + 0.10 * rng.random(2 * M)
    A = np.outer(tau, tau) * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2)
    m = 0.5 * (x[:, None] + x[None, :])
    V = (A[:, :, None, None] * A[None, None, :, :]
         / (1.0 + np.abs(m[:, :, None, None] - m[None, None, :, :])))
    h = np.zeros((2 * M, 2 * M))
    diag_b = -1.25 + 0.08 * rng.random(M)
    gap = 0.45 + 0.55 * rng.random(M)
    for a in range(M):
        h[2 * a, 2 * a] = diag_b[a]
        h[2 * a + 1, 2 * a + 1] = diag_b[a] + gap[a]
    off = 0.02 * rng.standard_normal((2 * M, 2 * M))
    off = 0.5 * (off + off.T)
    np.fill_diagonal(off, 0.0)
    h = h + off
    return h, V
"""

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

def _relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    """Exactly n_sweeps damped Jacobi sweeps of
    w_a <- w_a + alpha * ((eps_{a1} - eps_{a0}) / L_{a0,a1} - w_a) from w = 0."""
    _, _, L, G = _pair_integrals(V)
    M = h.shape[0] // 2
    l_intra = np.array([L[2 * a, 2 * a + 1] for a in range(M)])
    if np.any(np.abs(l_intra) < 1e-12):
        raise ValueError("an intra-unit pair-transfer integral L_{a0,a1} vanishes")
    omega = np.zeros(M)
    for _ in range(int(n_sweeps)):
        n, _ = _occupations(omega)
        eps = _orbital_energies(h, L, G, n)
        omega = omega + alpha * ((eps[1::2] - eps[0::2]) / l_intra - omega)
    return omega

def _oracle_relax_bond_gaps(h, V, alpha=0.55, n_sweeps=5):
    """Reference implementation. Deterministic."""
    h, V, _ = _validate_system(h, V)
    alpha = float(alpha)
    if not (np.isfinite(alpha) and 0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    if not float(n_sweeps).is_integer() or int(n_sweeps) < 1:
        raise ValueError("n_sweeps must be a positive integer")
    return _relax_bond_gaps(h, V, alpha=alpha, n_sweeps=int(n_sweeps))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": _MK_SYSTEM + "\nh, V = _mk_system(11, 3)\n",
            "call": "relax_bond_gaps(h, V)",
            "gold_call": "_oracle_relax_bond_gaps(h, V)",
        },
        {
            "setup": _MK_SYSTEM + "\nh, V = _mk_system(5, 2)\n",
            "call": "relax_bond_gaps(h, V, alpha=0.4, n_sweeps=3)",
            "gold_call": "_oracle_relax_bond_gaps(h, V, alpha=0.4, n_sweeps=3)",
        },
        {
            "setup": _MK_SYSTEM + "\nh, V = _mk_system(23, 4)\n",
            "call": "relax_bond_gaps(h, V, alpha=1.0, n_sweeps=7)",
            "gold_call": "_oracle_relax_bond_gaps(h, V, alpha=1.0, n_sweeps=7)",
        },
        {
            "setup": _MK_SYSTEM + """
h, V = _mk_system(11, 2)
Vbad = V.copy()
Vbad[0, 1, 2, 3] += 0.05

def run_model():
    try:
        relax_bond_gaps(h, Vbad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_relax_bond_gaps(h, Vbad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _MK_SYSTEM + """
h, V = _mk_system(11, 2)

def run_model():
    try:
        relax_bond_gaps(h, V, alpha=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_relax_bond_gaps(h, V, alpha=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _MK_SYSTEM + """
h, V = _mk_system(11, 2)

def run_model():
    try:
        relax_bond_gaps(h[:3, :3], V[:3, :3, :3, :3])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_relax_bond_gaps(h[:3, :3], V[:3, :3, :3, :3])
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
