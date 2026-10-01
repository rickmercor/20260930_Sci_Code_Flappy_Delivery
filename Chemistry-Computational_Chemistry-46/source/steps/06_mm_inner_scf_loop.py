"""
Solve self-consistently for the induced dipole and induced quadrupole at each of the three MM sites (the QM site is a fixed point multipole and does not respond), then return the converged total electrostatic-plus-induction energy of the coupled system.

In a polarizable embedding both subsystems carry permanent plus induced moments that adjust to each other's field, so the total energy is a functional of the full converged moment set rather than a fixed charge distribution.  At self-consistency the induced moment at each site is exactly the polarizability response to the total on-site field, summed over every other site and its periodic images -- directly analogous to an electronic self-consistent-field cycle.

Returns
-------
return energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erf
 
def mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha: float, C: float,
                  mix: float = 0.35, tol: float = 1e-8, max_iter: int = 20000) -> float:
    '''Solve the MM induced-moment SCF and return the coupled electrostatic+induction energy.
 
    Parameters
    ----------
    perm_mu : array_like, shape (4, 3)
        Permanent dipole of each target -- row 0 QM, rows 1-3 the MM sites (a.u.).
    perm_th : array_like, shape (4, 3, 3) or (4, 5)
        Permanent traceless quadrupole of each target, same ordering.
    v_const, g_const : array_like, shape (4, 3) and (4, 3, 3)
        SCF-independent on-site field and field gradient at each target (fixed QM
        multipole plus far-field expansion centres).
    mm_terms : list of length 4
        ``mm_terms[t]`` is a list of ``(j, T2, T3, T4, iso)`` MM-source contributions to
        target ``t`` (``j`` in 0..2 the MM site index; ``T2`` (3,3), ``T3`` (3,3,3),
        ``T4`` (3,3,3,3); ``iso`` a scalar prefactor).
    alpha, C : float
        Isotropic MM dipole-dipole and quadrupole-quadrupole polarizabilities (a.u.).
        The induced dipole is Delta_mu = alpha * V and the induced quadrupole is
        C times the traceless part of the assembled field gradient,
        Delta_Theta = C * (V_grad - (1/3) tr(V_grad) I).
    mix : float
        Linear mixing parameter t in (0, 1) (default 0.35):
        Delta_M_new = (1 - t) * Delta_M_fresh + t * Delta_M_old.
    tol : float
        SCF convergence threshold (default 1e-8) on the largest absolute change of any
        induced component between the mixed update Delta_M_new and Delta_M_old.
    max_iter : int
        Safety cap on SCF iterations.
 
    Returns
    -------
    energy : float
        Converged total electrostatic-plus-induction energy of the coupled QM--MM
        system, in Hartree.
 
    Raises
    ------
    ValueError
        If ``perm_mu`` is not shape (4, 3), ``v_const`` not (4, 3) or ``g_const`` not
        (4, 3, 3); if ``perm_th`` is not (4, 3, 3) or (4, 5); if ``mm_terms`` does not
        have length 4; if ``mix`` is not strictly inside (0, 1); if ``tol`` is not
        finite and positive; or if ``alpha`` / ``C`` is not finite or ``alpha`` is
        negative.
    RuntimeError
        If the SCF does not converge within ``max_iter`` iterations.
 
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf
 
def _detrace(m: np.ndarray) -> np.ndarray:
    """Traceless part of a (3,3) matrix (the 'traceless quadrupole response')."""
    m = np.asarray(m, dtype=float)
    return m - np.eye(3) * (np.trace(m) / 3.0)
 
 
def _quad_stack(theta, n) -> np.ndarray:
    """Coerce an (n,3,3) or (n,5) quadrupole stack to (n,3,3) symmetric matrices."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (n, 3, 3):
        return 0.5 * (theta + np.transpose(theta, (0, 2, 1)))
    if theta.shape == (n, 5):
        xx, yy, xy, xz, yz = (theta[:, i] for i in range(5))
        out = np.zeros((n, 3, 3))
        out[:, 0, 0] = xx
        out[:, 1, 1] = yy
        out[:, 2, 2] = -(xx + yy)
        out[:, 0, 1] = out[:, 1, 0] = xy
        out[:, 0, 2] = out[:, 2, 0] = xz
        out[:, 1, 2] = out[:, 2, 1] = yz
        return out
    raise ValueError(f"quadrupole stack must be ({n},3,3) or ({n},5)")
 
 
def _contract(T2, T3, T4, mu, th):
    """q = 0 field (3,) and field gradient (3,3) at the target from a source (mu, th)."""
    V = mu @ T2 - (1.0 / 3.0) * np.einsum("bc,abc->a", th, T3)
    G = np.einsum("c,abc->ab", mu, T3) - (1.0 / 3.0) * np.einsum("cd,abcd->ab", th, T4)
    return V, G
 
 
def _target_field(v0, g0, terms, mm_mu, mm_th):
    """Total on-site field / gradient at one target for the current MM moments."""
    V = np.array(v0, dtype=float)
    G = np.array(g0, dtype=float)
    for (j, T2, T3, T4, iso) in terms:
        Vj, Gj = _contract(np.asarray(T2, float), np.asarray(T3, float), np.asarray(T4, float),
                           mm_mu[int(j)], mm_th[int(j)])
        V += iso * Vj
        G += iso * Gj
    return V, G
 
def _oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha: float, C: float,
                          mix: float = 0.35, tol: float = 1e-8, max_iter: int = 20000) -> float:
    """Reference MM SCF + coupled-energy evaluation (pure linear-response solver)."""
    perm_mu = np.asarray(perm_mu, dtype=float)
    if perm_mu.shape != (4, 3):
        raise ValueError("perm_mu must have shape (4, 3)")
    perm_th = _quad_stack(perm_th, 4)
    v_const = np.asarray(v_const, dtype=float)
    g_const = np.asarray(g_const, dtype=float)
    if v_const.shape != (4, 3) or g_const.shape != (4, 3, 3):
        raise ValueError("v_const must be (4, 3) and g_const (4, 3, 3)")
    if not hasattr(mm_terms, "__len__") or len(mm_terms) != 4:
        raise ValueError("mm_terms must be a length-4 list (QM + 3 MM targets)")
    if not (0.0 < float(mix) < 1.0):
        raise ValueError("mix must lie strictly in (0, 1)")
    if not np.isfinite(tol) or float(tol) <= 0.0:
        raise ValueError("tol must be a finite positive number")
    if not np.isfinite(alpha) or not np.isfinite(C) or float(alpha) < 0.0:
        raise ValueError("alpha must be finite and non-negative, C finite")
 
    a, Cc, t = float(alpha), float(C), float(mix)
    mm_perm_mu = perm_mu[1:]        # (3, 3)
    mm_perm_th = perm_th[1:]        # (3, 3, 3)
 
    ind_mu = np.zeros((3, 3))
    ind_th = np.zeros((3, 3, 3))
    converged = False
    for _ in range(int(max_iter)):
        tot_mu = mm_perm_mu + ind_mu
        tot_th = mm_perm_th + ind_th
        fresh_mu = np.zeros((3, 3))
        fresh_th = np.zeros((3, 3, 3))
        for i in range(3):                 # MM targets are list entries 1..3
            V, G = _target_field(v_const[i + 1], g_const[i + 1], mm_terms[i + 1], tot_mu, tot_th)
            fresh_mu[i] = a * V
            fresh_th[i] = Cc * _detrace(G)
        new_mu = (1.0 - t) * fresh_mu + t * ind_mu
        new_th = (1.0 - t) * fresh_th + t * ind_th
        delta = max(np.max(np.abs(new_mu - ind_mu)), np.max(np.abs(new_th - ind_th)))
        ind_mu, ind_th = new_mu, new_th
        if delta < float(tol):
            converged = True
            break
    if not converged:
        raise RuntimeError("MM SCF did not converge within max_iter")
 
    tot_mu = mm_perm_mu + ind_mu
    tot_th = mm_perm_th + ind_th
    total = 0.0
    for tgt in range(4):
        V, G = _target_field(v_const[tgt], g_const[tgt], mm_terms[tgt], tot_mu, tot_th)
        total += perm_mu[tgt] @ V + (1.0 / 3.0) * np.einsum("ab,ab->", perm_th[tgt], G)
    return float(-0.5 * total)

# =============================================================================
# TEST CASES
# =============================================================================

def _rand_terms_setup():
    return """import numpy as np
rng = np.random.default_rng(1)
perm_mu = rng.normal(scale=0.3, size=(4, 3))
perm_th = rng.normal(scale=0.3, size=(4, 5))
v_const = rng.normal(scale=0.01, size=(4, 3))
g_const = rng.normal(scale=0.01, size=(4, 3, 3)); g_const = g_const + g_const.transpose(0, 2, 1)
def _mkT(seed):
    r = np.random.default_rng(seed)
    import itertools
    T2 = r.normal(size=(3, 3)); T2 = T2 + T2.T
    T3 = r.normal(size=(3, 3, 3)); T4 = r.normal(size=(3, 3, 3, 3))
    # fully symmetrise, as real interaction tensors (gradients of 1/r) are
    T3 = sum(np.transpose(T3, p) for p in itertools.permutations(range(3))) / 6.0
    T4 = sum(np.transpose(T4, p) for p in itertools.permutations(range(4))) / 24.0
    return T2 * 1e-3, T3 * 1e-4, T4 * 1e-5
mm_terms = [[], [], [], []]
for tgt in range(4):
    for j in range(3):
        if tgt >= 1 and j == tgt - 1:
            continue
        T2, T3, T4 = _mkT(10 * tgt + j)
        mm_terms[tgt].append((j, T2, T3, T4, 1.0))
alpha, C = 9.85, 24.5
"""
 
 
def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": _rand_terms_setup(),
            "call": "mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha, C)",
            "gold_call": "_oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha, C)",
        },
        {
            "setup": _rand_terms_setup(),
            "call": "mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, 0.0, 0.0)",
            "gold_call": "_oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, 0.0, 0.0)",
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(2)
perm_mu = rng.normal(scale=0.3, size=(4, 3))
perm_th = np.zeros((4, 3, 3))
v_const = rng.normal(scale=0.02, size=(4, 3))
g_const = np.zeros((4, 3, 3))
mm_terms = [[], [], [], []]
""",
            "call": "mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, 9.85, 24.5)",
            "gold_call": "_oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, 9.85, 24.5)",
        },
        {
            "setup": _rand_terms_setup() + """
def run_model():
    try:
        mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha, C, mix=1.5); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha, C, mix=1.5); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": _rand_terms_setup() + """
mm_terms3 = mm_terms[:3]
def run_model():
    try:
        mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms3, alpha, C); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms3, alpha, C); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
