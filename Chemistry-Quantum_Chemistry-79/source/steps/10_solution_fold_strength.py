"""
The continuum-remover strength at which one chosen CAP solution ceases to exist.

A remedy proposed for the multiplicity of CAP solutions is the continuum remover: a real multiple of the same one-body potential is added to the absorber, so that the Hamiltonian becomes

  H(eta, kappa) = H0 + (kappa - i eta) W

with kappa >= 0; kappa is unrelated to the exponent lambda_pot of the potential. The claim behind it is selective, that the real term destroys the solutions which are artefacts of the discretised continuum while leaving the physical one in place. Deciding whether that is so requires knowing, for each solution separately, how large kappa must be before it is gone.

The solution examined here is picked out at kappa = 0 as the velocity minimum lying nearest in ln(eta) to the reference strength supplied by the caller. Every rule that defines a solution carries over unchanged to kappa > 0, with H0 + kappa W taking the place of H0: the seed window bounded by the barrier top A / (lambda_pot e) of the unshifted potential, the smallest-extent identification, the overlap tracking along eta, and the c-product Hellmann-Feynman velocity. What does not carry over is the identification of the state itself. Re-applying the seed rule at every strength is unreliable, because over some intervals of kappa the smallest-extent state inside the seed window belongs to the discretised continuum rather than to the branch being followed, and a calculation that re-seeds there loses the branch and reports a spurious disappearance. The identity is therefore carried in kappa: the eigenvector accepted at the previous strength, at the position its minimum occupied there, fixes the state at the next strength through the largest c-product overlap, and the branch is tracked outwards in eta from that position as usual.

The removal is not gradual. As kappa increases, the local minimum of the velocity and the local maximum immediately below it in eta approach one another and coalesce; at the coalescence strength the pair annihilates and the solution no longer exists. Existence is judged on the velocity as a smooth function of ln(eta) inside the window spanned by that pair, not on the grid, because a stationary point narrower than a grid cell is still a stationary point. The window is the one inherited from the largest strength at which the solution was still present, which is what makes the disappearance of this solution, rather than of a neighbour, the quantity measured. The coalescence strength has no closed form, since every evaluation of the velocity requires the full non-Hermitian eigenproblem, and it is asked to an absolute accuracy of 1e-10 in kappa. That accuracy is far beyond anything the grid can resolve, so the coalescence has to be located in the continuous variable, using analytic first and second derivatives of the resonance energy with respect to eta.

The caller supplies a ladder spacing in kappa, used only to find the interval that contains the coalescence, and a largest strength beyond which the search is abandoned. The answer does not depend on the spacing. Use the earlier public steps to obtain the unshifted branch energies, velocity and analytic velocity slope; the slope identifies a genuine continuous stationary point, and the branch energy identifies the state vector carried into the remover search.

Returns
-------
float: the continuum-remover strength at which the chosen CAP solution merges with the velocity maximum immediately below it and vanishes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solution_fold_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                           potential_width: float, cap_onset: float, threshold: float,
                           eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                           eta_ref: float, kappa_step: float, kappa_max: float) -> float:
    '''Continuum-remover strength at which one chosen CAP solution vanishes.

    Parameters
    ----------
    alpha0 : float
        Exponent of the most diffuse Gaussian.
    beta : float
        Even-tempered ratio.
    n_basis : int
        Number of Gaussians.
    barrier_strength : float
        Coefficient A of V(x) = A x^2 exp(-lambda_pot x^2).
    potential_width : float
        Exponent lambda_pot of the potential.
    cap_onset : float
        Onset of the box CAP.
    threshold : float
        Overlap-eigenvalue cut for canonical orthogonalisation.
    eta_min : float
        Smallest CAP strength on the grid; must be positive.
    eta_max : float
        Largest CAP strength on the grid; must exceed eta_min.
    n_eta : int
        Number of geometrically spaced grid points; must be at least 3.
    eta_seed : float
        CAP strength at which the resonance is identified at kappa = 0.
    eta_ref : float
        Reference CAP strength; the solution followed is the velocity minimum of the
        unshifted problem nearest to it in ln(eta). Must be positive.
    kappa_step : float
        Positive ladder spacing in kappa used to bracket the coalescence.
    kappa_max : float
        Largest remover strength examined; must exceed kappa_step.

    Returns
    -------
    kappa_c : float
        The strength of the real term kappa in H0 + (kappa - i eta) W at which the chosen
        CAP solution coalesces with the velocity maximum immediately below it in eta and
        vanishes, to an absolute accuracy of 1e-10.

    Raises
    ------
    ValueError
        If eta_ref is not positive, if kappa_step is not positive, if kappa_max does not
        exceed kappa_step, if the unshifted problem has no CAP solution, if the followed
        branch is lost before it coalesces, if the solution still exists at kappa_max, or
        if any other argument violates the constraints of the steps it feeds.
    '''
    return kappa_c

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _remover_shift(integrals, lam):
    import numpy as np
    J = np.array(integrals, dtype=float, copy=True)
    J[1] = J[1] + float(lam) * J[2]
    return J


def _branch_from(integrals, transform, etas, index, reference):
    """Velocity and eigenvectors of the branch through the state nearest reference at etas[index]."""
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    g = np.asarray(etas, dtype=float)
    n = g.size
    vel = np.empty(n, dtype=float)
    vecs = [None] * n

    def _record(i, C_i, idx):
        c = C_i[:, idx]
        vel[i] = abs(g[i] * (-1j) * (c @ W @ c))
        vecs[i] = c
        return c

    vals, C = _cap_solve(integrals, transform, g[index])
    prev = _record(index, C, int(np.argmax(np.abs(np.asarray(reference) @ S @ C))))
    for i in range(index - 1, -1, -1):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        prev = _record(i, C_i, int(np.argmax(np.abs(prev @ S @ C_i))))
    prev = vecs[index]
    for i in range(index + 1, n):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        prev = _record(i, C_i, int(np.argmax(np.abs(prev @ S @ C_i))))
    return vel, vecs


def _fold_window(etas, velocity, eta_ref, slope=None):
    """Grid minimum nearest eta_ref in ln(eta), plus the maximum immediately below it."""
    import numpy as np
    g = np.asarray(etas, dtype=float)
    v = np.asarray(velocity, dtype=float)
    minima = _velocity_minima(v)
    if slope is not None:
        s = np.asarray(slope, dtype=float)
        if s.shape != v.shape:
            raise ValueError("velocity and slope must have the same shape")
        # A sampled minimum represents a smooth stationary minimum only when
        # the analytic slope changes from negative to positive nearby.
        minima = [i for i in minima
                  if np.min(s[max(0, i - 1):i + 1]) <= 0.0
                  and np.max(s[i:min(s.size, i + 2)]) >= 0.0]
    if len(minima) == 0:
        return None
    i = min(minima, key=lambda m: abs(np.log(g[m]) - np.log(float(eta_ref))))
    left = i
    while left > 0 and v[left - 1] >= v[left]:
        left -= 1
    return float(g[left]), float(g[i]), int(i)


def _fold_depth(integrals, transform, etas, vecs, eta_lo, eta_hi):
    """Smallest value of the smooth slope dv/dln(eta) over [eta_lo, eta_hi]."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    g = np.asarray(etas, dtype=float)
    idx = np.where((g >= eta_lo) & (g <= eta_hi))[0]
    if idx.size == 0:
        raise ValueError("the window contains no grid point")
    slope = np.array([_slope_at(integrals, transform, g[i], vecs[i])[0] for i in idx])
    j = int(np.argmin(slope))
    s = np.log(g)
    a = s[idx[max(j - 1, 0)]]
    b = s[idx[min(j + 1, idx.size - 1)]]
    best, eta_best = float(slope[j]), float(g[idx[j]])
    if b > a:
        res = minimize_scalar(lambda t: _slope_at(integrals, transform, float(np.exp(t)), vecs[idx[j]])[0],
                              bounds=(a, b), method='bounded', options={'xatol': 1e-13})
        if float(res.fun) < best:
            best, eta_best = float(res.fun), float(np.exp(res.x))
    return eta_best, best


def _oracle_solution_fold_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                                   potential_width: float, cap_onset: float, threshold: float,
                                   eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                                   eta_ref: float, kappa_step: float, kappa_max: float) -> float:
    import numpy as np
    from scipy.optimize import brentq
    if not np.isfinite(eta_ref) or eta_ref <= 0.0:
        raise ValueError("eta_ref must be positive")
    if not np.isfinite(kappa_step) or kappa_step <= 0.0:
        raise ValueError("kappa_step must be positive")
    if not np.isfinite(kappa_max) or kappa_max <= kappa_step:
        raise ValueError("kappa_max must exceed kappa_step")
    if not np.isfinite(eta_min) or not np.isfinite(eta_max) or eta_min <= 0.0 or eta_max <= eta_min:
        raise ValueError("need 0 < eta_min < eta_max")
    if int(n_eta) < 3:
        raise ValueError("n_eta must be at least 3")
    e_max = float(barrier_strength) / (float(potential_width) * np.e)
    I = _oracle_basis_integrals(alpha0, beta, n_basis, barrier_strength, potential_width, cap_onset)
    X = _oracle_canonical_transform(I[0].copy(), threshold)
    etas = np.geomspace(float(eta_min), float(eta_max), int(n_eta))

    energies = _oracle_branch_energies(I, X, etas, eta_seed, e_max)
    velocity = _oracle_branch_velocity(I, X, etas, eta_seed, e_max)
    slope = _oracle_branch_velocity_slope(I, X, etas, eta_seed, e_max)
    window = _fold_window(etas, velocity, eta_ref, slope)
    if window is None:
        raise ValueError("the unshifted problem has no CAP solution")
    lo, hi, anchor = window
    vals, C = _cap_solve(I, X, float(etas[anchor]))
    cref = C[:, int(np.argmin(np.abs(vals - energies[anchor])))]

    kappa = 0.0
    step = float(kappa_step)
    upper = float(kappa_max)
    while kappa < upper:
        nxt = min(kappa + step, upper)
        J = _remover_shift(I, nxt)
        vel_n, vecs_n = _branch_from(J, X, etas, anchor, cref)
        if _fold_depth(J, X, etas, vecs_n, lo, hi)[1] >= 0.0:
            break
        kappa = nxt
        window = _fold_window(etas, vel_n, etas[anchor])
        if window is None:
            raise ValueError("the followed branch was lost before it coalesced")
        lo, hi, anchor = window
        cref = vecs_n[anchor]
    else:
        raise ValueError("the chosen solution still exists at kappa_max")

    def _depth(lam):
        J = _remover_shift(I, lam)
        _, vv = _branch_from(J, X, etas, anchor, cref)
        return _fold_depth(J, X, etas, vv, lo, hi)[1]

    f_lo, f_hi = _depth(kappa), _depth(nxt)
    if not (f_lo < 0.0 < f_hi):
        raise ValueError("the solution does not vanish by coalescence inside the bracketing interval")
    kappa_c = float(brentq(_depth, kappa, nxt, xtol=1e-14, rtol=4.0 * np.finfo(float).eps))
    if abs(_depth(kappa_c)) > 1e-8:
        raise ValueError("the coalescence condition is not satisfied at the located strength")
    return kappa_c

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the production model, the solution that carries the largest first-order width ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.02, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Normal: the innermost solution of the same model, which goes first ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.0282045, 0.02, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.0282045, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Normal: the outermost solution, whose identity is lost by any calculation that re-seeds ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 2.1530456, 0.05, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 2.1530456, 0.05, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: a four times coarser ladder, which must not move a coalescence located in the continuous variable ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.04, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.04, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: two more Gaussians ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 22, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.02, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 22, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: a wider absorber onset, where the model carries two solutions ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.76, 1e-8, 1e-4, 4.0, 600, 0.15, 0.4703455, 0.02, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.76, 1e-8, 1e-4, 4.0, 600, 0.15, 0.4703455, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: a stronger barrier ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.5, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.2, 0.02, 2.0)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.5, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.2, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: kappa_max cuts the last ladder interval after the physical fold ---
        {
            "setup": """import numpy as np
""",
            "call": "solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 200, 0.15, 2.1530456, 0.3, 1.3)",
            "gold_call": "_oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 200, 0.15, 2.1530456, 0.3, 1.3)",
            "tol": 4e-11,
        },
        # --- Invalid: a largest strength the solution outlives ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.01, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.01, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a ladder spacing that is not positive ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solution_fold_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.1513564, 0.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
