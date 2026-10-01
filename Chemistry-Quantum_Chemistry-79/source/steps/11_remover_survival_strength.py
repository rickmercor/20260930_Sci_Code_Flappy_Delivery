"""
The continuum-remover strength beyond which no CAP solution of the model survives.

The continuum remover is advertised as selective: the real term added to the absorber is supposed to destroy the solutions that belong to the discretised continuum and to spare the physical one. A single coalescence strength cannot test that claim. What tests it is the whole set: every solution of the unshifted problem is followed separately in kappa until it merges with the velocity maximum immediately below it, and the strengths at which they go are compared with the widths they carry.

The largest of those strengths is the quantity returned here. Below it at least one solution of the velocity criterion still exists, so a CAP calculation still has something to report; above it the criterion selects nothing at all and the method has been switched off rather than disambiguated. Which solution is the last to go is the substantive question, because a remover that deserved the name would retire the artefacts and leave the resonance. The order in which the solutions disappear is set by where they sit on the eta axis, not by the widths they predict, so the survivor is not in general the solution a stability argument would have kept.

Solutions are counted once, at kappa = 0, by the velocity criterion on the specified grid; a strength at which two of them have already merged with each other is not reached in the models considered here, and no solution that is absent at kappa = 0 is picked up later. Build the integrals and canonical transform with the earlier public steps, obtain the solution positions and corrected widths with solution_etas and deperturbed_widths, and obtain each disappearance strength with solution_fold_strength. They are followed in order of decreasing first-order corrected width, so that each coalescence strength is obtained against the width the solution predicts, and where two coalescence strengths agree to within 1e-12 the one belonging to the wider solution is the one returned.

Returns
-------
float: the largest continuum-remover strength at which any CAP solution of the unshifted problem still exists, that is the strength at which the last of them vanishes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def remover_survival_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                              potential_width: float, cap_onset: float, threshold: float,
                              eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                              kappa_step: float, kappa_max: float) -> float:
    '''Remover strength at which the last surviving CAP solution of the model vanishes.

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
    kappa_step : float
        Positive ladder spacing in kappa used to bracket each coalescence.
    kappa_max : float
        Largest remover strength examined; must exceed kappa_step.

    Returns
    -------
    kappa_last : float
        The largest of the coalescence strengths of the CAP solutions of the unshifted
        problem, that is the smallest remover strength at which none of them is left,
        to an absolute accuracy of 1e-10. Solutions are followed in order of decreasing
        first-order corrected width, and coalescence strengths that agree to within
        1e-12 are resolved in favour of the wider solution.

    Raises
    ------
    ValueError
        If the unshifted problem has no CAP solution, if any solution still exists at
        kappa_max or is lost before it coalesces, or if any other argument violates the
        constraints of the steps it feeds.
    '''
    return kappa_last

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_remover_survival_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                                      potential_width: float, cap_onset: float, threshold: float,
                                      eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                                      kappa_step: float, kappa_max: float) -> float:
    import numpy as np
    e_max = float(barrier_strength) / (float(potential_width) * np.e)
    I = _oracle_basis_integrals(alpha0, beta, n_basis, barrier_strength, potential_width, cap_onset)
    X = _oracle_canonical_transform(I[0].copy(), threshold)
    etas = np.geomspace(float(eta_min), float(eta_max), int(n_eta))
    eta_opt = np.asarray(_oracle_solution_etas(I, X, etas, eta_seed, e_max), dtype=float)
    if eta_opt.size == 0:
        raise ValueError("the unshifted problem has no CAP solution")
    widths = np.asarray(_oracle_deperturbed_widths(I, X, etas, eta_seed, e_max), dtype=float)
    if widths.size != eta_opt.size:
        raise ValueError("the solutions and their first-order widths disagree in number")
    order = np.argsort(-widths, kind='stable')
    best, best_width = None, None
    for k in order:
        fold = _oracle_solution_fold_strength(alpha0, beta, n_basis, barrier_strength, potential_width,
                                              cap_onset, threshold, eta_min, eta_max, n_eta, eta_seed,
                                              float(eta_opt[k]), kappa_step, kappa_max)
        if best is None or fold > best + 1e-12 or (abs(fold - best) <= 1e-12 and widths[k] > best_width):
            best, best_width = fold, float(widths[k])
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the production model, whose three solutions go at widely separated strengths ---
        {
            "setup": """import numpy as np
""",
            "call": "remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 1200, 0.15, 0.02, 2.0)",
            "gold_call": "_oracle_remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 1200, 0.15, 0.02, 2.0)",
            "tol": 4e-11,
        },
        # --- Normal: the wider absorber onset adopted for molecular shape resonances ---
        {
            "setup": """import numpy as np
""",
            "call": "remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.76, 1e-8, 1e-4, 4.0, 600, 0.15, 0.04, 2.0)",
            "gold_call": "_oracle_remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.76, 1e-8, 1e-4, 4.0, 600, 0.15, 0.04, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: half the grid and a coarser ladder, neither of which may move the answer ---
        {
            "setup": """import numpy as np
""",
            "call": "remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.05, 2.0)",
            "gold_call": "_oracle_remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.05, 2.0)",
            "tol": 4e-11,
        },
        # --- Edge: a weaker, wider barrier with its own solution structure ---
        {
            "setup": """import numpy as np
""",
            "call": "remover_survival_strength(0.02, 2.0, 20, 1.0, 0.4, 2.5, 1e-8, 1e-4, 4.0, 600, 0.15, 0.04, 2.0)",
            "gold_call": "_oracle_remover_survival_strength(0.02, 2.0, 20, 1.0, 0.4, 2.5, 1e-8, 1e-4, 4.0, 600, 0.15, 0.04, 2.0)",
            "tol": 4e-11,
        },
        # --- Invalid: a largest strength that the outermost solution outlives ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.02, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 600, 0.15, 0.02, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a grid of two points, which cannot carry a local minimum ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 2, 0.15, 0.02, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_remover_survival_strength(0.02, 2.0, 20, 1.2, 0.5, 2.0, 1e-8, 1e-4, 4.0, 2, 0.15, 0.02, 2.0)
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
