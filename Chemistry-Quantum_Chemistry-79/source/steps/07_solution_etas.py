"""
Every CAP solution on the trajectory, located as an interior local minimum of the velocity and sharpened by parabolic interpolation.

The optimisation criterion for the CAP strength is often written as a global minimisation of the velocity, but that statement does not describe what is actually done. For a purely imaginary CAP the velocity vanishes identically at eta = 0, which would make the origin the global minimiser while leaving the trajectory entirely unperturbed and the resonance undetermined. The criterion that is really applied is stationarity: a CAP solution sits where

  dv/deta = 0   and   d2v/deta2 > 0

that is, at a local minimum of the velocity, which shows up as an inflection of the eta-trajectory in the complex plane.

Nothing in the formalism promises that such a point is unique. In a finite basis the velocity generally possesses several local minima, and each is a distinct CAP solution, a different balance between an incomplete continuum and the perturbation introduced to compensate for it. All of them are located here; none is privileged.

A local minimum is first bracketed on the grid, at an interior index i where the velocity is strictly smaller than at both neighbours. Because the grid is geometric, the position is then sharpened by fitting a parabola to the three points (ln eta, v) at i - 1, i and i + 1 and taking its vertex; fitting in eta rather than in ln eta biases the result. Solutions are returned in increasing order of CAP strength.

Returns
-------
numpy.ndarray of shape (n_solutions,), real: the refined CAP strengths eta_opt of all velocity local minima, in increasing order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solution_etas(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                  eta_seed: float, e_max: float) -> "np.ndarray":
    '''Refined CAP strengths of every solution on the trajectory.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified.
    e_max : float
        Top of the barrier, the upper edge of the seed search window.

    Returns
    -------
    eta_opt : numpy.ndarray
        Real array of shape (n_solutions,) holding the parabolically refined CAP
        strength of each velocity local minimum, in increasing order. Empty if the
        velocity has no interior local minimum on the grid.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return eta_opt

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _velocity_minima(velocity):
    import numpy as np
    v = np.asarray(velocity, dtype=float)
    return [i for i in range(1, v.size - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]]


def _parabolic_eta(etas, velocity, i):
    """Vertex of the parabola through the three bracketing points in ln(eta)."""
    import numpy as np
    x0, x1, x2 = np.log(etas[i - 1]), np.log(etas[i]), np.log(etas[i + 1])
    y0, y1, y2 = velocity[i - 1], velocity[i], velocity[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    return float(np.exp(-b / (2.0 * a)))


def _oracle_solution_etas(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                          eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    v = _oracle_branch_velocity(integrals, transform, etas, eta_seed, e_max)
    return np.array([_parabolic_eta(g, v, i) for i in _velocity_minima(v)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the production grid, which carries several solutions ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Normal: a coarse grid, where refinement does most of the work ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Boundary: a grid truncated before the outermost solution ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
gt = np.geomspace(1e-4, 0.5, 400)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), gt.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), gt.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Edge: a very short window holding no interior minimum at all ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
gn = np.geomspace(0.3, 0.6, 12)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), gn.copy(), 0.4, 0.88291066)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), gn.copy(), 0.4, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Edge: the alternative basis and potential ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
I_gold = _oracle_basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
gb = np.geomspace(1e-3, 2.0, 70)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), gb.copy(), 0.3, 1.8394)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), gb.copy(), 0.3, 1.8394)',
            "tol": 1e-08,
        },
        # --- Edge: a different potential and a tighter, larger basis ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
I_gold = _oracle_basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g2 = np.geomspace(1e-4, 4.0, 1200)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), g2.copy(), 0.15, 0.9197005)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), g2.copy(), 0.15, 0.9197005)',
            "tol": 1e-08,
        },
        # --- Edge: CAP onset inside the barrier, which leaves only two solutions ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g3 = np.geomspace(1e-4, 4.0, 1200)
""",
            "call": 'solution_etas(I_model.copy(), X_model.copy(), g3.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_solution_etas(I_gold.copy(), X_gold.copy(), g3.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Invalid: a grid containing a negative CAP strength ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
neg = np.linspace(-0.1, 1.0, 30)
def run_model():
    try:
        solution_etas(I_model.copy(), X_model.copy(), neg.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_solution_etas(I_gold.copy(), X_gold.copy(), neg.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_oracle()',
        },
    ]
