"""
Run the complete pipeline end to end and return the target scalar.

Build the grid as n_nodes equally spaced points on the unit interval, with spacing dx equal to the distance between neighbours. Form the nodal diffusion coefficients as d (1 + d_var sin(2 pi x)). Form the initial state by stacking A + 0.1 sin(2 pi x), B / A + 0.1 sin(2 pi x), and B + 0.1 sin(2 pi x).

Evaluate the split components at the initial state and check that the reactive component vanishes at both boundary nodes of every species, so that the stationary boundary condition holds before any stepping begins.

Estimate the dominant eigenvalue of the diffusion operator at the initial state using the given seed, weighted-root-mean-square tolerances, stopping tolerance, and safety factor. This estimate is formed once and reused for every step. Select the internal stage counts for the two super-time-stepping stages from the step size h = T / n_steps, the abscissa increments of those two stages, and the safeguarded magnitude.

Advance the first step by assembling it explicitly: set the first stage value to the initial state, and for each subsequent stage assemble the coupling vector from the tableau row differences against the stage values already computed, then either take one Runge-Kutta-Legendre super-step with the previously selected stage count, or copy the previous stage value plus the coupling vector when the abscissa increment vanishes and the implicit diagonal is zero, or solve the implicit reaction stage. Evaluate the split components after each stage to supply the advective and reactive stage values.

Advance the remaining n_steps - 1 steps with the composite single-step routine, using the same fixed step size and the same safeguarded eigenvalue magnitude throughout. No step-size adaptivity is performed and the embedded solution is not used to modify the step size.

Return the entry of the final state at position target_index.

The function raises ValueError when n_nodes is not an integer of at least 3; when n_steps is not a positive integer; when T is not a finite positive scalar; when d is not a finite positive scalar; when target_index is not an integer index into the state vector; when the reactive component fails the boundary check at the initial state; when the split operator returns an unexpected shape; when the eigenvalue estimate is not negative; when any selected stage count falls below 2; or when the number of super-time-stepping stages encountered does not match the number of selected stage counts. Errors raised by the called sub-problem functions propagate unchanged.

Assembling the pipeline exposes a division of labour that is easy to lose sight of when each piece is examined alone. The eigenvalue estimate is a property of the spatial discretization and is computed once, before any stepping; the stage counts derived from it are properties of the estimate together with the step size, and are likewise fixed for the whole integration; and only the stage values themselves change from step to step. In an adaptive implementation the first two would be revisited, since a rejected step changes the step size and therefore the stage counts, and a slowly varying operator eventually invalidates the estimate. Fixing the step size removes that feedback and makes the entire integration a deterministic composition of identical maps.

That determinism is what allows the whole trajectory to be characterized by a small number of integers and a few scalars. Two integrations that agree on the eigenvalue estimate, the stage counts, the tableau, and the coupling construction agree to rounding; two that differ in any one of them produce different trajectories, even when both are stable and both converge at second order under refinement. In particular, a difference of one internal stage in a diffusion sub-step does not signal an error: both schemes are stable, both are second order, and both are legitimate members of the stabilized family. They are simply different methods, and the difference persists in the final answer.

Reading a single component of the final state, rather than a norm or an average, is a deliberate choice for verification purposes. Norms average over the domain and can conceal compensating errors; a single nodal value is sensitive to every stage of the pipeline that touches that node, including the advective coupling that transports information toward it and the boundary treatment near it. Nodes close to a fixed boundary are the most demanding in this respect, since it is there that the advective, diffusive, and reactive contributions are largest and most nearly in opposition, and where weakly coupled integrators lose accuracy first.

Returns
-------
float, the entry of the final state vector at target_index as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_pipeline(n_nodes: int, n_steps: int, T: float, c: float, d: float,
                 r: float, eps: float, A: float, B: float, d_var: float,
                 seed: int, rtol: float, atol: float, tau: float,
                 q_lambda: float, target_index: int) -> float:
    '''Run the full integration and return the target state entry.

    Parameters
    ----------
    n_nodes : int
        Number of spatial nodes per species, at least 3.
    n_steps : int
        Number of uniform time steps, positive.
    T : float
        Final time, positive.
    c : float
        Advection speed.
    d : float
        Diffusion strength scaling, positive.
    r : float
        Reaction rate scaling.
    eps : float
        Stiffness parameter of the third reaction channel, positive.
    A : float
        Constant concentration parameter A of the Brusselator source.
    B : float
        Constant concentration parameter B of the Brusselator source.
    d_var : float
        Amplitude of the sinusoidal variation of the diffusion coefficient.
    seed : int
        Seed for the eigenvalue estimator's initial iterate.
    rtol : float
        Relative tolerance entering the weighted root-mean-square weights.
    atol : float
        Absolute floor entering the weighted root-mean-square weights.
    tau : float
        Relative-change stopping tolerance of the eigenvalue estimator.
    q_lambda : float
        Safety factor applied to the eigenvalue estimate.
    target_index : int
        Zero-based index into the species-stacked final state vector.

    Returns
    -------
    value : float
        The entry of the final state at target_index, as a native Python float.
    '''
    return value  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_pipeline(n_nodes: int, n_steps: int, T: float, c: float, d: float,
                         r: float, eps: float, A: float, B: float, d_var: float,
                         seed: int, rtol: float, atol: float, tau: float,
                         q_lambda: float, target_index: int) -> float:
    """Reference implementation."""
    if int(n_nodes) != n_nodes or int(n_nodes) < 3:
        raise ValueError("n_nodes must be an integer of at least 3")
    if int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError("n_steps must be a positive integer")
    if not (np.isscalar(T) or np.ndim(T) == 0) or not np.isfinite(float(T)) \
            or float(T) <= 0.0:
        raise ValueError("T must be a finite positive scalar")
    if not (np.isscalar(d) or np.ndim(d) == 0) or not np.isfinite(float(d)) \
            or float(d) <= 0.0:
        raise ValueError("d must be a finite positive scalar")
    if int(target_index) != target_index \
            or not (0 <= int(target_index) < 3 * int(n_nodes)):
        raise ValueError("target_index must be an integer index into the state vector")
    n_nodes = int(n_nodes)
    n_steps = int(n_steps)
    T = float(T)
    target_index = int(target_index)

    sq2 = np.sqrt(2.0)
    gam = (2.0 - sq2) / 2.0
    dl = sq2 / 4.0
    Cv = np.array([0.0, 2.0 * gam, 2.0 * gam, 1.0, 1.0, 1.0])
    AI = np.zeros((6, 6))
    AE = np.zeros((6, 6))
    AE[1, 0] = 2.0 * gam
    AE[2, 0] = 2.0 * gam
    AE[3, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[3, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[4, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[4, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[5, 0] = dl
    AE[5, 2] = dl
    AE[5, 4] = gam
    AI[1, 0] = 2.0 * gam
    AI[2, 0] = gam
    AI[2, 2] = gam
    AI[3, 2] = 1.0
    AI[4, 0] = dl
    AI[4, 2] = dl
    AI[4, 4] = gam
    AI[5, 0] = dl
    AI[5, 2] = dl
    AI[5, 4] = gam
    dI = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
    dE = dI.copy()

    x = np.linspace(0.0, 1.0, n_nodes)
    dx = x[1] - x[0]
    D_nodes = float(d) * (1.0 + float(d_var) * np.sin(2.0 * np.pi * x))
    s0 = 0.1 * np.sin(2.0 * np.pi * x)
    y = np.concatenate([A + s0, B / A + s0, B + s0])

    comp0 = _oracle_compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)
    if comp0.shape != (3, 3 * n_nodes):
        raise ValueError("split operator returned an unexpected shape")
    for k in range(3):
        if comp0[2][k * n_nodes] != 0.0 or comp0[2][(k + 1) * n_nodes - 1] != 0.0:
            raise ValueError("stationary boundary condition violated in the "
                             "reaction component")

    eig = _oracle_estimate_dominant_eigenvalue(y, D_nodes, dx, seed, rtol, atol, tau, q_lambda)
    lam = float(eig[0])
    lam_eff = float(eig[1])
    if lam >= 0.0:
        raise ValueError("diffusion eigenvalue estimate must be negative")

    h = T / n_steps
    dcs = np.array([Cv[1] - Cv[0], Cv[3] - Cv[2]])
    counts = _oracle_select_rkl_stage_counts(h, dcs, lam_eff)
    if np.any(counts < 2):
        raise ValueError("stage counts must be at least 2")

    m = y.size
    Z = np.zeros((6, m))
    FR = np.zeros((6, m))
    FA = np.zeros((6, m))
    Z[0] = y
    FA[0] = comp0[0]
    FR[0] = comp0[2]
    sts_seen = 0
    for i in range(1, 6):
        g = _oracle_assemble_coupling_vector(h, AI, AE, dI, dE, i, FR, FA)
        dci = Cv[i] - Cv[i - 1]
        if AI[i, i] == 0.0 and dci > 0.0:
            H = dci * h
            Z[i] = _oracle_rkl2_super_step(Z[i - 1], H, int(counts[sts_seen]), g / H,
                                           D_nodes, dx)
            sts_seen += 1
        elif AI[i, i] == 0.0:
            Z[i] = Z[i - 1] + g
        else:
            Z[i] = _oracle_solve_reaction_stage(Z[i - 1] + g, h * AI[i, i], Z[i - 1],
                                                r, eps, A, B)
        comp = _oracle_compute_split_operators(Z[i], D_nodes, dx, c, r, eps, A, B)
        FA[i] = comp[0]
        FR[i] = comp[2]
    if sts_seen != counts.size:
        raise ValueError("stage count vector does not match the tableau structure")
    y = Z[5]

    for _ in range(n_steps - 1):
        out = _oracle_extsts_step(y, h, lam_eff, D_nodes, dx, c, r, eps, A, B)
        y = out[0]

    return float(y[target_index])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: locked configuration, the target scalar of the problem ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)",
            "gold_call": "_oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)",
        },
        # --- Normal: locked configuration read at a different node and species ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 100)",
            "gold_call": "_oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 100)",
        },
        # --- Normal: halved step size on the locked configuration ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(65, 146, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)",
            "gold_call": "_oracle_run_pipeline(65, 146, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)",
        },
        # --- Normal: different estimator seed and safety factor ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 4, 1e-4, 1e-11, 0.01, 1.2, 62)",
            "gold_call": "_oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 4, 1e-4, 1e-11, 0.01, 1.2, 62)",
        },
        # --- Boundary: target index at a fixed boundary node ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 0)",
            "gold_call": "_oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 0)",
        },
        # --- Boundary: single time step ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(33, 1, 0.02, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 20)",
            "gold_call": "_oracle_run_pipeline(33, 1, 0.02, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 20)",
        },
        # --- Edge: coarse grid, uniform diffusion coefficient, zero advection ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(17, 20, 0.5, 0.0, 0.05, 1.1, 0.02, 1.05, 2.9, 0.0, 7, 1e-4, 1e-11, 0.01, 1.1, 10)",
            "gold_call": "_oracle_run_pipeline(17, 20, 0.5, 0.0, 0.05, 1.1, 0.02, 1.05, 2.9, 0.0, 7, 1e-4, 1e-11, 0.01, 1.1, 10)",
        },
        # --- Edge: stiffer relaxation channel on a coarser grid ---
        {
            "setup": """import numpy as np
""",
            "call": "run_pipeline(33, 40, 0.8, 0.45, 0.1, 1.1, 1e-4, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 25)",
            "gold_call": "_oracle_run_pipeline(33, 40, 0.8, 0.45, 0.1, 1.1, 1e-4, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 25)",
        },
        # --- Invalid: fewer than three nodes ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_pipeline(2, 10, 1.0, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_pipeline(2, 10, 1.0, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive number of steps ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_pipeline(65, 0, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_pipeline(65, 0, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: target index outside the state vector ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 999)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 999)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive diffusion strength ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_pipeline(65, 73, 1.5, 0.45, 0.0, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_pipeline(65, 73, 1.5, 0.45, 0.0, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.003, 1.1, 62)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive stopping tolerance propagated from the estimator ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.0, 1.1, 62)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_pipeline(65, 73, 1.5, 0.45, 0.1, 1.1, 0.02, 1.05, 2.9, 0.9, 12, 1e-4, 1e-11, 0.0, 1.1, 62)
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
