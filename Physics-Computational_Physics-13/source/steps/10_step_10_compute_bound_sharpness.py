"""
Evaluate the theoretical eigenvalue bound implied by the spectral-equivalence constant and divide the observed spectral deviation by it.

The spectral theory of this preconditioner is derived under an idealisation: the constraint block of the preconditioner is taken to be the exact contraction of the V-cycle operator with the constraint Jacobian, with no clustering, no loop-cutting correction and no regularisation. Under that idealisation the analysis proceeds by scaling the system symmetrically so that the preconditioner becomes an involution and the coupling operator becomes a partial isometry, whose Gram product is the identity. Every eigenvalue then satisfies a scalar quadratic equation in two parameters, the Rayleigh quotient of the scaled mechanical block and the squared norm of the projection of the eigenvector onto the constraint directions. The first parameter is confined to an interval of half-width equal to the spectral-equivalence constant around one, and the second is confined between zero and one because the isometry property makes the projection an orthogonal projector. Maximising the larger root over that admissible rectangle puts the maximum at the corner where both parameters are extreme, which yields a closed-form bound on the distance of any eigenvalue from one that depends on nothing but the spectral-equivalence constant. For small values of that constant the bound behaves like the square root of the constant, plus one half of the constant, plus an eighth of its three-halves power, so the bound is not linear near zero: the leading term is the square root, and the bound is therefore much larger than the constant itself when the multigrid approximation is good.

The importance of the bound is that it contains no reference to the problem size, the number of constraints or the conditioning of the mechanical block. It therefore predicts that iteration counts stay bounded as the mesh is refined and as the time step shrinks, provided only that the multigrid component keeps its approximation quality. That is the theoretical basis for calling the method size-independent.

What the bound does not cover is precisely the part that makes the method practical. The block-diagonal sparsification, the low-rank correction on the cut constraints and the Tikhonov shift are all perturbations of the idealised constraint block, and none of them enters the derivation. Comparing the measured spectral deviation against the bound therefore quantifies how much of the true spectral spread is explained by the quality of the multigrid approximation alone. A ratio near or below one would say the idealisation is faithful and the multigrid quality is the binding constraint on convergence; a ratio well above one says the sparsification of the constraint block, not the multigrid, dominates the spectrum, and that improving the V-cycle would buy nothing until the Schur approximation is improved first.

Returns
-------
float: the ratio of the observed spectral deviation to the theoretical eigenvalue bound implied by gamma, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_bound_sharpness(gamma: float, deviation: float) -> float:
    """Divide the observed spectral deviation by its theoretical bound.

    Parameters
    ----------
    gamma : float
        Spectral-equivalence constant of the V-cycle, 0 <= gamma < 1.
    deviation : float
        Observed largest deviation from unity of the preconditioned spectrum
        (deviation >= 0).

    Returns
    -------
    sharpness : float
        Observed deviation divided by the theoretical bound implied by gamma,
        as a native Python float.
    """
    return sharpness  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_bound_sharpness(gamma: float, deviation: float) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(gamma, (int, float, np.floating, np.integer))
            and not isinstance(gamma, bool) and np.isfinite(gamma)
            and 0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must be a finite number in the half-open interval [0, 1)")
    if not (isinstance(deviation, (int, float, np.floating, np.integer))
            and not isinstance(deviation, bool) and np.isfinite(deviation)
            and float(deviation) >= 0.0):
        raise ValueError("deviation must be a finite number >= 0")

    gamma = float(gamma)
    bound = 0.5 * (gamma + np.sqrt(gamma * (4.0 + gamma)))
    if bound <= 0.0:
        raise ValueError("the theoretical bound vanishes, so the ratio is undefined")

    return float(float(deviation) / bound)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark-scale spectral equivalence (normal scenario) ---
        {
            "setup": """import numpy as np
gamma = 0.85
deviation = 7.5
""",
            "call": "compute_bound_sharpness(gamma, deviation)",
            "gold_call": "_oracle_compute_bound_sharpness(gamma, deviation)",
        },
        # --- Valid: high-quality V-cycle where the square-root term dominates ---
        {
            "setup": """import numpy as np
gamma = 1.0e-3
deviation = 0.05
""",
            "call": "compute_bound_sharpness(gamma, deviation)",
            "gold_call": "_oracle_compute_bound_sharpness(gamma, deviation)",
        },
        # --- Boundary: deviation exactly on the bound, giving a ratio of one ---
        {
            "setup": """import numpy as np
gamma = 0.5
deviation = 0.5 * (gamma + np.sqrt(gamma * (4.0 + gamma)))
""",
            "call": "compute_bound_sharpness(gamma, deviation)",
            "gold_call": "_oracle_compute_bound_sharpness(gamma, deviation)",
        },
        # --- Edge: vanishing observed deviation with a nearly degenerate V-cycle ---
        {
            "setup": """import numpy as np
gamma = 0.999
deviation = 0.0
""",
            "call": "compute_bound_sharpness(gamma, deviation)",
            "gold_call": "_oracle_compute_bound_sharpness(gamma, deviation)",
        },
        # --- Invalid: spectral-equivalence constant at or beyond unity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_bound_sharpness(1.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bound_sharpness(1.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: exact V-cycle, so the bound vanishes and the ratio is undefined ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_bound_sharpness(0.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bound_sharpness(0.0, 1.0)
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
