"""
Compute, for one carrier, the two drift coefficients that the construction freezes on the edges of an interior diamond cell, and the defect that the construction introduces into the primal flux when it re-associates those coefficients with the two mesh directions.




Write z = +1 for electrons and z = -1 for holes, and define the carrier's scaled variable Phi = rho * exp(-z * psi) at each cell, where rho is the carrier density. On an edge whose two endpoints carry potentials a and b, the frozen coefficient is the reciprocal of the mean of exp(-z * u) along the edge, with u varying linearly between a and b, which evaluates to

    E(a, b) = exp(z * a) * B(z * (a - b)),    B(t) = t / (exp(t) - 1).

The primal edge joins the two dual cell centres and the dual edge joins the two primal cell centres, so the two coefficients of the cell are


    E_primal = E(psi_Kstar, psi_Lstar),    E_dual = E(psi_K, psi_L).

Keeping E_primal in both terms of the primal flux gives one flux; the construction instead places E_dual on the term along the primal direction. The cross-direction terms are identical, so the unpaired flux minus the construction's flux is

    defect = |sigma|^2 * D / (2 |D|) * (E_primal - E_dual) * (Phi_K - Phi_L).

Exponential fitting enters the construction through a drift coefficient frozen on each edge of the cell. The natural first choice attaches to both terms of the primal flux the coefficient of the edge being crossed, but that leaves the diagonal term paired with an exponential factor taken along the transverse direction, so it does not reduce to the classical one-dimensional form. The construction therefore re-associates the coefficients with directions. That choice is not free: it changes the flux by a product of two first-order differences, one in the frozen coefficients and one in the scaled carrier variable, which is formally second order in the mesh size but does not vanish on any finite cell. Measuring it on a concrete cell shows how much of the computed current is owed to the re-association rather than to the transport physics.

Returns
-------
`numpy.ndarray` of shape `(3,)` containing `[coefficient_primal_edge, coefficient_dual_edge, defect]` as native floats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_consistency_defect(
    area: float,
    primal_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    dens_k: float,
    dens_l: float,
    charge_sign: float,
) -> np.ndarray:
    """Return the two frozen edge coefficients and the primal-flux defect of one carrier.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    diffusion : float
        Carrier diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    dens_k, dens_l : float
        Carrier densities at the primal cells K and L. Must be finite.
    charge_sign : float
        +1.0 for electrons, -1.0 for holes.

    Returns
    -------
    result : np.ndarray
        Array of shape (3,) holding [coefficient_primal_edge,
        coefficient_dual_edge, defect], where defect is the unpaired primal
        flux minus the construction's primal flux.

    Raises
    ------
    ValueError
        If area, primal_length or diffusion is not finite or is <= 0, if
        charge_sign is not exactly +1.0 or -1.0, or if any potential or density
        is not finite.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _frozen_coefficient(potential_a: float, potential_b: float, sign: float) -> float:
    """Reciprocal mean of exp(-z u) along an edge, u linear between a and b."""
    return float(np.exp(sign * potential_a) * _oracle_evaluate_bernoulli(sign * (potential_a - potential_b)))


def _oracle_compute_consistency_defect(
    area: float,
    primal_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    dens_k: float,
    dens_l: float,
    charge_sign: float,
) -> np.ndarray:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    if charge_sign not in (1.0, -1.0):
        raise ValueError("charge_sign must be +1.0 for electrons or -1.0 for holes")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar, dens_k, dens_l):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    sign = float(charge_sign)
    coefficient_primal = _frozen_coefficient(psi_kstar, psi_lstar, sign)
    coefficient_dual = _frozen_coefficient(psi_k, psi_l, sign)
    phi_k = float(dens_k) * np.exp(-sign * float(psi_k))
    phi_l = float(dens_l) * np.exp(-sign * float(psi_l))
    defect = (
        primal_length**2 * diffusion
        * (coefficient_primal - coefficient_dual) * (phi_k - phi_l)
        / (2.0 * area)
    )
    return np.array([coefficient_primal, coefficient_dual, float(defect)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: electrons on the non-orthogonal diamond of the task
        {
            "setup": "import numpy as np\nargs = (1.454922678357857, 1.6, 1.15, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.0)\n",
            "call": "compute_consistency_defect(*args)",
            "gold_call": "_oracle_compute_consistency_defect(*args)",
        },
        # Normal: holes on the same diamond, whose coefficient carries exp(-psi)
        {
            "setup": "import numpy as np\nargs = (1.454922678357857, 1.6, 0.85, 0.25, -0.35, 0.5, -0.15, 0.8, 1.5, -1.0)\n",
            "call": "compute_consistency_defect(*args)",
            "gold_call": "_oracle_compute_consistency_defect(*args)",
        },
        # Boundary: equal potentials, so both coefficients agree and the defect vanishes
        {
            "setup": "import numpy as np\nargs = (2.0, 1.5, 1.0, 0.3, 0.3, 0.3, 0.3, 2.0, 1.0, 1.0)\n",
            "call": "compute_consistency_defect(*args)",
            "gold_call": "_oracle_compute_consistency_defect(*args)",
        },
        # Edge: near-zero potential jumps with unequal densities, holes
        {
            "setup": "import numpy as np\nargs = (1.0, 1.0, 0.8, 1e-10, 0.0, -1e-10, 0.0, 1.2, 0.7, -1.0)\n",
            "call": "compute_consistency_defect(*args)",
            "gold_call": "_oracle_compute_consistency_defect(*args)",
        },
        # Edge: strongly convection-dominated potential jumps, electrons
        {
            "setup": "import numpy as np\nargs = (1.2, 1.1, 0.95, 18.0, -18.0, -22.0, 22.0, 1.4, 0.6, 1.0)\n",
            "call": "compute_consistency_defect(*args)",
            "gold_call": "_oracle_compute_consistency_defect(*args)",
        },
        # Invalid: charge sign other than +1 or -1
        {
            "setup": """import numpy as np
args = (1.454922678357857, 1.6, 1.15, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 0.5)
def run_model():
    try:
        compute_consistency_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_consistency_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: non-positive diamond area
        {
            "setup": """import numpy as np
args = (0.0, 1.6, 1.15, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.0)
def run_model():
    try:
        compute_consistency_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_consistency_defect(*args)
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
