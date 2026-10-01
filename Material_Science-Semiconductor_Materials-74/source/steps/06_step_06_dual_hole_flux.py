"""
Compute the hole flux across the dual interface of the same interior diamond cell treated in the previous steps.



This is the dual-direction counterpart of the primal hole flux: it carries the dual edge-length prefactor on its diagonal contribution and the mixed prefactor with the normal projection on its cross-direction contribution, and it uses the same reversed Bernoulli arguments that distinguish holes from electrons.

The dual-direction hole flux, completing the set of four carrier fluxes carried by one interior diamond cell: electron and hole, each across the primal and the dual interface.

Returns
-------
native Python `float` containing the outward dual hole flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_dual_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    """Return the outward dual hole flux across the interface.

    Parameters
    ----------
    area : float
        Diamond-cell area |D|. Must be finite and > 0.
    coupling : float
        Normal projection c = n_KL . n_K*L*. Must be finite.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    p_k : float
        Hole density at cell K. Must be finite.
    p_l : float
        Hole density at cell L. Must be finite.
    p_kstar : float
        Hole density at cell K*. Must be finite.
    p_lstar : float
        Hole density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward dual hole flux across the interface, oriented from
        K to L (K* to L*).

    Raises
    ------
    ValueError
        If area, primal_length, dual_length or diffusion is not finite or is
        <= 0, or if coupling, any potential or any density is not finite.
    """
    return flux  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_dual_hole_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> float:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("diffusion", diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (coupling, psi_k, psi_l, psi_kstar, psi_lstar,
                   p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = _oracle_evaluate_bernoulli(-delta_primal) * p_k - _oracle_evaluate_bernoulli(delta_primal) * p_l
    dual_bracket = _oracle_evaluate_bernoulli(-delta_dual) * p_kstar - _oracle_evaluate_bernoulli(delta_dual) * p_lstar
    return float(
        primal_length * dual_length * diffusion * coupling * primal_bracket / (2.0 * area)
        + dual_length**2 * diffusion * dual_bracket / (2.0 * area)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the non-orthogonal diamond of the task (c = 0.5)
        {
            "setup": "import numpy as np\nargs = (1.454922678357857, 0.5, 1.6, 2.1, 0.85, 0.25, -0.35, 0.5, -0.15, 0.8, 1.5, 1.2, 0.6)\n",
            "call": "compute_dual_hole_flux(*args)",
            "gold_call": "_oracle_compute_dual_hole_flux(*args)",
        },
        # Boundary: orthogonal diamond, c = 0, cross-direction term drops out
        {
            "setup": "import numpy as np\nargs = (1.5, 0.0, 2.0, 1.5, 0.8, 0.3, -0.2, 0.7, -0.1, 1.0, 2.0, 0.5, 3.0)\n",
            "call": "compute_dual_hole_flux(*args)",
            "gold_call": "_oracle_compute_dual_hole_flux(*args)",
        },
        # Edge: near-zero potential jumps and a negative normal projection
        {
            "setup": "import numpy as np\nargs = (1.0, -0.4, 1.0, 2.0, 0.7, 1e-10, 0.0, -1e-10, 0.0, 1.2, 1.2, 0.7, 0.7)\n",
            "call": "compute_dual_hole_flux(*args)",
            "gold_call": "_oracle_compute_dual_hole_flux(*args)",
        },
        # Edge: strongly convection-dominated jumps, one Bernoulli factor near 0
        {
            "setup": "import numpy as np\nargs = (1.2, 0.25, 1.1, 1.9, 0.75, 18.0, -18.0, -22.0, 22.0, 0.5, 1.6, 1.1, 0.4)\n",
            "call": "compute_dual_hole_flux(*args)",
            "gold_call": "_oracle_compute_dual_hole_flux(*args)",
        },
        # Invalid: non-positive diamond area
        {
            "setup": """import numpy as np
args = (0.0, 0.5, 1.6, 2.1, 0.85, 0.25, -0.35, 0.5, -0.15, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_dual_hole_flux(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dual_hole_flux(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: non-positive diffusion coefficient
        {
            "setup": """import numpy as np
args = (1.454922678357857, 0.5, 1.6, 2.1, 0.0, 0.25, -0.35, 0.5, -0.15, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_dual_hole_flux(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dual_hole_flux(*args)
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
