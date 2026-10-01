"""
Evaluate the same four carrier fluxes on the auxiliary diamond whose primal and dual outward normals are orthogonal.


Orthogonal normals make the projection between the two directions vanish, so the cross-direction contribution of every flux disappears and each flux collapses to a single exponentially fitted edge term along its own direction. Recovering this limit is the consistency check that ties the non-orthogonal discretization back to the classical edge-based scheme it generalizes. The auxiliary diamond keeps the edge lengths, potentials, densities and diffusion coefficients of the original one and changes only the angle and the two normals, so its area is not the same as the non-orthogonal one and has to be recomputed.

The consistency check that ties the construction back to the classical scheme it generalises. When the two directions of the cell are perpendicular, the transverse coupling disappears, and each flux collapses to a single exponentially fitted edge term, recovering the standard edge-based method one direction at a time. The auxiliary cell keeps every physical parameter of the original and changes only the angle and the two normals, so its area is not the same and has to be recomputed.

Returns
-------
`numpy.ndarray` of shape `(4,)` containing the orthogonal-limit `[primal_electron, dual_electron, primal_hole, dual_hole]` fluxes as native floats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_orthogonal_sg_fluxes(
    area: float,
    primal_length: float,
    dual_length: float,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    """Return the four decoupled carrier fluxes of the orthogonal diamond.

    Parameters
    ----------
    area : float
        Area |D_perp| of the auxiliary orthogonal diamond. Must be finite
        and > 0.
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    electron_diffusion : float
        Electron diffusion coefficient. Must be finite and > 0.
    hole_diffusion : float
        Hole diffusion coefficient. Must be finite and > 0.
    psi_k, psi_l, psi_kstar, psi_lstar : float
        Electrostatic potentials at cells K, L, K* and L*. Must be finite.
    n_k, n_l, n_kstar, n_lstar : float
        Electron densities at cells K, L, K* and L*. Must be finite.
    p_k, p_l, p_kstar, p_lstar : float
        Hole densities at cells K, L, K* and L*. Must be finite.

    Returns
    -------
    fluxes : np.ndarray
        Array of shape (4,) holding the orthogonal-limit fluxes in the order
        [primal_electron, dual_electron, primal_hole, dual_hole].

    Raises
    ------
    ValueError
        If area, primal_length, dual_length, electron_diffusion or
        hole_diffusion is not finite or is <= 0, or if any potential or density
        is not finite.
    """
    return fluxes  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _edge_flux(edge_length, area, diffusion, delta_psi, dens_a, dens_b, charge_sign):
    """One decoupled exponentially fitted edge flux."""
    t = charge_sign * float(delta_psi)
    bracket = _oracle_evaluate_bernoulli(t) * dens_a - _oracle_evaluate_bernoulli(-t) * dens_b
    return float(edge_length**2 * diffusion * bracket / (2.0 * area))


def _oracle_compute_orthogonal_sg_fluxes(
    area: float,
    primal_length: float,
    dual_length: float,
    electron_diffusion: float,
    hole_diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
    p_k: float,
    p_l: float,
    p_kstar: float,
    p_lstar: float,
) -> np.ndarray:
    for label, positive in (
        ("area", area),
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("electron_diffusion", electron_diffusion),
        ("hole_diffusion", hole_diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar, p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    return np.array(
        [
            _edge_flux(primal_length, area, electron_diffusion, delta_primal, n_k, n_l, 1.0),
            _edge_flux(dual_length, area, electron_diffusion, delta_dual, n_kstar, n_lstar, 1.0),
            _edge_flux(primal_length, area, hole_diffusion, delta_primal, p_k, p_l, -1.0),
            _edge_flux(dual_length, area, hole_diffusion, delta_dual, p_kstar, p_lstar, -1.0),
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the auxiliary orthogonal diamond of the task, |D_perp| = 1.68
        {
            "setup": "import numpy as np\nargs = (1.68, 1.6, 2.1, 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)\n",
            "call": "compute_orthogonal_sg_fluxes(*args)",
            "gold_call": "_oracle_compute_orthogonal_sg_fluxes(*args)",
        },
        # Boundary: equal potentials, where every Bernoulli factor equals one
        {
            "setup": "import numpy as np\nargs = (1.5, 2.0, 1.5, 1.0, 0.8, 0.4, 0.4, -0.2, -0.2, 2.0, 1.0, 3.0, 0.5, 1.0, 2.0, 0.5, 3.0)\n",
            "call": "compute_orthogonal_sg_fluxes(*args)",
            "gold_call": "_oracle_compute_orthogonal_sg_fluxes(*args)",
        },
        # Edge: near-zero potential jumps probing the cancellation-safe branch
        {
            "setup": "import numpy as np\nargs = (1.0, 1.0, 2.0, 0.8, 0.7, 1e-10, 0.0, -1e-10, 0.0, 1.2, 1.2, 0.7, 0.7, 0.4, 0.4, 1.5, 1.5)\n",
            "call": "compute_orthogonal_sg_fluxes(*args)",
            "gold_call": "_oracle_compute_orthogonal_sg_fluxes(*args)",
        },
        # Edge: strongly convection-dominated jumps, one Bernoulli factor near 0
        {
            "setup": "import numpy as np\nargs = (1.2, 1.1, 1.9, 0.95, 0.75, 18.0, -18.0, -22.0, 22.0, 1.4, 0.6, 0.9, 1.3, 0.5, 1.6, 1.1, 0.4)\n",
            "call": "compute_orthogonal_sg_fluxes(*args)",
            "gold_call": "_oracle_compute_orthogonal_sg_fluxes(*args)",
        },
        # Invalid: non-positive diamond area
        {
            "setup": """import numpy as np
args = (0.0, 1.6, 2.1, 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_orthogonal_sg_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_orthogonal_sg_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: non-finite hole density
        {
            "setup": """import numpy as np
args = (1.68, 1.6, 2.1, 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, np.nan, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_orthogonal_sg_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_orthogonal_sg_fluxes(*args)
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
