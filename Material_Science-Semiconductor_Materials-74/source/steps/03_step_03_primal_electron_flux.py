"""
Compute the electron flux across the primal interface of one interior diamond cell, for a locally conservative primal-dual finite-volume discretization of the stationary drift-diffusion model that does not assume mesh orthogonality.

The flux is assembled from two contributions. One acts along the primal edge direction and reproduces the classical exponentially fitted form built from the electron densities at K and L and the Bernoulli weights of the primal potential jump. The other is a cross-direction contribution built from the dual-cell densities at K* and L* and the Bernoulli weights of the dual potential jump; it enters weighted by the projection of the two outward normals on each other and by the mixed edge-length product, and it is the term that a one-dimensional edge-based scheme omits.

The electron flux leaving a primal cell across one interface. On an orthogonal mesh, this would be a single exponentially fitted term along the primal edge. On a general cell, the discrete gradient has a second component along the dual direction, and contracting it with the primal normal leaves a residual contribution that a one-dimensional edge scheme discards. Both contributions are of the same order, so dropping the second is not a small correction.

Returns
-------
native Python `float` containing the outward primal electron flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_primal_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
) -> float:
    """Return the outward primal electron flux across the interface.

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
        Electron diffusion coefficient. Must be finite and > 0.
    psi_k : float
        Electrostatic potential at primal cell K. Must be finite.
    psi_l : float
        Electrostatic potential at primal cell L. Must be finite.
    psi_kstar : float
        Electrostatic potential at dual cell K*. Must be finite.
    psi_lstar : float
        Electrostatic potential at dual cell L*. Must be finite.
    n_k : float
        Electron density at cell K. Must be finite.
    n_l : float
        Electron density at cell L. Must be finite.
    n_kstar : float
        Electron density at cell K*. Must be finite.
    n_lstar : float
        Electron density at cell L*. Must be finite.

    Returns
    -------
    flux : float
        The outward primal electron flux across the interface, oriented from
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

def _oracle_compute_primal_electron_flux(
    area: float,
    coupling: float,
    primal_length: float,
    dual_length: float,
    diffusion: float,
    psi_k: float,
    psi_l: float,
    psi_kstar: float,
    psi_lstar: float,
    n_k: float,
    n_l: float,
    n_kstar: float,
    n_lstar: float,
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
                   n_k, n_l, n_kstar, n_lstar):
        if not np.isfinite(finite):
            raise ValueError("coupling, potentials and densities must be finite")

    delta_primal = float(psi_k) - float(psi_l)
    delta_dual = float(psi_kstar) - float(psi_lstar)
    primal_bracket = _oracle_evaluate_bernoulli(delta_primal) * n_k - _oracle_evaluate_bernoulli(-delta_primal) * n_l
    dual_bracket = _oracle_evaluate_bernoulli(delta_dual) * n_kstar - _oracle_evaluate_bernoulli(-delta_dual) * n_lstar
    return float(
        primal_length**2 * diffusion * primal_bracket / (2.0 * area)
        + primal_length * dual_length * diffusion * coupling * dual_bracket / (2.0 * area)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the non-orthogonal diamond of the task (c = 0.5)
        {
            "setup": "import numpy as np\nargs = (1.454922678357857, 0.5, 1.6, 2.1, 1.15, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1)\n",
            "call": "compute_primal_electron_flux(*args)",
            "gold_call": "_oracle_compute_primal_electron_flux(*args)",
        },
        # Boundary: orthogonal diamond, c = 0, cross-direction term drops out
        {
            "setup": "import numpy as np\nargs = (1.5, 0.0, 2.0, 1.5, 1.0, 0.3, -0.2, 0.7, -0.1, 2.0, 1.0, 3.0, 0.5)\n",
            "call": "compute_primal_electron_flux(*args)",
            "gold_call": "_oracle_compute_primal_electron_flux(*args)",
        },
        # Edge: near-zero potential jumps and a negative normal projection
        {
            "setup": "import numpy as np\nargs = (1.0, -0.4, 1.0, 2.0, 0.8, 1e-10, 0.0, -1e-10, 0.0, 1.2, 1.2, 0.7, 0.7)\n",
            "call": "compute_primal_electron_flux(*args)",
            "gold_call": "_oracle_compute_primal_electron_flux(*args)",
        },
        # Edge: strongly convection-dominated jumps, one Bernoulli factor near 0
        {
            "setup": "import numpy as np\nargs = (1.2, 0.25, 1.1, 1.9, 0.95, 18.0, -18.0, -22.0, 22.0, 1.4, 0.6, 0.9, 1.3)\n",
            "call": "compute_primal_electron_flux(*args)",
            "gold_call": "_oracle_compute_primal_electron_flux(*args)",
        },
        # Invalid: non-positive diamond area
        {
            "setup": """import numpy as np
args = (0.0, 0.5, 1.6, 2.1, 1.15, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1)
def run_model():
    try:
        compute_primal_electron_flux(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_primal_electron_flux(*args)
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
args = (1.454922678357857, 0.5, 1.6, 2.1, 0.0, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1)
def run_model():
    try:
        compute_primal_electron_flux(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_primal_electron_flux(*args)
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
