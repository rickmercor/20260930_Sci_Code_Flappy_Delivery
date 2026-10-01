"""
Rebuild the four carrier fluxes of the same interior interface as seen from the neighbouring pair of cells.

Every interior interface is shared by two primal cells and two dual cells, and each side of it computes its own outward flux. Re-deriving the fluxes from the neighbour's point of view means exchanging the primal labels K and L, exchanging the dual labels K* and L*, and reversing both outward unit normals, then evaluating exactly the same discretization again. The diamond geometry itself is unchanged. This step must recompute all four fluxes from the swapped data rather than reuse the forward values, because the relation between the two orientations is the property being established, not an assumption.

Every interior interface is computed twice, once by each of the two pairs of control volumes that share it, and each side reports its own outward flux. Whether those two computations cancel is a property of the interface alone, independent of any source or recombination term, and it has to be established by re-evaluating the discretization as the neighbour sees it rather than assumed. It is the ingredient an assembled control-volume balance would later rely on, not a balance in itself. Seen by the neighbour, the two cell labels swap and both outward normals reverse; the cell geometry itself is unchanged.

Returns
-------
`numpy.ndarray` of shape `(4,)` containing the reverse-oriented `[primal_electron, dual_electron, primal_hole, dual_hole]` fluxes as native floats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_reversed_interface_fluxes(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
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
    """Return the four interface fluxes seen from the neighbouring cells.

    Parameters
    ----------
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    angle_radians : float
        Angle theta between the primal and dual directions, in radians. Must be
        finite and must give a strictly positive diamond area.
    primal_normal : np.ndarray
        Forward outward unit normal n_KL, shape (2,), finite.
    dual_normal : np.ndarray
        Forward outward unit normal n_K*L*, shape (2,), finite.
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
        Array of shape (4,) holding the reverse-oriented fluxes in the order
        [primal_electron, dual_electron, primal_hole, dual_hole].

    Raises
    ------
    ValueError
        If primal_length, dual_length, electron_diffusion or hole_diffusion is
        not finite or is <= 0, if angle_radians is not finite or gives a
        non-positive diamond area, if either normal is not a finite array of
        shape (2,), or if any potential or density is not finite.
    """
    return fluxes  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _diamond_flux(
    area, coupling, primal_length, dual_length, diffusion,
    psi_a, psi_b, psi_astar, psi_bstar,
    dens_a, dens_b, dens_astar, dens_bstar,
    charge_sign, primal_side,
):
    """One harmonic-average diamond flux for either carrier and either direction."""
    delta_primal = charge_sign * (float(psi_a) - float(psi_b))
    delta_dual = charge_sign * (float(psi_astar) - float(psi_bstar))
    primal_bracket = _oracle_evaluate_bernoulli(delta_primal) * dens_a - _oracle_evaluate_bernoulli(-delta_primal) * dens_b
    dual_bracket = _oracle_evaluate_bernoulli(delta_dual) * dens_astar - _oracle_evaluate_bernoulli(-delta_dual) * dens_bstar
    if primal_side:
        return float(
            primal_length**2 * diffusion * primal_bracket / (2.0 * area)
            + primal_length * dual_length * diffusion * coupling * dual_bracket / (2.0 * area)
        )
    return float(
        primal_length * dual_length * diffusion * coupling * primal_bracket / (2.0 * area)
        + dual_length**2 * diffusion * dual_bracket / (2.0 * area)
    )


def _oracle_compute_reversed_interface_fluxes(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
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
        ("primal_length", primal_length),
        ("dual_length", dual_length),
        ("electron_diffusion", electron_diffusion),
        ("hole_diffusion", hole_diffusion),
    ):
        if not np.isfinite(positive) or positive <= 0.0:
            raise ValueError(f"{label} must be finite and > 0")
    if not np.isfinite(angle_radians):
        raise ValueError("angle_radians must be finite")

    forward_primal = np.asarray(primal_normal, dtype=float)
    forward_dual = np.asarray(dual_normal, dtype=float)
    if forward_primal.shape != (2,) or forward_dual.shape != (2,):
        raise ValueError("Both normals must be arrays of shape (2,)")
    if not np.all(np.isfinite(forward_primal)) or not np.all(np.isfinite(forward_dual)):
        raise ValueError("Both normals must be finite")
    for finite in (psi_k, psi_l, psi_kstar, psi_lstar,
                   n_k, n_l, n_kstar, n_lstar, p_k, p_l, p_kstar, p_lstar):
        if not np.isfinite(finite):
            raise ValueError("All potentials and densities must be finite")

    area = 0.5 * float(primal_length) * float(dual_length) * np.sin(angle_radians)
    if area <= 0.0:
        raise ValueError("angle_radians must define a strictly positive diamond area")

    # Reverse both outward normals; their projection on each other is unchanged.
    reversed_primal = -forward_primal
    reversed_dual = -forward_dual
    coupling = float(np.dot(reversed_primal, reversed_dual))

    # Exchange K with L and K* with L*, then re-evaluate the same discretization.
    electron_primal = _diamond_flux(
        area, coupling, primal_length, dual_length, electron_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, n_l, n_k, n_lstar, n_kstar, 1.0, True,
    )
    electron_dual = _diamond_flux(
        area, coupling, primal_length, dual_length, electron_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, n_l, n_k, n_lstar, n_kstar, 1.0, False,
    )
    hole_primal = _diamond_flux(
        area, coupling, primal_length, dual_length, hole_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, p_l, p_k, p_lstar, p_kstar, -1.0, True,
    )
    hole_dual = _diamond_flux(
        area, coupling, primal_length, dual_length, hole_diffusion,
        psi_l, psi_k, psi_lstar, psi_kstar, p_l, p_k, p_lstar, p_kstar, -1.0, False,
    )
    return np.array(
        [electron_primal, electron_dual, hole_primal, hole_dual], dtype=float
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the non-orthogonal diamond of the task
        {
            "setup": "import numpy as np\nargs = (1.6, 2.1, np.pi / 3, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)\n",
            "call": "compute_reversed_interface_fluxes(*args)",
            "gold_call": "_oracle_compute_reversed_interface_fluxes(*args)",
        },
        # Boundary: orthogonal diamond, cross-direction terms drop out
        {
            "setup": "import numpy as np\nargs = (2.0, 1.5, np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 0.8, 0.3, -0.2, 0.7, -0.1, 2.0, 1.0, 3.0, 0.5, 1.0, 2.0, 0.5, 3.0)\n",
            "call": "compute_reversed_interface_fluxes(*args)",
            "gold_call": "_oracle_compute_reversed_interface_fluxes(*args)",
        },
        # Edge: near-zero potential jumps on a strongly distorted diamond
        {
            "setup": "import numpy as np\nargs = (1.0, 2.0, np.pi / 60, np.array([0.6, 0.8]), np.array([-0.8, 0.6]), 0.8, 0.7, 1e-10, 0.0, -1e-10, 0.0, 1.2, 1.2, 0.7, 0.7, 0.4, 0.4, 1.5, 1.5)\n",
            "call": "compute_reversed_interface_fluxes(*args)",
            "gold_call": "_oracle_compute_reversed_interface_fluxes(*args)",
        },
        # Edge: strongly convection-dominated potential jumps
        {
            "setup": "import numpy as np\nargs = (1.1, 1.9, np.pi / 4, np.array([1.0, 0.0]), np.array([np.sqrt(2) / 2, np.sqrt(2) / 2]), 0.95, 0.75, 18.0, -18.0, -22.0, 22.0, 1.4, 0.6, 0.9, 1.3, 0.5, 1.6, 1.1, 0.4)\n",
            "call": "compute_reversed_interface_fluxes(*args)",
            "gold_call": "_oracle_compute_reversed_interface_fluxes(*args)",
        },
        # Invalid: degenerate angle giving zero diamond area
        {
            "setup": """import numpy as np
args = (1.6, 2.1, 0.0, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_reversed_interface_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_reversed_interface_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: normal vector of the wrong shape
        {
            "setup": """import numpy as np
args = (1.6, 2.1, np.pi / 3, np.array([1.0, 0.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_reversed_interface_fluxes(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_reversed_interface_fluxes(*args)
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
