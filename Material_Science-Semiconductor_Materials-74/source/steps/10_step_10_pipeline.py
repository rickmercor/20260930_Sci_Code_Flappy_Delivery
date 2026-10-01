"""
Run the complete interior-diamond diagnostic end to end and return the requested scalar.

The orchestrator chains every preceding step: it forms the diamond area and the normal projection, evaluates the four carrier fluxes of the non-orthogonal diamond, rebuilds the same four fluxes seen by the neighbouring cells and checks that each forward/reverse pair cancels to machine precision, evaluates the auxiliary orthogonal diamond and checks that its decoupled fluxes are finite, and evaluates the primal-flux defect of each carrier. The electron flux carries the sign opposite to the electron current, so the signed electrical current combines the two carriers as -F(n) + F(p); the value returned is the same signed combination of the two defects.

The end-to-end diagnostic. The four fluxes of the non-orthogonal cell are assembled and confirmed to cancel pairwise against the neighbour's view of the same interface, and the perpendicular limit is checked to stay finite. The quantity reported is not a current but the part of the signed electrical current that the re-association of the frozen coefficients changes. The two carriers do not enter with the same sign: a carrier flux counts particles, and electrons carry negative charge, so the electron flux runs against the electron current while the hole flux runs with it. The result is a direct, cell-level measure of the second-order consistency error the scheme accepts in exchange for recovering the classical form along each direction.

Returns
-------
native Python `float` containing the defect of the signed electrical current through the primal interface of the non-orthogonal diamond, in the same scaled units as the carrier fluxes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_signed_current_defect(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    angle_radians_perp: float,
    primal_normal_perp: np.ndarray,
    dual_normal_perp: np.ndarray,
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
) -> float:
    """Return the defect of the signed electrical current across the primal interface.

    Parameters
    ----------
    primal_length : float
        Primal edge length |sigma|. Must be finite and > 0.
    dual_length : float
        Dual edge length |sigma*|. Must be finite and > 0.
    angle_radians : float
        Angle theta of the non-orthogonal diamond, in radians. Must be finite
        and must give a strictly positive area.
    primal_normal : np.ndarray
        Outward unit normal n_KL of the non-orthogonal diamond, shape (2,).
    dual_normal : np.ndarray
        Outward unit normal n_K*L* of the non-orthogonal diamond, shape (2,).
    angle_radians_perp : float
        Angle theta_perp of the auxiliary diamond, in radians. Must be finite
        and must give a strictly positive area.
    primal_normal_perp : np.ndarray
        Outward unit normal of the auxiliary diamond, shape (2,).
    dual_normal_perp : np.ndarray
        Outward unit normal of the auxiliary diamond, shape (2,).
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
    current_defect : float
        The signed combination -delta_n + delta_p of the two primal-flux
        defects, where each defect is the flux obtained by keeping the
        primal-edge frozen coefficient in both of its terms minus the
        construction's primal flux. The signs follow from the electron carrier
        flux representing -J_n / q and the hole carrier flux representing
        +J_p / q.

    Raises
    ------
    ValueError
        If primal_length, dual_length, electron_diffusion or hole_diffusion is
        not finite or is <= 0, if either angle is not finite or gives a
        non-positive diamond area, if any normal is not a finite array of shape
        (2,), if any potential or density is not finite, if any exponential
        fitting weight is not finite and strictly positive, or if the
        forward/reverse interface fluxes fail to cancel pairwise to within
        1e-12.
    """
    return current_defect  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_signed_current_defect(
    primal_length: float,
    dual_length: float,
    angle_radians: float,
    primal_normal: np.ndarray,
    dual_normal: np.ndarray,
    angle_radians_perp: float,
    primal_normal_perp: np.ndarray,
    dual_normal_perp: np.ndarray,
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
) -> float:
    for label, positive in (
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

    psi = (psi_k, psi_l, psi_kstar, psi_lstar)
    n = (n_k, n_l, n_kstar, n_lstar)
    p = (p_k, p_l, p_kstar, p_lstar)

    # Step 2: the exponential fitting weights of both directions must be usable.
    weights = [
        _oracle_evaluate_bernoulli(psi_k - psi_l),
        _oracle_evaluate_bernoulli(psi_l - psi_k),
        _oracle_evaluate_bernoulli(psi_kstar - psi_lstar),
        _oracle_evaluate_bernoulli(psi_lstar - psi_kstar),
    ]
    if not all(np.isfinite(w) and w > 0.0 for w in weights):
        raise ValueError("Exponential fitting weights must be finite and strictly positive")

    # Step 1: geometry of the non-orthogonal diamond.
    area, coupling = _oracle_compute_diamond_metrics(
        primal_length, dual_length, angle_radians, primal_normal, dual_normal
    )
    area, coupling = float(area), float(coupling)

    # Steps 3-6: the four carrier fluxes in the forward orientation.
    forward = np.array(
        [
            _oracle_compute_primal_electron_flux(
                area, coupling, primal_length, dual_length, electron_diffusion, *psi, *n),
            _oracle_compute_dual_electron_flux(
                area, coupling, primal_length, dual_length, electron_diffusion, *psi, *n),
            _oracle_compute_primal_hole_flux(
                area, coupling, primal_length, dual_length, hole_diffusion, *psi, *p),
            _oracle_compute_dual_hole_flux(
                area, coupling, primal_length, dual_length, hole_diffusion, *psi, *p),
        ],
        dtype=float,
    )

    # Step 7: the same interface seen by the neighbouring cells must cancel pairwise.
    reverse = np.asarray(
        _oracle_compute_reversed_interface_fluxes(
            primal_length, dual_length, angle_radians, primal_normal, dual_normal,
            electron_diffusion, hole_diffusion, *psi, *n, *p),
        dtype=float,
    )
    if not np.all(np.abs(forward + reverse) <= 1.0e-12 * (1.0 + np.abs(forward))):
        raise ValueError("Interface fluxes do not cancel pairwise under orientation reversal")

    # Step 8: the auxiliary orthogonal diamond must give finite decoupled fluxes.
    area_perp, _ = _oracle_compute_diamond_metrics(
        primal_length, dual_length, angle_radians_perp,
        primal_normal_perp, dual_normal_perp,
    )
    orthogonal = np.asarray(
        _oracle_compute_orthogonal_sg_fluxes(
            float(area_perp), primal_length, dual_length,
            electron_diffusion, hole_diffusion, *psi, *n, *p),
        dtype=float,
    )
    if not np.all(np.isfinite(orthogonal)):
        raise ValueError("Orthogonal-limit fluxes must be finite")

    # Step 9: primal-flux defect of each carrier.
    electron_defect = np.asarray(
        _oracle_compute_consistency_defect(
            area, primal_length, electron_diffusion, *psi, n_k, n_l, 1.0),
        dtype=float,
    )[2]
    hole_defect = np.asarray(
        _oracle_compute_consistency_defect(
            area, primal_length, hole_diffusion, *psi, p_k, p_l, -1.0),
        dtype=float,
    )[2]

    # Step 10: defect of the signed electrical current. The electron carrier flux
    # represents -J_n / q and the hole carrier flux +J_p / q, so the carriers enter
    # the current with opposite signs.
    return float(-electron_defect + hole_defect)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the full configuration of the task
        {
            "setup": "import numpy as np\nargs = (1.6, 2.1, np.pi / 3, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)\n",
            "call": "compute_signed_current_defect(*args)",
            "gold_call": "_oracle_compute_signed_current_defect(*args)",
        },
        # Boundary: the main diamond is itself orthogonal, so coupling vanishes
        {
            "setup": "import numpy as np\nargs = (2.0, 1.5, np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 0.8, 0.3, -0.2, 0.7, -0.1, 2.0, 1.0, 3.0, 0.5, 1.0, 2.0, 0.5, 3.0)\n",
            "call": "compute_signed_current_defect(*args)",
            "gold_call": "_oracle_compute_signed_current_defect(*args)",
        },
        # Edge: near-zero potential jumps on a strongly distorted diamond
        {
            "setup": "import numpy as np\nargs = (1.0, 2.0, np.pi / 60, np.array([0.6, 0.8]), np.array([-0.8, 0.6]), np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.8, 0.7, 1e-10, 0.0, -1e-10, 0.0, 1.2, 0.7, 0.7, 0.7, 0.4, 1.3, 1.5, 1.5)\n",
            "call": "compute_signed_current_defect(*args)",
            "gold_call": "_oracle_compute_signed_current_defect(*args)",
        },
        # Edge: strongly convection-dominated potential jumps
        {
            "setup": "import numpy as np\nargs = (1.1, 1.9, np.pi / 4, np.array([1.0, 0.0]), np.array([np.sqrt(2) / 2, np.sqrt(2) / 2]), np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.95, 0.75, 18.0, -18.0, -22.0, 22.0, 1.4, 0.6, 0.9, 1.3, 0.5, 1.6, 1.1, 0.4)\n",
            "call": "compute_signed_current_defect(*args)",
            "gold_call": "_oracle_compute_signed_current_defect(*args)",
        },
        # Invalid: non-positive electron diffusion coefficient
        {
            "setup": """import numpy as np
args = (1.6, 2.1, np.pi / 3, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), np.pi / 2, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_signed_current_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signed_current_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: degenerate auxiliary diamond with zero area
        {
            "setup": """import numpy as np
args = (1.6, 2.1, np.pi / 3, np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2]), 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.15, 0.85, 0.25, -0.35, 0.5, -0.15, 1.7, 0.9, 1.3, 1.1, 0.8, 1.5, 1.2, 0.6)
def run_model():
    try:
        compute_signed_current_defect(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_signed_current_defect(*args)
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
