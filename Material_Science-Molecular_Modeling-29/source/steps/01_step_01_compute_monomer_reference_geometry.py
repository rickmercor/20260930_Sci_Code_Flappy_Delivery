"""
Minimize the intramolecular potential of a single isolated water monomer and return the resulting oxygen-centred Cartesian geometry in the Jacobi body frame of Paper I.

Rigidification of a molecular cluster rests on a bijection between an all-atom monomer configuration and a pair consisting of a laboratory-frame position and orientation on the one hand, and a set of body-frame internal coordinates on the other. The coarse-grained description keeps only the first half of that pair. To turn a coarse-grained configuration back into a point in the full atomic configuration space one therefore needs a distinguished internal geometry to attach to every monomer, and the choice that makes the construction unique is the geometry that minimizes the potential energy of the monomer in isolation, that is, with every intermolecular term switched off.

For a flexible four-site water model the isolated-monomer energy separates into two independent bond-stretch terms and one bend term, so the minimization is over the two O-H distances and the H-O-H angle. The stretch is a quartic expansion of a Morse function, truncated so that the molecule cannot dissociate; because the expansion is taken about the Morse minimum, the linear term is absent and the stationary point sits exactly at the equilibrium bond length. The bend is harmonic, so its minimum is the equilibrium angle. A numerical minimization is used here rather than an appeal to those two observations, because the construction must remain valid for stretch and bend functions whose minima are not known in closed form.

The geometry is returned in the Jacobi body frame used by Paper I. The oxygen, which is the first and heaviest atom, is the monomer position and therefore sits at the origin. The H-O-H bisector is the positive y-axis, and the projected H2-H1 vector is the positive x-axis, so the hydrogens lie in the xy-plane with H1 on the negative-x side and H2 on the positive-x side. This convention makes the reference geometry compatible with the position-orientation map used to rebuild the zeroth-order manifold.

Returns
-------
np.ndarray of shape (3, 3), float: the oxygen-centred Jacobi-body-frame Cartesian geometry of the energy-minimized isolated monomer, in angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_monomer_reference_geometry(d_r: float = 116.09, alpha_r: float = 2.287,
                                       r_eq: float = 0.9419, k_theta: float = 87.85,
                                       theta_eq_deg: float = 107.4) -> np.ndarray:
    """Return the isolated-monomer minimum geometry in its Jacobi body frame.

    Parameters
    ----------
    d_r : float
        Dissociation energy of the quartic Morse O-H stretch in kcal/mol
        (d_r > 0).
    alpha_r : float
        Morse range parameter of the O-H stretch in inverse angstrom
        (alpha_r > 0).
    r_eq : float
        Equilibrium O-H bond length in angstrom (r_eq > 0).
    k_theta : float
        Harmonic force constant of the H-O-H bend in kcal/(mol rad**2)
        (k_theta > 0).
    theta_eq_deg : float
        Equilibrium H-O-H angle in degrees, strictly between 0 and 180.

    Returns
    -------
    geometry : np.ndarray
        Array of shape (3, 3) in angstrom holding the O, H and H positions of
        the energy-minimized isolated monomer, with the oxygen at the origin,
        the H-O-H bisector along +y and the projected H2-H1 vector along +x.

    Raises
    ------
    ValueError
        If a stretch or bend parameter is non-finite or non-positive, if
        ``theta_eq_deg`` is outside the open interval (0, 180), or if the
        isolated-monomer minimization cannot produce a physical geometry.
    """
    return geometry  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_monomer_reference_geometry(d_r: float = 116.09, alpha_r: float = 2.287,
                                               r_eq: float = 0.9419, k_theta: float = 87.85,
                                               theta_eq_deg: float = 107.4) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("d_r", d_r), ("alpha_r", alpha_r), ("r_eq", r_eq),
                        ("k_theta", k_theta)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(theta_eq_deg, (int, float, np.floating, np.integer))
            and not isinstance(theta_eq_deg, bool) and np.isfinite(theta_eq_deg)
            and 0.0 < float(theta_eq_deg) < 180.0):
        raise ValueError("theta_eq_deg must be a finite number strictly between 0 and 180")

    d_r, alpha_r, r_eq = float(d_r), float(alpha_r), float(r_eq)
    k_theta = float(k_theta)
    theta_eq = np.deg2rad(float(theta_eq_deg))

    # Newton minimization of the separable isolated-monomer energy in the three
    # internal coordinates, using the first two derivatives of the quartic Morse
    # stretch. Started away from the analytic stationary point so that the
    # iteration is genuinely exercised.
    bonds = np.array([1.05 * r_eq, 0.95 * r_eq], dtype=float)
    angle = theta_eq + 0.15
    for _ in range(200):
        shift = 0.0
        for index in range(2):
            x = bonds[index] - r_eq
            first = d_r * (2.0 * alpha_r ** 2 * x - 3.0 * alpha_r ** 3 * x ** 2
                           + (7.0 / 3.0) * alpha_r ** 4 * x ** 3)
            second = d_r * (2.0 * alpha_r ** 2 - 6.0 * alpha_r ** 3 * x
                            + 7.0 * alpha_r ** 4 * x ** 2)
            if second <= 0.0:
                raise ValueError("stretch curvature is non-positive; cannot minimize")
            step = first / second
            bonds[index] -= step
            shift = max(shift, abs(step))
        step = (k_theta * (angle - theta_eq)) / k_theta
        angle -= step
        shift = max(shift, abs(step))
        if shift < 1.0e-15:
            break
    else:
        raise ValueError("isolated-monomer minimization failed to converge")

    if np.any(bonds <= 0.0) or not (0.0 < angle < np.pi):
        raise ValueError("minimization left the physical range of the internal coordinates")

    # Paper-I Jacobi body frame: O is the position, the H-O-H bisector is +y,
    # and H2-H1 is +x.  For the optimized symmetric monomer those axes are
    # exactly orthogonal.
    half = 0.5 * angle
    geometry = np.array([
        [0.0, 0.0, 0.0],
        [-bonds[0] * np.sin(half), bonds[0] * np.cos(half), 0.0],
        [bonds[1] * np.sin(half), bonds[1] * np.cos(half), 0.0],
    ], dtype=float)
    return geometry

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: published q-TIP4P/F parameters (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_monomer_reference_geometry()",
            "gold_call": "_oracle_compute_monomer_reference_geometry()",
        },
        # --- Valid: a much stiffer, shorter-bonded monomer ---
        {
            "setup": """import numpy as np
d_r = 250.0
alpha_r = 3.1
r_eq = 0.85
k_theta = 140.0
theta_eq_deg = 98.0
""",
            "call": "compute_monomer_reference_geometry(d_r, alpha_r, r_eq, k_theta, theta_eq_deg)",
            "gold_call": "_oracle_compute_monomer_reference_geometry(d_r, alpha_r, r_eq, k_theta, theta_eq_deg)",
        },
        # --- Boundary: nearly linear monomer, where the bisector convention is
        #     most sensitive to the half-angle placement ---
        {
            "setup": """import numpy as np
theta_eq_deg = 179.0
""",
            "call": "compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, theta_eq_deg)",
            "gold_call": "_oracle_compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, theta_eq_deg)",
        },
        # --- Edge: an acute but still physical monomer ---
        {
            "setup": """import numpy as np
theta_eq_deg = 1.0
""",
            "call": "compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, theta_eq_deg)",
            "gold_call": "_oracle_compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, theta_eq_deg)",
        },
        # --- Invalid: non-positive dissociation energy ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_monomer_reference_geometry(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_monomer_reference_geometry(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: equilibrium angle outside the physical range ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, 180.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_monomer_reference_geometry(116.09, 2.287, 0.9419, 87.85, 180.0)
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
