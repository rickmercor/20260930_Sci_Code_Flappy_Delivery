"""
Evaluate the analytic gradient of the all-atom q-TIP4P/F potential with respect to every atomic coordinate.

The relaxation follows this gradient and the curvature is built by differencing it, so it is the most heavily used object in the whole pipeline. It is required analytically rather than by differencing the energy, because it is itself differenced later to obtain the second derivatives, and differencing a differenced quantity would leave far too few significant figures for a six-figure answer.

Each term of the potential contributes through the chain rule on a scalar geometric coordinate. The two stretch terms and the Lennard-Jones term depend only on a distance, so each contributes its scalar derivative times the unit vector joining the pair, added to one atom and subtracted from the other. The bend term depends on the angle at the oxygen, whose derivative with respect to either arm is obtained by differentiating the normalised dot product and dividing by the sine of the angle; the oxygen picks up the negative sum of the two arm derivatives, which is what keeps the total force on an isolated monomer equal to zero.

The electrostatic term needs one extra step, and it is the one most easily got wrong. Forces accumulate on charge sites, and one of the three sites of each monomer is the massless site on the bisector, which is not an atom. Because that site is defined as a *linear* combination of the three atomic positions, its derivative with respect to each atom is a constant multiple of the identity: the force collected on it is distributed onto the oxygen with weight gamma and onto each hydrogen with weight one minus gamma over two, with no geometric Jacobian and no torque bookkeeping. Treating the site as if it were a fourth atom, or attaching its force to the oxygen alone, changes the gradient on every hydrogen and therefore the direction the relaxation takes.

A useful check on the whole construction is that the analytic result agrees with a central difference of the energy to about eight decimal places, and that the gradient summed over all atoms vanishes to machine precision, since no external field acts on the cluster.

Returns
-------
np.ndarray of shape (n_atoms, 3), float: the analytic Cartesian gradient of the q-TIP4P/F energy in kcal/(mol angstrom).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_potential_gradient(coords: np.ndarray, d_r: float = 116.09,
                               alpha_r: float = 2.287, r_eq: float = 0.9419,
                               k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                               epsilon: float = 0.1852, sigma: float = 3.1589,
                               q_m: float = 1.1128, gamma: float = 0.73612,
                               coulomb_constant: float = 332.06371) -> np.ndarray:
    """Evaluate the analytic Cartesian gradient of the q-TIP4P/F potential.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions, ordered O, H, H within each monomer.
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
    epsilon : float
        Oxygen-oxygen Lennard-Jones well depth in kcal/mol (epsilon > 0).
    sigma : float
        Oxygen-oxygen Lennard-Jones diameter in angstrom (sigma > 0).
    q_m : float
        Magnitude of the negative charge on the massless site, in elementary
        charges; each hydrogen carries half of it with the opposite sign
        (q_m > 0).
    gamma : float
        Fraction defining the position of the massless charge site along the
        bisector, strictly between 0 and 1.
    coulomb_constant : float
        Electrostatic prefactor in kcal angstrom /(mol e**2)
        (coulomb_constant > 0).

    Returns
    -------
    gradient : np.ndarray
        Array with the same shape as ``coords`` holding the derivative of the
        energy with respect to every atomic coordinate, in
        kcal/(mol angstrom).

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, does not contain complete O-H-H
        monomers, or contains non-finite entries; if a positive surface
        parameter is invalid; if ``theta_eq_deg`` or ``gamma`` lies outside
        its stated open interval; or if a required intersite distance is zero.
    """
    return gradient  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_potential_gradient(coords: np.ndarray, d_r: float = 116.09,
                                       alpha_r: float = 2.287, r_eq: float = 0.9419,
                                       k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                                       epsilon: float = 0.1852, sigma: float = 3.1589,
                                       q_m: float = 1.1128, gamma: float = 0.73612,
                                       coulomb_constant: float = 332.06371) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of O, H, H monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    for name, value in (("d_r", d_r), ("alpha_r", alpha_r), ("r_eq", r_eq),
                        ("k_theta", k_theta), ("epsilon", epsilon), ("sigma", sigma),
                        ("q_m", q_m), ("coulomb_constant", coulomb_constant)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(theta_eq_deg, (int, float, np.floating, np.integer))
            and not isinstance(theta_eq_deg, bool) and np.isfinite(theta_eq_deg)
            and 0.0 < float(theta_eq_deg) < 180.0):
        raise ValueError("theta_eq_deg must be a finite number strictly between 0 and 180")
    if not (isinstance(gamma, (int, float, np.floating, np.integer))
            and not isinstance(gamma, bool) and np.isfinite(gamma) and 0.0 < float(gamma) < 1.0):
        raise ValueError("gamma must be a finite number strictly between 0 and 1")

    d_r, alpha_r, r_eq = float(d_r), float(alpha_r), float(r_eq)
    k_theta, theta_eq = float(k_theta), np.deg2rad(float(theta_eq_deg))
    epsilon, sigma = float(epsilon), float(sigma)
    q_m, gamma, coulomb = float(q_m), float(gamma), float(coulomb_constant)

    n_monomers = coords.shape[0] // 3
    gradient = np.zeros_like(coords)

    # -- Intramolecular: two quartic-Morse stretches and one harmonic bend.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        for i_h in (i_h1, i_h2):
            offset = coords[i_h] - coords[i_o]
            bond = float(np.linalg.norm(offset))
            if bond <= 0.0:
                raise ValueError("an O-H bond has zero length")
            x = bond - r_eq
            d_energy = d_r * (2.0 * alpha_r ** 2 * x - 3.0 * alpha_r ** 3 * x ** 2
                              + (7.0 / 3.0) * alpha_r ** 4 * x ** 3)
            unit = offset / bond
            gradient[i_h] += d_energy * unit
            gradient[i_o] -= d_energy * unit

        arm_1 = coords[i_h1] - coords[i_o]
        arm_2 = coords[i_h2] - coords[i_o]
        len_1, len_2 = np.linalg.norm(arm_1), np.linalg.norm(arm_2)
        unit_1, unit_2 = arm_1 / len_1, arm_2 / len_2
        cosine = float(np.clip(np.dot(unit_1, unit_2), -1.0, 1.0))
        angle = np.arccos(cosine)
        sine = np.sqrt(max(1.0 - cosine * cosine, 1.0e-300))
        d_energy = k_theta * (angle - theta_eq)
        d_angle_1 = -(unit_2 - cosine * unit_1) / (len_1 * sine)
        d_angle_2 = -(unit_1 - cosine * unit_2) / (len_2 * sine)
        gradient[i_h1] += d_energy * d_angle_1
        gradient[i_h2] += d_energy * d_angle_2
        gradient[i_o] -= d_energy * (d_angle_1 + d_angle_2)

    # -- Intermolecular dispersion/repulsion: oxygen pairs only.
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            i_o, j_o = 3 * mono_i, 3 * mono_j
            offset = coords[j_o] - coords[i_o]
            distance = float(np.linalg.norm(offset))
            if distance <= 0.0:
                raise ValueError("two oxygen atoms coincide")
            ratio6 = (sigma / distance) ** 6
            d_energy = 4.0 * epsilon * (-12.0 * ratio6 * ratio6 + 6.0 * ratio6) / distance
            unit = offset / distance
            gradient[j_o] += d_energy * unit
            gradient[i_o] -= d_energy * unit

    # -- Intermolecular electrostatics on the H, H and M sites of each monomer.
    site_charges = np.array([0.5 * q_m, 0.5 * q_m, -q_m], dtype=float)
    sites = np.zeros((n_monomers, 3, 3), dtype=float)
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        sites[mono, 0] = coords[i_h1]
        sites[mono, 1] = coords[i_h2]
        sites[mono, 2] = gamma * coords[i_o] + (1.0 - gamma) * 0.5 * (coords[i_h1] + coords[i_h2])

    site_gradient = np.zeros_like(sites)
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            for site_a in range(3):
                for site_b in range(3):
                    offset = sites[mono_j, site_b] - sites[mono_i, site_a]
                    distance = float(np.linalg.norm(offset))
                    if distance <= 0.0:
                        raise ValueError("two charge sites coincide")
                    scale = coulomb * site_charges[site_a] * site_charges[site_b]
                    d_energy = -scale / distance ** 2
                    unit = offset / distance
                    site_gradient[mono_j, site_b] += d_energy * unit
                    site_gradient[mono_i, site_a] -= d_energy * unit

    # The massless site is a fixed linear combination of the three atoms, so its
    # force is distributed by constant weights and carries no geometric factor.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        gradient[i_h1] += site_gradient[mono, 0] + (1.0 - gamma) * 0.5 * site_gradient[mono, 2]
        gradient[i_h2] += site_gradient[mono, 1] + (1.0 - gamma) * 0.5 * site_gradient[mono, 2]
        gradient[i_o] += gamma * site_gradient[mono, 2]

    return gradient

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark cyclic tetramer (normal scenario) ---
        {
            "setup": """import numpy as np
coords = np.array([
    [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
    [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
    [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
    [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
    [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
    [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]])
""",
            "call": "compute_potential_gradient(coords)",
            "gold_call": "_oracle_compute_potential_gradient(coords)",
        },
        # --- Valid: a distorted single monomer, isolating the intramolecular
        #     stretch and bend derivatives ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.75, 0.0, 0.60], [-0.78, 0.0, 0.55]])
""",
            "call": "compute_potential_gradient(coords)",
            "gold_call": "_oracle_compute_potential_gradient(coords)",
        },
        # --- Boundary: a monomer already at the reference geometry, where the
        #     intramolecular gradient vanishes and only round-off remains ---
        {
            "setup": """import numpy as np
half = np.deg2rad(107.4) / 2.0
coords = np.array([[0.0, 0.0, 0.0],
                   [0.9419 * np.sin(half), 0.0, 0.9419 * np.cos(half)],
                   [-0.9419 * np.sin(half), 0.0, 0.9419 * np.cos(half)]])
""",
            "call": "compute_potential_gradient(coords)",
            "gold_call": "_oracle_compute_potential_gradient(coords)",
        },
        # --- Edge: a close dimer with an altered bisector fraction, which
        #     changes how the massless-site force is redistributed ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [-0.35, 1.05, 0.0],
    [2.45, 0.10, 0.05], [3.30, 0.35, 0.40], [2.30, -0.60, 0.70]])
""",
            "call": "compute_potential_gradient(coords, 120.0, 2.4, 0.96, 90.0, 104.0, 0.2, 3.2, 1.05, 0.5)",
            "gold_call": "_oracle_compute_potential_gradient(coords, 120.0, 2.4, 0.96, 90.0, 104.0, 0.2, 3.2, 1.05, 0.5)",
        },
        # --- Invalid: atom count is not a whole number of monomers ---
        {
            "setup": """import numpy as np
coords = np.zeros((5, 3))
def run_model():
    try:
        compute_potential_gradient(coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_gradient(coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite coordinate ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [np.nan, 0.8983, 0.0]])
def run_model():
    try:
        compute_potential_gradient(coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_gradient(coords)
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
