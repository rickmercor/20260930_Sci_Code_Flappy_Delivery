"""
Evaluate the all-atom q-TIP4P/F potential energy of a water cluster.

The reported quantity is built from differences of the all-atom energy, so the surface has to be right before anything else can be. The surface used here is the q-TIP4P/F flexible water model, a four-site potential built on the rigid TIP4P/2005 geometry and charges with anharmonic intramolecular terms added.

The intramolecular part contributes, for each monomer, two O-H stretches and one H-O-H bend. The stretch is a quartic expansion of a Morse function about the equilibrium bond length, retaining the quadratic, cubic and quartic terms with the coefficient of the quartic term fixed at seven twelfths; the truncation is deliberate, since it reproduces the anharmonicity responsible for the broad O-H stretching band while removing the dissociation limit that a simple empirical model would describe badly. The bend is harmonic in the angle itself, not in a distance.

The intermolecular part is where the four-site structure matters. Dispersion and repulsion act only between oxygen atoms, through a single Lennard-Jones term. The electrostatics act between charge sites that are not all atoms: each monomer carries half the model charge on each hydrogen and the full negative charge on a massless site placed on the H-O-H bisector, at a fixed fraction of the way from the oxygen towards the midpoint of the two hydrogens. That site carries no Lennard-Jones term and the oxygen carries no charge. Only pairs of sites belonging to different monomers are included; intramolecular electrostatics are already absorbed into the fitted intramolecular terms, and including them would double-count.

There is no cutoff and no periodicity: the cluster sits in vacuum and every pair interaction is retained, so the energy is a plain double sum.

Returns
-------
float: the total q-TIP4P/F potential energy of the cluster in kcal/mol, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_potential_energy(coords: np.ndarray, d_r: float = 116.09,
                             alpha_r: float = 2.287, r_eq: float = 0.9419,
                             k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                             epsilon: float = 0.1852, sigma: float = 3.1589,
                             q_m: float = 1.1128, gamma: float = 0.73612,
                             coulomb_constant: float = 332.06371) -> float:
    """Evaluate the q-TIP4P/F potential energy of a water cluster.

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
    energy : float
        Total potential energy of the cluster in kcal/mol, as a native Python
        float.

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, does not contain complete O-H-H
        monomers, or contains non-finite entries; if a positive surface
        parameter is invalid; if ``theta_eq_deg`` or ``gamma`` lies outside
        its stated open interval; or if a required intersite distance is zero.
    """
    return energy  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_potential_energy(coords: np.ndarray, d_r: float = 116.09,
                                     alpha_r: float = 2.287, r_eq: float = 0.9419,
                                     k_theta: float = 87.85, theta_eq_deg: float = 107.4,
                                     epsilon: float = 0.1852, sigma: float = 3.1589,
                                     q_m: float = 1.1128, gamma: float = 0.73612,
                                     coulomb_constant: float = 332.06371) -> float:
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
    energy = 0.0

    # -- Intramolecular: two quartic-Morse stretches and one harmonic bend.
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        for i_h in (i_h1, i_h2):
            bond = float(np.linalg.norm(coords[i_h] - coords[i_o]))
            if bond <= 0.0:
                raise ValueError("an O-H bond has zero length")
            x = bond - r_eq
            energy += d_r * (alpha_r ** 2 * x ** 2 - alpha_r ** 3 * x ** 3
                             + (7.0 / 12.0) * alpha_r ** 4 * x ** 4)

        arm_1 = coords[i_h1] - coords[i_o]
        arm_2 = coords[i_h2] - coords[i_o]
        cosine = float(np.clip(np.dot(arm_1, arm_2)
                               / (np.linalg.norm(arm_1) * np.linalg.norm(arm_2)), -1.0, 1.0))
        energy += 0.5 * k_theta * (np.arccos(cosine) - theta_eq) ** 2

    # -- Intermolecular dispersion/repulsion: oxygen pairs only.
    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            distance = float(np.linalg.norm(coords[3 * mono_j] - coords[3 * mono_i]))
            if distance <= 0.0:
                raise ValueError("two oxygen atoms coincide")
            ratio6 = (sigma / distance) ** 6
            energy += 4.0 * epsilon * (ratio6 * ratio6 - ratio6)

    # -- Intermolecular electrostatics on the H, H and M sites of each monomer.
    site_charges = np.array([0.5 * q_m, 0.5 * q_m, -q_m], dtype=float)
    sites = np.zeros((n_monomers, 3, 3), dtype=float)
    for mono in range(n_monomers):
        i_o, i_h1, i_h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        sites[mono, 0] = coords[i_h1]
        sites[mono, 1] = coords[i_h2]
        sites[mono, 2] = gamma * coords[i_o] + (1.0 - gamma) * 0.5 * (coords[i_h1] + coords[i_h2])

    for mono_i in range(n_monomers):
        for mono_j in range(mono_i + 1, n_monomers):
            for site_a in range(3):
                for site_b in range(3):
                    distance = float(np.linalg.norm(sites[mono_j, site_b] - sites[mono_i, site_a]))
                    if distance <= 0.0:
                        raise ValueError("two charge sites coincide")
                    energy += coulomb * site_charges[site_a] * site_charges[site_b] / distance

    return float(energy)

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
            "call": "compute_potential_energy(coords)",
            "gold_call": "_oracle_compute_potential_energy(coords)",
        },
        # --- Valid: a single monomer, where every intermolecular term is absent ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.75, 0.0, 0.60], [-0.78, 0.0, 0.55]])
""",
            "call": "compute_potential_energy(coords)",
            "gold_call": "_oracle_compute_potential_energy(coords)",
        },
        # --- Boundary: a widely separated dimer, where the electrostatics are
        #     nearly cancelled and the dispersion term is negligible ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [40.0, 0.0, 0.0], [40.9419, 0.0, 0.0], [39.7164, 0.8983, 0.0]])
""",
            "call": "compute_potential_energy(coords)",
            "gold_call": "_oracle_compute_potential_energy(coords)",
        },
        # --- Edge: strongly compressed dimer with altered model parameters,
        #     which drives the quartic and repulsive terms hard ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [-0.35, 1.05, 0.0],
    [2.45, 0.10, 0.05], [3.30, 0.35, 0.40], [2.30, -0.60, 0.70]])
""",
            "call": "compute_potential_energy(coords, 120.0, 2.4, 0.96, 90.0, 104.0, 0.2, 3.2, 1.05, 0.5)",
            "gold_call": "_oracle_compute_potential_energy(coords, 120.0, 2.4, 0.96, 90.0, 104.0, 0.2, 3.2, 1.05, 0.5)",
        },
        # --- Invalid: atom count is not a whole number of monomers ---
        {
            "setup": """import numpy as np
coords = np.zeros((4, 3))
def run_model():
    try:
        compute_potential_energy(coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_energy(coords)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: charge-site fraction outside the unit interval ---
        {
            "setup": """import numpy as np
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
def run_model():
    try:
        compute_potential_energy(coords, 116.09, 2.287, 0.9419, 87.85, 107.4, 0.1852, 3.1589, 1.1128, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_potential_energy(coords, 116.09, 2.287, 0.9419, 87.85, 107.4, 0.1852, 3.1589, 1.1128, 1.0)
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
