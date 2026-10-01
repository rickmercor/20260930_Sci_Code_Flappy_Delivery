"""
Derive every scalar coefficient of the coupled electron-phonon system that the rest of the stability analysis consumes.

The linearised gray model represents electrons and phonons by intensity perturbation functions whose zeroth angular moments are the two energy densities. Electrons relax toward a local electron pseudo-temperature; phonons relax toward that same pseudo-temperature through electron-phonon scattering and toward their own pseudo-temperature through phonon-phonon scattering, so the phonon side carries two independent channels and has no single relaxation time until they are reduced to one transport scale. Neither pseudo-temperature is free: both are fixed by demanding that the scattering operators conserve energy, which makes each of them a definite linear combination of the two energy densities. The macroscopic system used later is written in energy densities rather than temperatures, so the first-order Chapman-Enskog heat fluxes - the diffusion limit of the coupled kinetic equations, each built on its own species' transport scale, its squared group speed, its heat capacity and the isotropic second angular moment of the direction vector - must be re-expressed through those same linear combinations. That substitution mixes the two species in every flux coefficient, and the phonon flux inherits a different mixture from each of its two relaxation channels. The linearised equilibria are isotropic, carrying the species heat capacity times the pseudo-temperature spread uniformly over the solid angle, and applying the same substitution to them gives the coefficients with which the two energy-density errors drive each species' scattering source. Getting any one of these coefficients wrong changes the final number without producing anything visibly wrong along the way.

Returns
-------
np.ndarray of shape (11,), float: effective phonon Knudsen number, two exchange rates, four flux coefficients, four scattering-source coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_system_coefficients(kn_ep: float, kn_pe: float, kn_pp: float,
                                C_e: float = 1.0, C_p: float = 1.0,
                                v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    """Derive the scalar coefficients of the coupled electron-phonon system.

    Parameters
    ----------
    kn_ep, kn_pe, kn_pp : float
        The three scattering Knudsen numbers, finite and strictly positive.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    coefficients : np.ndarray
        Shape (11,), in this order: the effective phonon transport Knudsen
        number; the two energy-exchange rates of the macroscopic energy
        balances, electron row then phonon row; the four flux coefficients of
        those balances written in the energy densities, ordered ee, ep, pe, pp;
        and the four scattering-source coefficients in the same order.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(11, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_system_coefficients(kn_ep: float, kn_pe: float, kn_pp: float,
                                        C_e: float = 1.0, C_p: float = 1.0,
                                        v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    vals = [float(v) for v in (kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in vals):
        raise ValueError("Knudsen numbers, capacities and speeds must be finite and > 0")
    kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p = vals

    kn_p = 1.0 / (1.0 / kn_pe + 1.0 / kn_pp)
    den = kn_pe * C_e + kn_ep * C_p
    w_e, w_p = kn_pe / den, kn_ep / den
    kappa_e = v_e * v_e * C_e * kn_ep / 3.0
    kappa_p = v_p * v_p * C_p * kn_p / 3.0
    four_pi = 4.0 * np.pi

    return np.array([
        kn_p, C_p / den, C_e / den,
        kappa_e * w_e, kappa_e * w_p,
        kappa_p * w_e * kn_p / kn_pe,
        kappa_p * w_p * kn_p / kn_pe + (kappa_p / C_p) * kn_p / kn_pp,
        w_e * C_e / four_pi, w_p * C_e / four_pi,
        w_e * C_p * kn_p / (four_pi * kn_pe),
        w_p * C_p * kn_p / (four_pi * kn_pe) + kn_p / (four_pi * kn_pp),
    ], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the benchmark's interior worst-case point and reported coefficients.
        {"setup": "import numpy as np\nkn = 10.0 ** 0.5\n",
         "call": "compute_system_coefficients(kn, kn, kn)",
         "gold_call": "_oracle_compute_system_coefficients(kn, kn, kn)"},
        # Normal: a transition-regime point with unequal heat capacities.
        {"setup": "import numpy as np\nkn = 2.5\n",
         "call": "compute_system_coefficients(kn, kn, kn, C_e=0.8, C_p=1.25)",
         "gold_call": "_oracle_compute_system_coefficients(kn, kn, kn, C_e=0.8, C_p=1.25)"},
        # Edge: unequal capacities and speeds, channels three decades apart.
        {"setup": "import numpy as np\n",
         "call": "compute_system_coefficients(0.01, 1.0, 100.0, 0.4, 2.5, 1.3, 0.7)",
         "gold_call": "_oracle_compute_system_coefficients(0.01, 1.0, 100.0, 0.4, 2.5, 1.3, 0.7)"},
        # Edge: phonon-phonon scattering four decades faster than phonon-electron.
        {"setup": "import numpy as np\n",
         "call": "compute_system_coefficients(1.0, 1.0, 1e-4)",
         "gold_call": "_oracle_compute_system_coefficients(1.0, 1.0, 1e-4)"},
        # Boundary: deep diffusive corner, all three scales tiny.
        {"setup": "import numpy as np\n",
         "call": "compute_system_coefficients(1e-6, 1e-6, 1e-6)",
         "gold_call": "_oracle_compute_system_coefficients(1e-6, 1e-6, 1e-6)"},
    ]
