"""
Run the explicit interfacial transient of the injection-layer/transport-layer stack from equilibrium and return the hole density on the transport side of the junction.

This is the orchestrator. It builds the non-uniform grid and the layer profiles, sets the equilibrium starting densities, and advances the coupled hole and electron populations in time, returning the hole density at node n_hil + 1, the first transport-layer node, once n_steps explicit steps have been taken.

Every step is a full sweep in a fixed order. The electrostatic potential is re-solved from the current carrier distribution, with 0 V at node 0 and the applied potential V at the last node, because the space charge changes as carriers redistribute and a frozen potential would decouple transport from its own electrostatics. The band-edge driving fields follow from that potential and the flat-band levels; the midpoint densities follow from the chosen scheme; the drift-diffusion currents follow from those; the recombination rate is evaluated on the nodes from the current densities; and the continuity update then advances every interior node while the two contact nodes stay at their equilibrium densities. All quantities within one step are computed from the densities at the start of that step.

Both midpoint schemes are supported so the two can be compared under identical conditions. With the conventional arithmetic mean at the junction the sparse interfacial density can be driven negative and the integration can then diverge; the orchestrator does not stop or repair such a run, and a divergent run simply returns whatever non-finite or unphysical value results.

Densities are handled in cm^-3 internally; the returned value is expressed in units of 1e18 cm^-3.

Returns
-------
float, the hole density at node n_hil + 1 after n_steps explicit steps, in units of 1e18 cm^-3, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def interfacial_hole_density(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float, dt: float,
                             n_steps: int, T: float, V: float,
                             EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                             NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                             er_hil: float, er_htl: float, mup_hil: float, mup_htl: float,
                             mun_hil: float, mun_htl: float, tau_p: float, tau_n: float,
                             Cp: float, Cn: float, scheme: str) -> float:
    '''Hole density at the first transport-layer node after an explicit transient.

    Parameters
    ----------
    n_hil, n_htl : int
        Numbers of injection-layer and transport-layer intervals (the junction interval counts in the
        transport layer); each at least 1.
    dz_hil, dz_htl : float
        Injection-layer and transport-layer grid spacings in cm, positive.
    dt : float
        Time step in s, positive.
    n_steps : int
        Number of explicit steps, at least 1.
    T : float
        Temperature in K, positive.
    V : float
        Potential in V applied at the last node; node 0 is held at 0 V.
    EV_hil, EV_htl, EC_hil, EC_htl : float
        Flat-band valence and conduction levels of the two layers in eV.
    NA_hil, NA_htl, ni_hil, ni_htl : float
        Acceptor doping and intrinsic densities of the two layers in cm^-3, positive.
    er_hil, er_htl : float
        Relative permittivities of the two layers, positive.
    mup_hil, mup_htl, mun_hil, mun_htl : float
        Hole and electron mobilities of the two layers in cm^2 V^-1 s^-1.
    tau_p, tau_n : float
        SRH lifetimes in s, positive.
    Cp, Cn : float
        Auger capture probabilities in cm^6 s^-1, non-negative.
    scheme : str
        "field" for the field-selected junction density, "mean" for the arithmetic mean.

    Returns
    -------
    p_int : float
        Hole density at node n_hil + 1 after n_steps steps, in units of 1e18 cm^-3.

    Raises
    ------
    ValueError
        If n_steps is not an integer of at least 1, if dt is not positive, or if any argument violates
        the requirements of the grid, equilibrium, potential, field, midpoint, current, recombination
        or update calculations it is passed to.
    '''
    return p_int

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interfacial_hole_density(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float, dt: float,
                                     n_steps: int, T: float, V: float,
                                     EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                                     NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                                     er_hil: float, er_htl: float, mup_hil: float, mup_htl: float,
                                     mun_hil: float, mun_htl: float, tau_p: float, tau_n: float,
                                     Cp: float, Cn: float, scheme: str) -> float:
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be an integer of at least 1")
    if not dt > 0.0:
        raise ValueError("dt must be positive")
    if scheme not in ("mean", "field"):
        raise ValueError("scheme must be 'mean' or 'field'")
    prof = _oracle_grid_and_layer_profiles(n_hil, n_htl, dz_hil, dz_htl, EV_hil, EV_htl, EC_hil, EC_htl,
                                           NA_hil, NA_htl, ni_hil, ni_htl, er_hil, er_htl)
    z, EV0, EC0, NA, ni, er = prof
    N = z.size
    in_hil = np.arange(N) <= n_hil
    mu_p = np.where(in_hil, float(mup_hil), float(mup_htl))
    mu_n = np.where(in_hil, float(mun_hil), float(mun_htl))
    p, n = _oracle_equilibrium_densities(NA, ni)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(int(n_steps)):
            phi = _oracle_poisson_potential(p, n, NA, er, z, 0.0, V)
            FV, FC = _oracle_band_edge_fields(EV0, EC0, phi, z)
            p_mid, n_mid = _oracle_midpoint_densities(p, n, FV, FC, n_hil, scheme)
            Jp, Jn = _oracle_drift_diffusion_currents(p, n, p_mid, n_mid, FV, FC, mu_p, mu_n, z, T)
            U = _oracle_recombination_rate(p, n, ni, tau_p, tau_n, Cp, Cn)
            p, n = _oracle_continuity_update(p, n, Jp, Jn, U, z, dt)
    return float(p[n_hil + 1] / 1.0e18)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
cfg = (20, 80, 1.0e-7, 0.25e-7, 1.0e-12, 1000, 300.0, 1.0, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17,
       1.63e6, 1.59e-6, 3.0, 9.4, 2.0e-4, 2.0e-3, 2.0e-6, 2.0e-5, 1.0e-7, 1.0e-7, 0.0, 0.0)""",
            "call": "interfacial_hole_density(*cfg, 'field')",
            "gold_call": "_oracle_interfacial_hole_density(*cfg, 'field')",
        },
        {
            "setup": """import numpy as np
cfg = (20, 20, 1.0e-7, 1.0e-7, 1.0e-12, 1000, 300.0, 1.0, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17,
       1.63e6, 1.59e-6, 3.0, 3.2, 2.0e-4, 2.0e-3, 2.0e-6, 2.0e-5, 1.0e-7, 1.0e-7, 0.0, 0.0)""",
            "call": "interfacial_hole_density(*cfg, 'field')",
            "gold_call": "_oracle_interfacial_hole_density(*cfg, 'field')",
        },
        {
            "setup": """import numpy as np
cfg = (10, 20, 1.0e-7, 0.5e-7, 1.0e-12, 40, 300.0, 0.5, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17,
       1.63e6, 1.59e-6, 3.0, 9.4, 2.0e-4, 2.0e-3, 2.0e-6, 2.0e-5, 1.0e-7, 1.0e-7, 0.0, 0.0)""",
            "call": "interfacial_hole_density(*cfg, 'mean')",
            "gold_call": "_oracle_interfacial_hole_density(*cfg, 'mean')",
        },
        {
            "setup": """import numpy as np
cfg = (8, 16, 1.0e-7, 0.5e-7, 5.0e-13, 300, 350.0, -0.5, -5.10, -5.45, -3.50, -2.70, 1.0e19, 5.0e16,
       1.0e6, 1.0e-5, 3.0, 3.5, 5.0e-4, 1.0e-3, 1.0e-5, 1.0e-4, 2.0e-8, 5.0e-8, 1.0e-30, 2.0e-30)""",
            "call": "interfacial_hole_density(*cfg, 'field')",
            "gold_call": "_oracle_interfacial_hole_density(*cfg, 'field')",
        },
        {
            "setup": """import numpy as np
cfg = (10, 20, 1.0e-7, 0.5e-7, 1.0e-12, 200, 300.0, 1.0, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17,
       1.63e6, 1.0e16, 3.0, 9.4, 2.0e-4, 2.0e-3, 2.0e-6, 2.0e-5, 1.0e-10, 1.0e-10, 1.0e-30, 1.0e-30)""",
            "call": "interfacial_hole_density(*cfg, 'field')",
            "gold_call": "_oracle_interfacial_hole_density(*cfg, 'field')",
        },
        {
            "setup": """import numpy as np
cfg = (5, 5, 1.0e-7, 1.0e-7, 1.0e-12, 0, 300.0, 1.0, -5.17, -5.60, -3.60, -2.60, 2.81e19, 1.00e17,
       1.63e6, 1.59e-6, 3.0, 9.4, 2.0e-4, 2.0e-3, 2.0e-6, 2.0e-5, 1.0e-7, 1.0e-7, 0.0, 0.0)
def _probe(fn):
    hits = 0
    for scheme, c in (('field', cfg), ('upwind', cfg[:5] + (10,) + cfg[6:]), ('field', cfg[:4] + (-1.0e-12, 10) + cfg[6:])):
        try:
            fn(*c, scheme)
        except ValueError:
            hits += 1
    return float(hits)""",
            "call": "_probe(interfacial_hole_density)",
            "gold_call": "_probe(_oracle_interfacial_hole_density)",
        },
    ]
