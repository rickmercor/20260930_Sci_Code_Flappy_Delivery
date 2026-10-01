"""
This stage answers the question the whole chain exists for: the valley splitting of the ground state of a strained silicon quantum well with an engineered germanium profile, computed in a multi-valley envelope-function description without a slowly varying potential approximation, in which each valley's envelope is band limited to its own sector of the Brillouin zone. It starts from the device and material description and reruns every earlier stage in order.

1. Stage 1 strains the silicon layer to the relaxed buffer and returns eps_zz, the strained growth-direction reciprocal-lattice vector G0z, the open sector (0, G0z / 2), the valley wave number k0 and its distance k1 to the strained zone boundary.

2. Stage 2 lays out the periodic supercell matched to G0z and evaluates the wiggle-well germanium profile and the confinement energy U(z) = Delta E_c X(z) - e F z on it.

3. Stage 3 assembles the back-folded, sector-projected potential operator on the open sector of the +k0 valley.

4. Stage 4 adds the kinetic energy measured from the valley minimum, solves the band-limited single-valley eigenproblem on that operator, and returns the ground-state envelope, its energy and the first excited energy.

5. Stage 5 evaluates the first-order intervalley coupling Delta of that envelope and the splitting 2 |Delta|, with the contribution of each harmonic order.

6. Stage 6 solves the conventional local effective-mass equation for the same stack on the same cell, and stage 5 evaluates its coupling, as the baseline the band-limited construction replaces.

7. Stage 7 measures, for the band-limited and for the local envelope, the response R of the coupling to a constant added to the confinement, and projects the local envelope onto its sector, after which stage 5 evaluates the coupling of the filtered envelope.

The graded quantity is the band-limited splitting in meV. The local and filtered results, the two responses, the dominant harmonic and its share, and the geometry of the strained zone are returned alongside it, because they are what distinguishes the construction from its approximations.

The ground state of a field-biased stack must be a state of the well. Stage 4 returns the lowest eigenvalue of the cell, which is that state only when the spacer is short enough that the field cannot pull the energy below the well ground state anywhere under the dielectric; this stage therefore rejects a result whose mean position lies more than 2 nm outside the silicon layer.

Returns
-------
dict holding the floats splitting_mev, the graded band-limited valley splitting in meV; ground_energy_mev and excited_energy_mev, the two lowest band-limited levels in meV in the zero of U; mean_position and spread in nm; eps_zz; sector_edge_fraction and long_period_fraction, G0z / 2 and 2 k1 in units of 2 pi / a_Si; dominant_share, the modulus of the dominant harmonic's contribution divided by |Delta|; local_splitting_mev and local_ground_energy_mev; local_leakage; filtered_splitting_mev; local_ambiguity_abs and exact_ambiguity_abs, the moduli of R for the local and band-limited envelopes; and the integer dominant_order.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_valley_splitting(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
    longitudinal_mass: float,
    band_offset: float,
    x_barrier: float,
    x_wiggle: float,
    well_monolayers: float,
    sigma_upper: float,
    sigma_lower: float,
    wiggle_period: float,
    field: float,
    spacer: float,
    orders,
    coefficients,
    backfold,
    n_bz: int,
    n_fbz: int,
) -> dict:
    """Compute the band-limited valley splitting of a strained Si/SiGe wiggle well from the device description.

    Parameters
    ----------
    lattice_si : float
        Lattice constant of silicon in nm.
    lattice_ge : float
        Lattice constant of germanium in nm.
    bowing : float
        Bowing length of the alloy lattice constant in nm.
    substrate_ge_fraction : float
        Germanium fraction of the relaxed buffer.
    c11 : float
        Elastic constant C11 of silicon in GPa.
    c12 : float
        Elastic constant C12 of silicon in GPa.
    valley_fraction : float
        Valley minimum as a fraction of 2 pi / a_Si.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses.
    band_offset : float
        Conduction-band offset per unit germanium fraction in eV.
    x_barrier : float
        Barrier germanium fraction.
    x_wiggle : float
        Amplitude of the in-well germanium oscillation.
    well_monolayers : float
        Well thickness in monolayers.
    sigma_upper : float
        Upper interface width in nm.
    sigma_lower : float
        Lower interface width in nm.
    wiggle_period : float
        Oscillation period in nm.
    field : float
        Vertical field in volts per nm.
    spacer : float
        Distance from the upper interface to the dielectric in nm.
    orders : sequence of int
        Orders n, containing zero and closed under negation.
    coefficients : sequence of complex
        Intervalley overlap sums C_n.
    backfold : sequence of complex
        Intravalley back-folding coefficients B_n.
    n_bz : int
        Resolved Brillouin zones.
    n_fbz : int
        Grid wave numbers per zone.

    Returns
    -------
    dict
        Under the keys splitting_mev, ground_energy_mev, excited_energy_mev, mean_position, spread, eps_zz, sector_edge_fraction, long_period_fraction, dominant_order, dominant_share, local_splitting_mev, local_ground_energy_mev, local_leakage, filtered_splitting_mev, local_ambiguity_abs and exact_ambiguity_abs.

    Raises
    ------
    ValueError
        When any stage rejects its inputs, or when the band-limited ground state lies more than 2 nm outside the silicon layer.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_exact_valley_splitting(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
    longitudinal_mass: float,
    band_offset: float,
    x_barrier: float,
    x_wiggle: float,
    well_monolayers: float,
    sigma_upper: float,
    sigma_lower: float,
    wiggle_period: float,
    field: float,
    spacer: float,
    orders,
    coefficients,
    backfold,
    n_bz: int,
    n_fbz: int,
) -> dict:
    """Reference implementation: reruns every stage from the device description."""
    geometry = _oracle_strained_valley_geometry(lattice_si, lattice_ge, bowing, substrate_ge_fraction, c11, c12, valley_fraction)  # noqa: F821
    g0 = geometry["zone_vector"]
    k0 = geometry["valley_wavenumber"]
    cell = _oracle_wiggle_well_supercell(g0, n_bz, n_fbz, spacer, well_monolayers, lattice_si, sigma_upper, sigma_lower, x_barrier, x_wiggle, wiggle_period, band_offset, field)  # noqa: F821
    z = cell["z"]
    u = cell["potential"]

    projected = _oracle_sector_projected_potential(z, u, g0, n_fbz, orders, backfold)  # noqa: F821
    exact = _oracle_nonlocal_valley_ground_state(z, projected["wavenumbers"], projected["matrix"], k0, longitudinal_mass)  # noqa: F821
    depth = float(well_monolayers) * float(lattice_si) / 4.0
    if not -depth - 2.0 < exact["mean_position"] < 2.0:
        raise ValueError("the band-limited ground state is not a state of the well")
    coupling = _oracle_intervalley_coupling(z, exact["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821

    local = _oracle_local_valley_ground_state(z, u, longitudinal_mass, k0, g0)  # noqa: F821
    local_coupling = _oracle_intervalley_coupling(z, local["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821
    filtered = _oracle_spectral_filter_envelope(z, local["envelope"], k0, g0, orders, coefficients)  # noqa: F821
    filtered_coupling = _oracle_intervalley_coupling(z, filtered["envelope"], u, k0, g0, orders, coefficients)  # noqa: F821
    exact_response = _oracle_spectral_filter_envelope(z, exact["envelope"], k0, g0, orders, coefficients)  # noqa: F821

    parts = coupling["contribution_real"] + 1j * coupling["contribution_imag"]
    unit = 2.0 * math.pi / float(lattice_si)
    return {
        "splitting_mev": 1.0e3 * coupling["splitting"],
        "ground_energy_mev": 1.0e3 * exact["energy"],
        "excited_energy_mev": 1.0e3 * exact["excited_energy"],
        "mean_position": exact["mean_position"],
        "spread": exact["spread"],
        "eps_zz": geometry["eps_zz"],
        "sector_edge_fraction": geometry["sector_edge"] / unit,
        "long_period_fraction": 2.0 * geometry["edge_distance"] / unit,
        "dominant_order": coupling["dominant_order"],
        "dominant_share": float(np.max(np.abs(parts))) / coupling["delta_abs"],
        "local_splitting_mev": 1.0e3 * local_coupling["splitting"],
        "local_ground_energy_mev": 1.0e3 * local["energy"],
        "local_leakage": local["leakage"],
        "filtered_splitting_mev": 1.0e3 * filtered_coupling["splitting"],
        "local_ambiguity_abs": filtered["ambiguity_abs"],
        "exact_ambiguity_abs": exact_response["ambiguity_abs"],
    }

# =============================================================================
# TEST CASES
# =============================================================================

FLAT = """
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

SETUP = """
ORDERS = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
C = (1.84e-3, 5.99e-4, -2.44e-2, -3.18e-2, -0.221, -4.02e-4, 1.58e-3, 1.04e-5, 3.35e-5)
B = (-2.47e-4, -8.55e-5, -5.79e-4, 2.92e-3, 1.0, 2.92e-3, -5.79e-4, -8.55e-5, -2.47e-4)
MATERIAL = dict(lattice_si=0.543, lattice_ge=0.565, bowing=0.0200326, substrate_ge_fraction=0.3, c11=167.5,
                c12=65.0, valley_fraction=0.8394, longitudinal_mass=0.909, band_offset=0.5, x_barrier=0.3)
def digest(out):
    return (round(out["splitting_mev"], 6), round(out["ground_energy_mev"], 5), round(out["excited_energy_mev"], 5),
            round(out["mean_position"], 5), round(out["spread"], 5), round(out["eps_zz"], 10),
            round(out["sector_edge_fraction"], 10), round(out["long_period_fraction"], 10), out["dominant_order"],
            round(out["dominant_share"], 6), round(out["local_splitting_mev"], 6), round(out["local_ground_energy_mev"], 5),
            round(out["local_leakage"], 9), round(out["filtered_splitting_mev"], 6), round(out["local_ambiguity_abs"], 8),
            int(out["exact_ambiguity_abs"] < 1e-12))
"""


def test_cases():
    return [
        {
            # a thin wiggle well near the long-period resonance, on a reduced cell
            "setup": SETUP + """
ARGS = dict(MATERIAL, x_wiggle=0.15, well_monolayers=40, sigma_upper=0.5, sigma_lower=0.5, wiggle_period=1.629,
            field=3.0e-3, spacer=12.0, orders=ORDERS, coefficients=C, backfold=B, n_bz=10, n_fbz=128)
""" + FLAT,
            "call": "flat(digest(exact_valley_splitting(**ARGS)))",
            "gold_call": "flat(digest(_oracle_exact_valley_splitting(**ARGS)))",
        },
        {
            # the short-period resonance at 2 k0 with no field and no shear channel open
            "setup": SETUP + """
C0 = tuple(c if n % 2 == 0 else 0.0 for n, c in zip(ORDERS, C))
B0 = tuple(b if n % 2 == 0 else 0.0 for n, b in zip(ORDERS, B))
ARGS = dict(MATERIAL, x_wiggle=0.05, well_monolayers=40, sigma_upper=0.4, sigma_lower=0.6, wiggle_period=0.543 / 1.6788,
            field=0.0, spacer=10.0, orders=ORDERS, coefficients=C0, backfold=B0, n_bz=10, n_fbz=128)
""" + FLAT,
            "call": "flat(digest(exact_valley_splitting(**ARGS)))",
            "gold_call": "flat(digest(_oracle_exact_valley_splitting(**ARGS)))",
        },
        {
            # a spacer long enough for the field to bind a state under the dielectric must be rejected,
            # and so must inconsistent inputs caught by the earlier stages
            "setup": SETUP + """
ARGS = dict(MATERIAL, x_wiggle=0.0, well_monolayers=40, sigma_upper=0.5, sigma_lower=0.5, wiggle_period=1.0,
            field=6.0e-3, spacer=30.0, orders=ORDERS, coefficients=C, backfold=B, n_bz=10, n_fbz=256)
def verdict(fn, **kw):
    args = dict(ARGS)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(exact_valley_splitting), verdict(exact_valley_splitting, field=1.0e-3, n_fbz=64), verdict(exact_valley_splitting, field=1.0e-3, orders=(0, 1)), verdict(exact_valley_splitting, field=1.0e-3, valley_fraction=1.2), verdict(exact_valley_splitting, field=1.0e-3, spacer=10.0, n_fbz=128)))",
            "gold_call": "flat((verdict(_oracle_exact_valley_splitting), verdict(_oracle_exact_valley_splitting, field=1.0e-3, n_fbz=64), verdict(_oracle_exact_valley_splitting, field=1.0e-3, orders=(0, 1)), verdict(_oracle_exact_valley_splitting, field=1.0e-3, valley_fraction=1.2), verdict(_oracle_exact_valley_splitting, field=1.0e-3, spacer=10.0, n_fbz=128)))",
        },
    ]
