"""
The graded quantity is the standard deviation of the total number of settled individuals of species 2. The capacity of species 1, the establishment multiplier, the sites held by each species, the standard deviation for species 1 and the mean shift of species 2 are returned alongside it, because each is a step the graded number depends on.

This stage answers the question the chain exists for: how two competing species share a heterogeneous river landscape, and how large the demographic fluctuations of their settled populations are when the landscape holds a finite number of sites. It starts from the drainage map, the link weights, the site density and the individual-based rates of both species, and reruns every earlier stage in order.

1. Stage 1 builds the directed dispersal network and the site numbers from the drainage counts, and requires the network to be strongly connected.

2. Stage 2 assembles the size-weighted explorer operator and its eigenbasis resolvent; stage 3 calls it for each species, and the orchestrator calls it directly to report how far each species' operator spectrum is from real.

3. Stage 3 returns the effective colonisation kernel of each species.

4. Stage 4 computes the generalised metapopulation capacity of species 1 alone.

5. Stage 5 solves for the steady state of a species alone; the orchestrator calls it for species 2 alone, and stage 6 calls it for the resident.

6. Stage 6 computes the common fecundity multiplier at which species 2, introduced into the landscape held by species 1, changes from declining to increasing.

7. Stage 7 finds the unique stable steady state at which the two species share the landscape.

8. Stage 8 linearises the finite-site chain of settled individuals and explorers of both species at that state, returning the drift Jacobian and the jump covariance.

9. Stage 9 solves the Lyapunov equation for the stationary covariance and the standard deviations and correlation of the two settled totals.

10. Stage 10 computes the leading-order shift of the mean numbers away from the deterministic state.

Returns
-------
dict, the two-species fluctuation analysis, keyed by species2_total_sd (the graded value), species1_capacity, establishment_multiplier, held_sites (length 2), species1_total_sd, species2_mean_shift, species1_mean_shift, total_correlation, relaxation_rate, explorer_totals (length 2), invader_capacity, invasion_capacity, resident_sites, invader_alone_sites, operator_max_imag (length 2), and sites.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def two_species_fluctuation_report(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
    n_starts: int,
) -> dict:
    """Analyse how two competitors share a river landscape and the finite-site fluctuations of their settled totals.

    Parameters
    ----------
    downstream : sequence of int
        Downstream neighbour of each patch, -1 for the outlet.
    downstream_weight : float
        Weight of each downstream link.
    upstream_weight : float
        Weight of each upstream link.
    extra_links : sequence of (int, int, float)
        One-way overland links.
    sites_per_patch_drained : float
        Sites per patch of drainage count.
    fecundity : np.ndarray
        Fecundities, one row per species.
    extinction : np.ndarray
        Extinction rates, one row per species.
    max_explorability : np.ndarray
        Maximal explorability of each species.
    exploration_rate : np.ndarray
        Explorer movement rate of each species.
    colonisation_rate : np.ndarray
        Colonisation attempt rate of each species.
    explorer_death_rate : np.ndarray
        Explorer death rate of each species.
    n_starts : int
        Number of interior initial conditions for the coexistence state.

    Returns
    -------
    dict
        Under the keys species2_total_sd, species1_capacity, establishment_multiplier, held_sites, species1_total_sd, species2_mean_shift, species1_mean_shift, total_correlation, relaxation_rate, explorer_totals, invader_capacity, invasion_capacity, resident_sites, invader_alone_sites, operator_max_imag and sites.

    Raises
    ------
    ValueError
        When any stage rejects its inputs, when the network is not strongly connected, when the rates are not given for exactly two species, or when the two species do not share the landscape at a unique stable state.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_two_species_fluctuation_report(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
    fecundity: np.ndarray,
    extinction: np.ndarray,
    max_explorability: np.ndarray,
    exploration_rate: np.ndarray,
    colonisation_rate: np.ndarray,
    explorer_death_rate: np.ndarray,
    n_starts: int,
) -> dict:
    """Reference implementation."""
    landscape = _oracle_build_dispersal_landscape(  # noqa: F821
        downstream, downstream_weight, upstream_weight, extra_links, sites_per_patch_drained)
    if landscape["strongly_connected"] != 1:
        raise ValueError("the dispersal network is not strongly connected")
    weights = landscape["weights"]
    sites = landscape["sites"]
    n = sites.size
    c = np.asarray(fecundity, dtype=float)
    e = np.asarray(extinction, dtype=float)
    rates = [np.asarray(r, dtype=float) for r in (max_explorability, exploration_rate, colonisation_rate, explorer_death_rate)]
    if c.shape != (2, n) or e.shape != (2, n) or any(r.shape != (2,) for r in rates):
        raise ValueError("fecundity and extinction must be 2 by N and the explorer rates must be given for two species")
    xi, d, lam, gamma = rates

    kernels = np.zeros((2, n, n))
    max_imag = np.zeros(2)
    for a in range(2):
        out = _oracle_effective_colonisation_kernel(  # noqa: F821
            weights, sites, xi[a], d[a], lam[a], gamma[a])
        kernels[a] = out["kernel"]
        spectrum = _oracle_explorer_resolvent(  # noqa: F821
            weights, sites, out["exploration_efficiency"], out["mortality_ratio"])
        max_imag[a] = float(np.max(np.abs(spectrum["eigenvalues_imag"])))

    capacity = _oracle_generalised_capacity(kernels[0], c[0], e[0])  # noqa: F821
    invader_alone = _oracle_equilibrium_occupancy(kernels[1], c[1], e[1], sites)  # noqa: F821
    establish = _oracle_establishment_multiplier(  # noqa: F821
        kernels[0], kernels[1], c[0], e[0], c[1], e[1], sites)
    coexist = _oracle_coexistence_state(kernels, c, e, sites, n_starts)  # noqa: F821
    settled = coexist["occupancy"] * sites[None, :]
    chain = _oracle_chain_linearisation(  # noqa: F821
        weights, sites, settled, c, e, xi, d, lam, gamma)
    fluct = _oracle_stationary_covariance(chain["jacobian"], chain["jump_covariance"], 2, n)  # noqa: F821
    shift = _oracle_mean_shift(chain["jacobian"], fluct["covariance"], lam, sites)  # noqa: F821

    return {
        "species2_total_sd": float(fluct["total_sd"][1]),
        "species1_capacity": capacity["capacity"],
        "establishment_multiplier": establish["multiplier"],
        "held_sites": coexist["held_sites"],
        "species1_total_sd": float(fluct["total_sd"][0]),
        "species2_mean_shift": float(shift["total_shift"][1]),
        "species1_mean_shift": float(shift["total_shift"][0]),
        "total_correlation": float(fluct["total_correlation"][0, 1]),
        "relaxation_rate": fluct["relaxation_rate"],
        "explorer_totals": chain["explorers"].sum(axis=1),
        "invader_capacity": establish["invader_capacity"],
        "invasion_capacity": establish["invasion_capacity"],
        "resident_sites": establish["resident_sites"],
        "invader_alone_sites": invader_alone["occupied_sites"],
        "operator_max_imag": max_imag,
        "sites": sites,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
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
    import numpy as np
    DOWN = (-1, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 10)
    LINKS = ((7, 12, 0.6), (11, 4, 0.5))
    C = np.array([[0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 1.3, 0.7, 0.9, 1.2, 0.8, 1.0],
                  [1.0, 0.9, 1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 1.0, 1.2, 0.8, 1.1, 0.9]])
    E = np.array([[0.30, 0.22, 0.26, 0.18, 0.35, 0.20, 0.24, 0.15, 0.40, 0.28, 0.17, 0.33, 0.21],
                  [0.25, 0.30, 0.20, 0.28, 0.22, 0.30, 0.18, 0.32, 0.26, 0.20, 0.24, 0.19, 0.30]])
    XI = np.array([0.8, 0.8]); D = np.array([1.6, 3.2]); LAM = np.array([1.0, 1.0]); GAM = np.array([0.25, 0.5])
    def digest(out):
        keys = ("species2_total_sd", "species1_capacity", "establishment_multiplier", "species1_total_sd", "species2_mean_shift",
                "species1_mean_shift", "total_correlation", "relaxation_rate", "invader_capacity", "invasion_capacity",
                "resident_sites", "invader_alone_sites")
        return (tuple(round(out[k], 5) for k in keys), np.round(out["held_sites"], 4), np.round(out["explorer_totals"], 4),
                np.round(out["operator_max_imag"], 6), np.round(out["sites"], 6))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # the graded two-species river landscape
            "setup": SETUP + FLAT,
            "call": "flat(digest(two_species_fluctuation_report(DOWN, 1.0, 0.3, LINKS, 40.0, C, E, XI, D, LAM, GAM, 3)))",
            "gold_call": "flat(digest(_oracle_two_species_fluctuation_report(DOWN, 1.0, 0.3, LINKS, 40.0, C, E, XI, D, LAM, GAM, 3)))",
        },
        {
            # a five-patch chain with a return link, two species with opposed extinction gradients, faster explorers
            "setup": """
import numpy as np
DOWN5 = (-1, 0, 1, 2, 3)
C5 = np.array([[1.2, 1.0, 0.9, 1.1, 1.0], [0.9, 1.1, 1.0, 1.2, 1.1]])
E5 = np.array([[0.10, 0.14, 0.18, 0.24, 0.30], [0.30, 0.24, 0.18, 0.14, 0.10]])
def chain5(fn):
    out = fn(DOWN5, 1.0, 0.4, ((0, 4, 0.3),), 30.0, C5, E5, np.array([0.9, 0.9]), np.array([4.0, 4.0]),
             np.array([2.0, 2.0]), np.array([0.2, 0.2]), 2)
    return (round(out["species2_total_sd"], 5), round(out["species1_total_sd"], 5), np.round(out["held_sites"], 4),
            round(out["species2_mean_shift"], 6), round(out["establishment_multiplier"], 6))
""" + FLAT,
            "call": "flat(chain5(two_species_fluctuation_report))",
            "gold_call": "flat(chain5(_oracle_two_species_fluctuation_report))",
        },
        {
            # the settled totals scale with the site density: doubling it doubles the held sites, multiplies the
            # standard deviations by sqrt(2), and leaves the mean shift and the capacity unchanged
            "setup": SETUP + """
def scaling(fn):
    a = fn(DOWN, 1.0, 0.3, LINKS, 40.0, C, E, XI, D, LAM, GAM, 2)
    b = fn(DOWN, 1.0, 0.3, LINKS, 80.0, C, E, XI, D, LAM, GAM, 2)
    return (np.round(b["held_sites"] / a["held_sites"], 8), round(b["species2_total_sd"] / a["species2_total_sd"], 8),
            round(b["species2_mean_shift"] - a["species2_mean_shift"], 8), round(b["species1_capacity"] - a["species1_capacity"], 10))
""" + FLAT,
            "call": "flat(scaling(two_species_fluctuation_report))",
            "gold_call": "flat(scaling(_oracle_two_species_fluctuation_report))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(downstream=DOWN, downstream_weight=1.0, upstream_weight=0.3, extra_links=LINKS, sites_per_patch_drained=40.0,
                fecundity=C, extinction=E, max_explorability=XI, exploration_rate=D, colonisation_rate=LAM,
                explorer_death_rate=GAM, n_starts=2)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
WEAK = E.copy(); WEAK[1] *= 30.0
""",
            "call": "(verdict(two_species_fluctuation_report, upstream_weight=0.0, extra_links=()), verdict(two_species_fluctuation_report, fecundity=C[:1]), verdict(two_species_fluctuation_report, exploration_rate=np.array([1.6])), verdict(two_species_fluctuation_report, extinction=WEAK), verdict(two_species_fluctuation_report, n_starts=1), verdict(two_species_fluctuation_report))",
            "gold_call": "(verdict(_oracle_two_species_fluctuation_report, upstream_weight=0.0, extra_links=()), verdict(_oracle_two_species_fluctuation_report, fecundity=C[:1]), verdict(_oracle_two_species_fluctuation_report, exploration_rate=np.array([1.6])), verdict(_oracle_two_species_fluctuation_report, extinction=WEAK), verdict(_oracle_two_species_fluctuation_report, n_starts=1), verdict(_oracle_two_species_fluctuation_report))",
        },
    ]
