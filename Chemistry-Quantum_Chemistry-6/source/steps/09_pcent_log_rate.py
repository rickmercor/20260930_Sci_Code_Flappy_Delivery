"""
Compute the base-ten logarithm of the proton-coupled energy transfer rate constant for one reorganisation energy, starting from the proton potentials of every electronic state.

The reorganisation energy enters the rate twice. It fixes the product well displacement q_II = (2 * lambda / k) ** 0.5, and with it the crossing point of every reactant-product pair, the geometry at which that pair's coupling is evaluated. It also shifts the centre of every spectral overlap, whose Gaussian line shapes are separated by the vibronic gap E(II, nu) - E(I, mu) displaced by lambda. The Boltzmann populations and the gaps of the spectral overlaps belong to the equilibrium ladders, each state at the bottom of its own bath well; only the couplings are evaluated at the crossing points.

The proton potentials of the reactant and of the product do not depend on the bath, so their levels and functions are computed once. The unpopulated states are different: the bath tilts their proton potentials, so that at bath coordinate q the bias of state j is b_j(q) = b_j + t_j * (q - q_j), with t_j the tilt of that state and q_j its bath minimum. Their levels and proton vibrational functions therefore have to be rebuilt at every pair's crossing point, and it is those functions, not the equilibrium ones, that enter the overlaps and the energy denominators of the pathways.

The calculation runs as follows: the reactant and product ladders and functions from the proton grid; the crossing point of each of the pairs; at each crossing the tilted potentials of the unpopulated states with their levels and functions; the total coupling of that pair, the direct interaction plus the limit of the sum over pathways of every length through the unpopulated manifold, which is rejected if it does not converge; and finally the golden-rule sum over pairs of the reactant Boltzmann population, the squared coupling and the spectral overlap of Gaussian line shapes of standard deviation sigma normalised over angular frequency, divided by four pi squared hbar squared.

Returns
-------
float, the base-ten logarithm of the rate constant in inverse seconds at the given reorganisation energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pcent_log_rate(reorganization: float, r_min: float, r_max: float, n_points: int,
                   r_ref: float, potentials: dict, tilts: dict, virtual_states: tuple,
                   electronic_couplings: dict, direct: float, force_constant: float,
                   bath_minima: dict, temperature: float, width: float, n_states: int,
                   hbar2_over_2m: float, hbar: float, kb: float) -> float:
    '''Base-ten logarithm of the rate constant at one reorganisation energy.

    Parameters
    ----------
    reorganization : float
        Positive reorganisation energy lambda in eV.
    r_min, r_max : float
        Ends of the proton grid in angstrom, r_min < r_max.
    n_points : int
        Odd number of grid points, at least 3.
    r_ref : float
        Positive reduced-coordinate scale of every proton potential in angstrom.
    potentials : dict
        Maps "I", "II" and every label in virtual_states to a tuple (A, B, C)
        in eV of that state's potential A * ((r / r_ref)**2 - 1)**2
        + B * (r / r_ref) + C at the bottom of its bath well.
    tilts : dict
        Maps every label in virtual_states to the tilt t_j in eV by which the
        bath changes that state's bias, B -> B + t_j * (q - q_j).
    virtual_states : tuple of str
        Labels of the unpopulated electronic states.
    electronic_couplings : dict
        Maps a pair of state labels to their electronic coupling in eV; one of
        (a, b) and (b, a) is enough.
    direct : float
        Direct electronic coupling between the reactant and product in eV.
    force_constant : float
        Positive bath force constant k in eV.
    bath_minima : dict
        Maps every label in virtual_states to its bath minimum.
    temperature : float
        Positive temperature in kelvin.
    width : float
        Positive standard deviation sigma of each line shape in eV.
    n_states : int
        Number of vibrational levels retained on every state.
    hbar2_over_2m : float
        Positive hbar squared over twice the proton mass in eV angstrom squared.
    hbar : float
        Positive reduced Planck constant in eV seconds.
    kb : float
        Positive Boltzmann constant in eV per kelvin.

    Returns
    -------
    log_rate : float
        Base-ten logarithm of the rate constant in inverse seconds.

    Raises
    ------
    ValueError
        If a potential, tilt, bath minimum or coupling is missing or invalid,
        if the grid or any constant is out of range, if an energy denominator
        vanishes, if the sum over pathways does not converge for a pair, or if
        the resulting rate is not a positive finite number.
    '''
    return log_rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _state_potential(grid, potentials, label, r_ref, bias_shift=0.0):
    if label not in potentials:
        raise ValueError(f"the potential of state {label} is missing")
    params = tuple(potentials[label])
    if len(params) != 3:
        raise ValueError(f"the potential of state {label} needs (A, B, C)")
    a, b, c = (float(p) for p in params)
    return _oracle_proton_potential(grid, a, r_ref, b + float(bias_shift), c)


def _oracle_pcent_log_rate(reorganization: float, r_min: float, r_max: float, n_points: int,
                           r_ref: float, potentials: dict, tilts: dict, virtual_states: tuple,
                           electronic_couplings: dict, direct: float, force_constant: float,
                           bath_minima: dict, temperature: float, width: float,
                           n_states: int, hbar2_over_2m: float, hbar: float,
                           kb: float) -> float:
    """Reference implementation."""
    lam, k, _ = _bath_pair(reorganization, force_constant)
    virtual = _virtual_labels(virtual_states)
    if not isinstance(potentials, dict) or not isinstance(tilts, dict) \
            or not isinstance(bath_minima, dict):
        raise ValueError("potentials, tilts and bath_minima must be dictionaries")
    if not (np.isfinite(float(r_min)) and np.isfinite(float(r_max))
            and float(r_min) < float(r_max)):
        raise ValueError("the grid must satisfy r_min < r_max")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    grid = np.linspace(float(r_min), float(r_max), int(n_points))
    spacing = float(grid[1] - grid[0]) if grid.size > 1 else 0.0
    reactant = _state_potential(grid, potentials, "I", r_ref)
    product = _state_potential(grid, potentials, "II", r_ref)
    e_reactant = _oracle_vibrational_levels(grid, reactant, hbar2_over_2m, n_states)
    e_product = _oracle_vibrational_levels(grid, product, hbar2_over_2m, n_states)
    w_reactant = _oracle_vibrational_wavefunctions(grid, reactant, hbar2_over_2m, n_states)
    w_product = _oracle_vibrational_wavefunctions(grid, product, hbar2_over_2m, n_states)
    crossings = _oracle_crossing_points(lam, k, e_reactant, e_product)
    couplings = np.zeros(crossings.shape)
    for mu in range(crossings.shape[0]):
        for nu in range(crossings.shape[1]):
            q = float(crossings[mu, nu])
            levels, waves = {}, {}
            for label in virtual:
                if label not in tilts:
                    raise ValueError(f"the tilt of state {label} is missing")
                if label not in bath_minima:
                    raise ValueError(f"the bath minimum of state {label} is missing")
                shift = float(tilts[label]) * (q - float(bath_minima[label]))
                if not np.isfinite(shift):
                    raise ValueError(f"the tilted potential of state {label} is not finite")
                tilted = _state_potential(grid, potentials, label, r_ref, shift)
                levels[label] = _oracle_vibrational_levels(grid, tilted, hbar2_over_2m, n_states)
                waves[label] = _oracle_vibrational_wavefunctions(grid, tilted, hbar2_over_2m,
                                                                n_states)
            couplings[mu, nu] = _oracle_pair_coupling(
                q, lam, k, bath_minima, float(e_product[nu]), levels, waves,
                w_reactant[:, mu], w_product[:, nu], spacing, virtual, electronic_couplings,
                direct)
    return _oracle_rate_from_couplings(couplings, e_reactant, e_product, temperature,
                                       lam, width, hbar, kb)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: the five-state assembly on its own grid
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "CT2": (0.300, -0.185, 4.260),
       "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.15, "CT2": 0.10, "B": 0.05}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "CT2", "B"), vel, -7.909e-4, 0.8,
        qmin, 300.0, 0.185, 3, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
""",
            "call": "pcent_log_rate(0.280, *args)",
            "gold_call": "_oracle_pcent_log_rate(0.280, *args)",
        },
        {
            "setup": """import numpy as np
# normal: the same assembly at a larger reorganisation energy, which moves every
# crossing point, every tilted potential and every spectral overlap
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "CT2": (0.300, -0.185, 4.260),
       "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.15, "CT2": 0.10, "B": 0.05}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "CT2", "B"), vel, -7.909e-4, 0.8,
        qmin, 300.0, 0.185, 3, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
""",
            "call": "pcent_log_rate(0.450, *args)",
            "gold_call": "_oracle_pcent_log_rate(0.450, *args)",
        },
        {
            "setup": """import numpy as np
# boundary: untilted unpopulated states, where the equilibrium potentials serve
# at every crossing point
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615), "B": (0.320, -0.120, 3.880)}
tilt = {"CT1": 0.0, "B": 0.0}
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "B"): 0.0080,
       ("B", "II"): -0.0160, ("B", "CT1"): 0.1100}
qmin = {"CT1": 1.2, "B": 0.0}
args = (-0.90, 0.90, 121, 0.400, pot, tilt, ("CT1", "B"), vel, -7.909e-4, 0.8,
        qmin, 300.0, 0.185, 3, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
""",
            "call": "pcent_log_rate(0.280, *args)",
            "gold_call": "_oracle_pcent_log_rate(0.280, *args)",
        },
        {
            "setup": """import numpy as np
# edge: two levels on a coarser grid, a stiffer bath, a cold narrow line shape
# and a tilt of the opposite sign
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
tilt = {"CT1": -0.12}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
args = (-0.90, 0.90, 81, 0.400, pot, tilt, ("CT1",), vel, 1.5e-3, 2.0,
        {"CT1": 1.0}, 150.0, 0.090, 2, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
""",
            "call": "pcent_log_rate(0.200, *args)",
            "gold_call": "_oracle_pcent_log_rate(0.200, *args)",
        },
        {
            "setup": """import numpy as np
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
args = (-0.90, 0.90, 81, 0.400, pot, {}, ("CT1",), vel, 1.5e-3, 2.0,
        {"CT1": 1.0}, 300.0, 0.185, 2, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
def run(f):
    try:
        f(0.280, *args); return 0             # the tilt of CT1 is missing
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pcent_log_rate)",
            "gold_call": "run(_oracle_pcent_log_rate)",
        },
        {
            "setup": """import numpy as np
pot = {"I": (0.595, 0.180, 3.250), "II": (0.610, -0.175, 3.100),
       "CT1": (0.290, -0.080, 3.615)}
vel = {("I", "CT1"): 0.0250, ("CT1", "II"): -0.0220}
args = (-0.90, 0.90, 81, 0.400, pot, {"CT1": 0.0}, ("CT1",), vel, 1.5e-3, 2.0,
        {"CT1": 1.0}, 300.0, 0.185, 2, 2.07500e-3, 6.582119569e-16, 8.617333262e-5)
def run(f):
    try:
        f(-0.280, *args); return 0            # negative reorganisation energy
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pcent_log_rate)",
            "gold_call": "run(_oracle_pcent_log_rate)",
        },
    ]
