"""
Fold a table of vibronic couplings into the rate constant, weighting every reactant- product pair by the thermal population of its reactant member and by how readily the bath brings that pair into resonance.

With the coupling between every pair of vibronic states in hand, the rate is a golden- rule sum over those pairs. Three ingredients turn the table of couplings into a number, and each one carries a convention that changes the answer.

The reactant vibronic states are populated thermally, and only the reactant ones: the product states are the destination, not part of the equilibrium. The populations are the normalised Boltzmann weights of the reactant ladder at the given temperature. Because the ladder is quoted as absolute energies of order a few electronvolts while the thermal energy is a few tens of millielectronvolts, the exponentials must be taken relative to a reference on the ladder rather than of the raw energies, or every weight underflows to zero.

The spectral factor measures how easily the bath supplies or absorbs the difference between the two vibronic energies, and it is not given here as a formula because working it out is part of the step. It is the overlap of two line shapes: the donor emission line shape of the reactant level and the acceptor absorption line shape of the product level, multiplied together and integrated over the angular frequency of the transition. Each line shape is Gaussian with a standard deviation equal to the supplied width in energy, and each is normalised to unit area over angular frequency rather than over energy, which is where the factor gets its dimensions. The two centres are separated by the gap, meaning the product vibronic energy minus the reactant vibronic energy, displaced by the reorganisation energy, so the overlap is largest when that gap equals minus the reorganisation energy and the pair nearest resonance need not be the pair of ground levels.

Two things follow from that definition and both change the answer. The overlap of two Gaussians of equal width is itself a Gaussian, but a wider one than either, so the width that appears in the result is not the width that was supplied. And normalising over angular frequency rather than over energy leaves a factor of hbar behind, so the spectral factor carries units of time and the prefactor one over four pi squared hbar squared turns the sum into a rate in inverse seconds.

The rate is then the sum over every pair of the population, the squared coupling and the spectral factor, divided by four pi squared hbar squared, and what is returned is its base-ten logarithm with the rate expressed in inverse seconds. A rate that is not a positive finite number means the couplings or the constants are inconsistent and should be rejected rather than returned as a logarithm of zero.

Returns
-------
float, the base-ten logarithm of the rate constant expressed in inverse seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rate_from_couplings(v_total: np.ndarray, e_reactant: np.ndarray,
                        e_product: np.ndarray, temperature: float,
                        reorganization: float, width: float, hbar: float,
                        kb: float) -> float:
    '''Base-ten logarithm of the rate constant implied by a table of couplings.

    Weights every reactant-product vibronic pair by the thermal population of
    its reactant member and by the spectral factor of its energy gap, sums, and
    returns the logarithm of the resulting rate.

    Parameters
    ----------
    v_total : np.ndarray
        (n_mu, n_nu) total vibronic coupling in eV between every reactant and
        product vibronic level.
    e_reactant : np.ndarray
        (n_mu,) reactant vibronic energies in eV.
    e_product : np.ndarray
        (n_nu,) product vibronic energies in eV.
    temperature : float
        Positive temperature in kelvin.
    reorganization : float
        Reorganisation energy in eV.
    width : float
        Positive Gaussian width of the spectral factor in eV.
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
        If v_total does not match the two level arrays in shape, if any array is
        empty or non-finite, if temperature, width, hbar or kb is not positive,
        or if the resulting rate is not a positive finite number.
    '''
    return log_rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rate_from_couplings(v_total: np.ndarray, e_reactant: np.ndarray,
                                e_product: np.ndarray, temperature: float,
                                reorganization: float, width: float, hbar: float,
                                kb: float) -> float:
    """Reference implementation."""
    v = np.asarray(v_total, dtype=float)
    er = np.asarray(e_reactant, dtype=float)
    ep = np.asarray(e_product, dtype=float)
    for arr, name in ((er, "e_reactant"), (ep, "e_product")):
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} must be finite")
    if v.ndim != 2 or v.shape != (er.size, ep.size):
        raise ValueError("v_total must be two-dimensional and match the level arrays")
    if not np.isfinite(v).all():
        raise ValueError("v_total must be finite")
    if not (temperature > 0.0):
        raise ValueError("temperature must be positive")
    if not (width > 0.0):
        raise ValueError("width must be positive")
    if not (hbar > 0.0):
        raise ValueError("hbar must be positive")
    if not (kb > 0.0):
        raise ValueError("kb must be positive")
    if not np.isfinite(reorganization):
        raise ValueError("reorganization must be finite")

    weights = np.exp(-(er - er.min()) / (kb * float(temperature)))
    populations = weights / weights.sum()

    gap = ep[None, :] - er[:, None]
    spectral = (hbar / (2.0 * float(width) * np.sqrt(np.pi))) * np.exp(
        -(gap + float(reorganization)) ** 2 / (4.0 * float(width) ** 2))

    rate = float((populations[:, None] * v ** 2 * spectral).sum()
                 / (4.0 * np.pi ** 2 * hbar ** 2))
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("the computed rate constant is not a positive finite number")
    return float(np.log10(rate))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: a three-level table at room temperature
er = np.array([3.2458, 3.5748, 3.5826])
ep = np.array([3.1028, 3.4302, 3.4357])
v = np.array([[3.1e-06, -1.2e-04, 4.0e-06],
              [-4.0e-05, -2.8e-05, 3.9e-05],
              [-2.3e-04, -1.1e-05, -7.1e-05]])
""",
            "call": "rate_from_couplings(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# normal: the same table at 77 K, where the reactant ground level already holds
# essentially all of the population, so the answer barely moves
er = np.array([3.2458, 3.5748, 3.5826])
ep = np.array([3.1028, 3.4302, 3.4357])
v = np.array([[3.1e-06, -1.2e-04, 4.0e-06],
              [-4.0e-05, -2.8e-05, 3.9e-05],
              [-2.3e-04, -1.1e-05, -7.1e-05]])
""",
            "call": "rate_from_couplings(v, er, ep, 77.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 77.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# normal: a smaller reorganisation energy moves the resonance onto a different
# pair, which changes the answer even though the couplings are untouched
er = np.array([3.2458, 3.5748, 3.5826])
ep = np.array([3.1028, 3.4302, 3.4357])
v = np.array([[3.1e-06, -1.2e-04, 4.0e-06],
              [-4.0e-05, -2.8e-05, 3.9e-05],
              [-2.3e-04, -1.1e-05, -7.1e-05]])
""",
            "call": "rate_from_couplings(v, er, ep, 300.0, 0.140, 0.185, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 300.0, 0.140, 0.185, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# boundary: one level on each side, so the sum is a single term and the
# population weight is exactly one
er = np.array([3.20])
ep = np.array([3.00])
v = np.array([[1.0e-4]])
""",
            "call": "rate_from_couplings(v, er, ep, 300.0, 0.200, 0.185, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 300.0, 0.200, 0.185, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# boundary: absolute energies of order a few eV shifted bodily by 500 eV. The
# populations and hence the answer must be untouched by the shift, which they
# are only if the Boltzmann weights are taken relative to a reference.
er = np.array([3.2458, 3.5748, 3.5826]) + 500.0
ep = np.array([3.1028, 3.4302, 3.4357]) + 500.0
v = np.array([[3.1e-06, -1.2e-04, 4.0e-06],
              [-4.0e-05, -2.8e-05, 3.9e-05],
              [-2.3e-04, -1.1e-05, -7.1e-05]])
""",
            "call": "rate_from_couplings(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# edge: a rectangular table, three reactant levels against two product ones
er = np.array([3.20, 3.55, 3.60])
ep = np.array([3.00, 3.40])
v = np.array([[1.0e-4, -2.0e-4], [3.0e-5, 4.0e-5], [-1.0e-5, 2.5e-5]])
""",
            "call": "rate_from_couplings(v, er, ep, 250.0, 0.230, 0.150, 6.582119569e-16, 8.617333262e-5)",
            "gold_call": "_oracle_rate_from_couplings(v, er, ep, 250.0, 0.230, 0.150, 6.582119569e-16, 8.617333262e-5)",
        },
        {
            "setup": """import numpy as np
# invalid: an all-zero coupling table gives a rate of zero, whose logarithm
# does not exist, and must be rejected
er = np.array([3.20, 3.55])
ep = np.array([3.00, 3.40])
v = np.zeros((2, 2))
def run(f):
    try:
        f(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(rate_from_couplings)",
            "gold_call": "run(_oracle_rate_from_couplings)",
        },
        {
            "setup": """import numpy as np
# invalid: the coupling table does not match the two level arrays in shape
er = np.array([3.20, 3.55])
ep = np.array([3.00, 3.40, 3.42])
v = np.ones((2, 2)) * 1e-4
def run(f):
    try:
        f(v, er, ep, 300.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(rate_from_couplings)",
            "gold_call": "run(_oracle_rate_from_couplings)",
        },
        {
            "setup": """import numpy as np
# invalid: a temperature of absolute zero
er = np.array([3.20, 3.55])
ep = np.array([3.00, 3.40])
v = np.ones((2, 2)) * 1e-4
def run(f):
    try:
        f(v, er, ep, 0.0, 0.280, 0.185, 6.582119569e-16, 8.617333262e-5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(rate_from_couplings)",
            "gold_call": "run(_oracle_rate_from_couplings)",
        },
    ]
