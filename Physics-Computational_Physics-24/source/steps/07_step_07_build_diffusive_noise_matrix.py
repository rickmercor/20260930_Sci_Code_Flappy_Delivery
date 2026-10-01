"""
Assemble the diffusion matrix of the Itô stochastic point-kinetic description, in which the whole stochastic forcing is a single scalar Brownian term attached to the neutron balance.

The diffusive stochastic point-kinetic description keeps one Brownian term whose amplitude scales as the square root of the neutron population and leaves the precursor equations purely deterministic. Its diffusion matrix is therefore of rank one and supported entirely on the neutron component.

Returns
-------
np.ndarray of shape (1 + 2 * n_groups, 1 + 2 * n_groups), float: the symmetric diffusion matrix in inverse seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_diffusive_noise_matrix(state: np.ndarray,
                                 noise_coefficient: float) -> np.ndarray:
    """Assemble the diffusion matrix of the Itô point-kinetic description.

    Parameters
    ----------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the neutron population, the
        in-core precursor populations and the ex-core precursor populations in
        the ordering of sub-problem 03; every entry is non-negative and the
        length is odd and at least three.
    noise_coefficient : float
        The scalar coefficient in inverse square-root seconds that multiplies
        the square root of the neutron population in the diffusion term of the
        neutron equation, the fourth entry returned by sub-problem 02
        (noise_coefficient >= 0).

    Returns
    -------
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of shape
        (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds, the
        instantaneous covariance rate of the diffusive description at the
        supplied state.

    Raises
    ------
    ValueError
        If ``state`` is not a one-dimensional finite array of non-negative
        entries whose length is odd and at least three, or if
        ``noise_coefficient`` is not a finite non-negative number.
    """
    return diffusion_matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_diffusive_noise_matrix(state: np.ndarray,
                                         noise_coefficient: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(noise_coefficient, (int, float, np.floating, np.integer))
            and not isinstance(noise_coefficient, bool)
            and np.isfinite(noise_coefficient)):
        raise ValueError("noise_coefficient must be a finite number")
    noise_coefficient = float(noise_coefficient)
    if noise_coefficient < 0.0:
        raise ValueError("noise_coefficient must be non-negative")

    populations = np.asarray(state, dtype=float)
    if populations.ndim != 1:
        raise ValueError("state must be a one-dimensional array")
    size = populations.size
    if size < 3 or size % 2 == 0:
        raise ValueError("state must have odd length of at least three")
    if not np.all(np.isfinite(populations)):
        raise ValueError("state must contain only finite entries")
    if np.any(populations < 0.0):
        raise ValueError("state entries must be non-negative")

    # The single Brownian term acts on the neutron balance alone, with a
    # variance rate proportional to the neutron population; the precursor
    # equations of this description carry no noise at all, so every other entry
    # of the matrix is identically zero.
    matrix = np.zeros((size, size), dtype=float)
    matrix[0, 0] = noise_coefficient ** 2 * float(populations[0])

    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the testbed stationary state (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([352.0, 2492.439075512, 7445.082213733, 2379.199688901,
                  2224.164601468, 208.067409903, 30.604978237,
                  3596.593179121, 8983.507949306, 1660.872378464,
                  778.360315460, 22.739607572, 1.320603284])
noise_coefficient = 44.535132827
""",
            "call": "sig(build_diffusive_noise_matrix(state, noise_coefficient), 1.0e6)",
            "gold_call": "sig(_oracle_build_diffusive_noise_matrix(state, noise_coefficient), 1.0e6)",
        },
        # --- Valid: a two-group state with a much smaller neutron population ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([12.5, 900.0, 340.0, 210.0, 55.0])
noise_coefficient = 20.5
""",
            "call": "sig(build_diffusive_noise_matrix(state, noise_coefficient), 1.0e3)",
            "gold_call": "sig(_oracle_build_diffusive_noise_matrix(state, noise_coefficient), 1.0e3)",
        },
        # --- Boundary: a vanishing neutron population, which switches the noise
        #     off entirely while the precursor inventory is still large ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([0.0, 4200.0, 3100.0])
noise_coefficient = 44.535132827
""",
            "call": "sig(build_diffusive_noise_matrix(state, noise_coefficient), 1.0e0)",
            "gold_call": "sig(_oracle_build_diffusive_noise_matrix(state, noise_coefficient), 1.0e0)",
        },
        # --- Edge: a vanishing noise coefficient on a large population, the
        #     deterministic limit of the description ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
state = np.array([1.0e6, 4200.0, 3100.0])
noise_coefficient = 0.0
""",
            "call": "sig(build_diffusive_noise_matrix(state, noise_coefficient), 1.0e0)",
            "gold_call": "sig(_oracle_build_diffusive_noise_matrix(state, noise_coefficient), 1.0e0)",
        },
        # --- Invalid: a negative noise coefficient ---
        {
            "setup": """import numpy as np
state = np.array([352.0, 900.0, 340.0, 210.0, 55.0])
def run_model():
    try:
        build_diffusive_noise_matrix(state, -3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_diffusive_noise_matrix(state, -3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an even-length state, which cannot split into a neutron
        #     component and two equal precursor blocks ---
        {
            "setup": """import numpy as np
state = np.array([352.0, 900.0, 340.0, 210.0])
def run_model():
    try:
        build_diffusive_noise_matrix(state, 44.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_diffusive_noise_matrix(state, 44.5)
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
