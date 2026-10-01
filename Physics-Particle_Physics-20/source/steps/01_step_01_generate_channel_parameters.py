"""
Generate the deterministic ensemble of on/off channel parameters (s_i, b_i, tau_i) for the multi-channel sensitivity study from a single seeded pseudo-random draw.

The study is defined over an ensemble of independent on/off counting channels. Each channel is specified by its expected signal yield s, its expected background yield b, and the exposure ratio tau of the control region to the signal region. To make the instance fully reproducible, the ensemble is fixed by one call to NumPy's PCG64 generator: the matrix u = numpy.random.default_rng(seed).random((n_channels, 3)) is drawn once, and its columns are mapped affinely to the physical ranges s = 0.8 + 4.2*u[:, 0], b = 0.3 + 5.7*u[:, 1], tau = 0.5 + 2.5*u[:, 2]. No other random numbers are used anywhere in the pipeline.

Returns
-------
numpy.ndarray of shape (n_channels, 3), float64, columns [s, b, tau]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_channel_parameters(n_channels: int, seed: int):
    '''Generate the (n_channels, 3) array of channel parameters.

    Parameters
    ----------
    n_channels : int
        Number of on/off channels; must be a positive integer.
    seed : int
        Seed of the single numpy.random.default_rng draw; must be an
        integer (booleans are not accepted).

    Raises
    ------
    ValueError
        If n_channels is not a positive integer, or seed is not an
        integer, or either argument is a boolean.

    Returns
    -------
    params : numpy.ndarray
        Array of shape (n_channels, 3) whose columns are, in order,
        s = 0.8 + 4.2*u[:, 0], b = 0.3 + 5.7*u[:, 1],
        tau = 0.5 + 2.5*u[:, 2], with
        u = numpy.random.default_rng(seed).random((n_channels, 3)).
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_generate_channel_parameters(n_channels: int, seed: int):
    if isinstance(n_channels, bool) or not isinstance(n_channels, (int, np.integer)):
        raise ValueError("n_channels must be a positive integer")
    if n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(n_channels), 3))
    s = 0.8 + 4.2 * u[:, 0]
    b = 0.3 + 5.7 * u[:, 1]
    tau = 0.5 + 2.5 * u[:, 2]
    return np.column_stack([s, b, tau]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: normal ensemble ---
        {
            "setup": """import numpy as np
n_channels = 5
seed = 21
""",
            "call": "generate_channel_parameters(n_channels, seed)",
            "gold_call": "_oracle_generate_channel_parameters(n_channels, seed)",
        },
        # --- Valid: boundary, single channel ---
        {
            "setup": """import numpy as np
n_channels = 1
seed = 0
""",
            "call": "generate_channel_parameters(n_channels, seed)",
            "gold_call": "_oracle_generate_channel_parameters(n_channels, seed)",
        },
        # --- Valid: edge, large seed ---
        {
            "setup": """import numpy as np
n_channels = 12
seed = 987654321
""",
            "call": "generate_channel_parameters(n_channels, seed)",
            "gold_call": "_oracle_generate_channel_parameters(n_channels, seed)",
        },
        # --- Invalid: n_channels = 0 ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        generate_channel_parameters(0, 21)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_channel_parameters(0, 21)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-integer seed ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        generate_channel_parameters(5, 2.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_channel_parameters(5, 2.5)
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
