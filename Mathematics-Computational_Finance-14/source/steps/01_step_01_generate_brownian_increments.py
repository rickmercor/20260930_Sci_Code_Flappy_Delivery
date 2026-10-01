"""
Generate the Brownian increments for the fixed Monte Carlo paths.

A Brownian increment over a uniform interval of length h has independent

components distributed as N(0, h). Drawing the complete tensor in time-major

order fixes both the pseudorandom stream and the association between increments,

paths, and Brownian dimensions.

Inputs

------

seed: Nonnegative seed for the NumPy random generator.

n_steps: Positive number of uniform time increments.

n_paths: Positive number of Monte Carlo paths.

dim_w: Positive Brownian dimension.

h: Positive uniform time step.

Returns

-------

increments: Float array of shape (n_steps, n_paths, dim_w).

Returns
-------
np.ndarray of shape (n_steps, n_paths, dim_w), Brownian increments as float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def generate_brownian_increments(
    seed: int, n_steps: int, n_paths: int, dim_w: int, h: float
) -> np.ndarray:
    """Generate time-major Brownian increments with a fixed NumPy stream.

    Parameters
    ----------
    seed : int
        Nonnegative seed for ``numpy.random.default_rng``.
    n_steps : int
        Positive number of time increments.
    n_paths : int
        Positive number of simulated paths.
    dim_w : int
        Positive Brownian dimension.
    h : float
        Positive finite uniform step size.

    Raises
    ------
    ValueError
        If the seed is not a nonnegative integer, any dimension is not a
        positive integer, or h is not finite and positive.

    Returns
    -------
    increments : np.ndarray
        Array of shape ``(n_steps, n_paths, dim_w)``.
    """
    return increments  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_generate_brownian_increments(
    seed: int, n_steps: int, n_paths: int, dim_w: int, h: float
) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (("n_steps", n_steps), ("n_paths", n_paths), ("dim_w", dim_w)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(h, (int, float, np.integer, np.floating)) or not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    rng = np.random.default_rng(int(seed))
    return rng.normal(0.0, np.sqrt(float(h)), size=(int(n_steps), int(n_paths), int(dim_w)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "seed = 260118634\nn_steps = 6\nn_paths = 7\ndim_w = 2\nh = 0.125\n",
            "call": "np.round(generate_brownian_increments(seed, n_steps, n_paths, dim_w, h), 12).tolist()",
            "gold_call": "np.round(_oracle_generate_brownian_increments(seed, n_steps, n_paths, dim_w, h), 12).tolist()",
        },
        {
            "setup": "seed = 0\nn_steps = 1\nn_paths = 1\ndim_w = 1\nh = 0.5\n",
            "call": "np.round(generate_brownian_increments(seed, n_steps, n_paths, dim_w, h), 12).tolist()",
            "gold_call": "np.round(_oracle_generate_brownian_increments(seed, n_steps, n_paths, dim_w, h), 12).tolist()",
        },
        {
            "setup": """def run_model():
    try:
        generate_brownian_increments(-1, 2, 2, 1, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_generate_brownian_increments(-1, 2, 2, 1, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
