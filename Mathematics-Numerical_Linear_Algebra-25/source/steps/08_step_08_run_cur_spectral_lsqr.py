"""
Chain the seven numerical steps and return the requested coordinate.

The final step chains the generated data, one sketch, CUR indices, CUR core, captured spectrum, packed inverse factors, and fixed-step LSQR solve. It returns the requested first coordinate.

Returns
-------
native Python float: the first coordinate of the final iterate
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_cur_spectral_lsqr(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    n_sketch: int,
    sketch_seed: int,
    ell: int,
    mu: float,
    niter: int,
) -> float:
    """Return the first coordinate produced by the complete pipeline.

    Parameters
    ----------
    m, n : int
        Dimensions with m >= n >= 1.
    s : np.ndarray
        Positive singular values, shape (n,).
    data_seed : int
        Seed for A and b.
    n_sketch : int
        Gaussian sketch rows.
    sketch_seed : int
        Seed for the sketch.
    ell : int
        CUR block size.
    mu : float
        Regularization parameter.
    niter : int
        Number of LSQR iterations.

    Returns
    -------
    x0 : float
        First coordinate of the resulting iterate.

    Raises
    ------
    ValueError
        Propagated from the underlying steps whenever any argument is invalid,
        for example when m >= n >= 1 does not hold, when s is not positive
        and finite, when ell exceeds min(m, n, n_sketch), when mu is
        negative, or when niter is not an integer >= 1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_cur_spectral_lsqr(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    n_sketch: int,
    sketch_seed: int,
    ell: int,
    mu: float,
    niter: int,
) -> float:
    data = _oracle_construct_clustered_ls_data(m, n, s, data_seed)
    A = data[:, :-1]
    b = data[:, -1]
    Y = _oracle_form_sketched_range(A, n_sketch, sketch_seed)
    index_vector = _oracle_select_cur_indices(A, Y, ell)
    U = _oracle_cur_core_matrix(A, index_vector)
    sigma = _oracle_cur_captured_singular_values(A, index_vector, U)
    pinv_factors = _oracle_build_spectral_pinv_factors(A, index_vector, U, sigma, mu)
    x = _oracle_preconditioned_lsqr(A, b, pinv_factors, mu, niter)
    return float(x[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
m, n = 12, 8
s = np.array([10.0, 9.0, 8.0, 0.45, 0.3, 0.22, 0.15, 0.1])
data_seed, n_sketch, sketch_seed = 7, 3, 11
ell, mu, niter = 2, 0.05, 2
""",
            "call": "run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
            "gold_call": "_oracle_run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
        },
        {
            "setup": """import numpy as np
m, n = 6, 4
s = np.array([5.0, 4.0, 0.2, 0.1])
data_seed, n_sketch, sketch_seed = 1, 2, 4
ell, mu, niter = 1, 0.0, 1
""",
            "call": "run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
            "gold_call": "_oracle_run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
        },
        {
            "setup": """import numpy as np
m, n = 5, 5
s = np.array([3.0, 2.5, 2.0, 0.4, 0.3])
data_seed, n_sketch, sketch_seed = 0, 3, 2
ell, mu, niter = 2, 1.0, 3
""",
            "call": "run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
            "gold_call": "_oracle_run_cur_spectral_lsqr(m, n, s, data_seed, n_sketch, sketch_seed, ell, mu, niter)",
        },
        {
            "setup": """import numpy as np
s = np.array([1.0, 0.5])
def run_model():
    try:
        run_cur_spectral_lsqr(3, 2, s, 0, 2, 1, 3, 0.1, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_cur_spectral_lsqr(3, 2, s, 0, 2, 1, 3, 0.1, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
s = np.array([2.0, 1.0])
def run_model():
    try:
        run_cur_spectral_lsqr(4, 2, s, 0, 2, 1, 1, -1.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_cur_spectral_lsqr(4, 2, s, 0, 2, 1, 1, -1.0, 1)
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
