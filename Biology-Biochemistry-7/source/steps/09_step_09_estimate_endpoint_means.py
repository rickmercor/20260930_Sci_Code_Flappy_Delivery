"""
Estimate endpoint protein means for both simulator laws.

Weak error compares ensemble expectations. Disjoint path-indexed random streams keep the finite Monte Carlo benchmark deterministic and reproducible.

Returns
-------
np.ndarray: [exact_mean, approximate_mean, exact_standard_error, approximate_standard_error].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def estimate_endpoint_means(parameters: "np.ndarray", final_time: float,
                            tau: float, n_paths: int,
                            base_seed: int) -> "np.ndarray":
    """Return exact and approximate mean protein counts and standard errors.

    Both standard errors are over the paths, with one degree of freedom
    removed, so at least two paths are required.

    Raises
    ------
    ValueError
        If ``n_paths`` is not an integer of at least two, if ``base_seed`` is
        not a non-negative integer, or if the simulation arguments are
        inadmissible.
    """
    return estimates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_estimate_endpoint_means(parameters: "np.ndarray", final_time: float,
                                     tau: float, n_paths: int,
                                     base_seed: int) -> "np.ndarray":
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths < 2:
        raise ValueError("n_paths must be an integer of at least two")
    if isinstance(base_seed, bool) or not isinstance(base_seed, (int, np.integer)) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    exact = np.empty(int(n_paths), dtype=float)
    approximate = np.empty(int(n_paths), dtype=float)
    for index in range(int(n_paths)):
        exact[index] = _oracle_simulate_exact_gene_path(
            parameters, final_time, int(base_seed) + 2 * index)[1]
        approximate[index] = _oracle_simulate_tau_gene_path(
            parameters, final_time, tau, int(base_seed) + 2 * index + 1)[1]
    exact_mean = float(np.mean(exact))
    approximate_mean = float(np.mean(approximate))
    exact_se = float(np.std(exact, ddof=1) / np.sqrt(n_paths))
    approximate_se = float(np.std(approximate, ddof=1) / np.sqrt(n_paths))
    return np.array([exact_mean, approximate_mean, exact_se, approximate_se])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\np=np.array([2,.4,.5,.2,.1,2,5,.2,2,3.])",
            "call": "float(np.dot(estimate_endpoint_means(p,4,1,5,7),[1,2,3,4]))",
            "gold_call": "float(np.dot(_oracle_estimate_endpoint_means(p,4,1,5,7),[1,2,3,4]))",
        },
        {
            "setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,1,5,0,1,2.])",
            "call": "float(np.dot(estimate_endpoint_means(p,2,.1,3,0),[1,2,3,4]))",
            "gold_call": "float(np.dot(_oracle_estimate_endpoint_means(p,2,.1,3,0),[1,2,3,4]))",
        },
        {
            "setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,3,5,0,1,2.])",
            "call": "float(np.dot(estimate_endpoint_means(p,4,3,4,99),[4,3,2,1]))",
            "gold_call": "float(np.dot(_oracle_estimate_endpoint_means(p,4,3,4,99),[4,3,2,1]))",
        },
        {
            # The benchmark configuration itself, so both endpoint means and both
            # standard errors are pinned at step level as well as end to end.
            "setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.dot(estimate_endpoint_means(p,60,2.5,80,314159),[1,4,2,3]))",
            "gold_call": "float(np.dot(_oracle_estimate_endpoint_means(p,60,2.5,80,314159),[1,4,2,3]))",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\ndef bad():\n    try: estimate_endpoint_means(p,5,1,1,0); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_estimate_endpoint_means(p,5,1,1,0); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
