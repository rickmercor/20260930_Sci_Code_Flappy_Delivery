"""
Search all integer allocations of a fixed reinforcement-ply budget between the two outer faces of an asymmetric joint and return the largest complete-decohesion separation.

The represented resin-rich-layer counts stay fixed while an allocation changes only the total upper and lower arm thicknesses. If `$p$` plies are assigned to the upper face from a budget `$n_reinforcement$`, the two total arm counts are `$(n_cohesive[0] + p, n_cohesive[1] + n_reinforcement - p)$`. For every allocation the full structural-cohesive pipeline is evaluated from the two mode-dependent arm sums through the bilinear failure separation.

Returns
-------
float: the largest effective separation at complete decohesion among all feasible integer allocations, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimal_reinforcement_failure_separation(
        n_cohesive: tuple = (15, 11), n_reinforcement: int = 12,
        t_ply: float = 0.1875, h_rr: float = 0.02286,
        e_rr: float = 4700.0, g_rr: float = 1715.0,
        tau_ic: float = 30.0, tau_iic: float = 60.0,
        g_ic: float = 0.212, g_iic: float = 0.774,
        eta: float = 2.1, disp_ratio: float = 0.75) -> float:
    """Return the best failure separation for a fixed reinforcement budget.

    Parameters
    ----------
    n_cohesive : tuple
        Two integers giving the represented upper and lower ply counts.
    n_reinforcement : int
        Total number of reinforcement plies to distribute between the upper
        and lower outer faces (n_reinforcement >= 0).
    t_ply, h_rr : float
        Ply thickness and resin-rich-layer thickness, both greater than zero.
    e_rr, g_rr : float
        Resin-rich-layer tensile and shear moduli, both greater than zero.
    tau_ic, tau_iic : float
        Opening and shearing strengths, both greater than zero.
    g_ic, g_iic : float
        Opening and shearing fracture toughnesses, both greater than zero.
    eta : float
        Benzeggagh-Kenane exponent (eta > 0).
    disp_ratio : float
        Tangential-to-normal separation ratio (disp_ratio >= 0).

    Returns
    -------
    delta_failure_max : float
        Maximum effective separation at complete decohesion over every integer
        allocation, as a native Python float.

    Raises ValueError if ``n_cohesive`` is not a sequence of exactly two
    integers, if ``n_reinforcement`` is not an integer greater than or equal to
    zero, if a scalar parameter other than ``disp_ratio`` is not a finite real
    number greater than zero, if ``disp_ratio`` is not a finite real number
    greater than or equal to zero, or if an earlier step rejects its inputs.

    Notes
    -----
    This is the final orchestrating step. For each integer upper allocation
    from zero through ``n_reinforcement``, construct the complete-arm ply
    counts and call the public functions of steps 01-06 in order. Return the
    largest failure separation without rounding. Include every import inside
    the function body.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_optimal_reinforcement_failure_separation(
        n_cohesive: tuple = (15, 11), n_reinforcement: int = 12,
        t_ply: float = 0.1875, h_rr: float = 0.02286,
        e_rr: float = 4700.0, g_rr: float = 1715.0,
        tau_ic: float = 30.0, tau_iic: float = 60.0,
        g_ic: float = 0.212, g_iic: float = 0.774,
        eta: float = 2.1, disp_ratio: float = 0.75) -> float:
    if not isinstance(n_cohesive, (tuple, list)) or len(n_cohesive) != 2:
        raise ValueError("n_cohesive must be a sequence of exactly two integers")
    for val in n_cohesive:
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("n_cohesive must contain integers only")
    if (isinstance(n_reinforcement, bool)
            or not isinstance(n_reinforcement, (int, np.integer))):
        raise ValueError("n_reinforcement must be an integer")
    n_reinforcement = int(n_reinforcement)
    if n_reinforcement < 0:
        raise ValueError("n_reinforcement must be >= 0")

    upper_base, lower_base = (int(n_cohesive[0]), int(n_cohesive[1]))
    best = -np.inf
    for upper_extra in range(n_reinforcement + 1):
        lower_extra = n_reinforcement - upper_extra
        n_total = (upper_base + upper_extra, lower_base + lower_extra)

        ratio_sums = {"opening": 0.0, "shear": 0.0}
        for mode in ("opening", "shear"):
            for n_coh, n_tot in zip((upper_base, lower_base), n_total):
                ratio_sums[mode] += _oracle_arm_ratio_sum(
                    mode, n_coh, n_tot, t_ply)

        k_n = _oracle_resin_rich_penalty_stiffness(
            ratio_sums["opening"], h_rr, e_rr)
        k_s = _oracle_resin_rich_penalty_stiffness(
            ratio_sums["shear"], h_rr, g_rr)
        delta_onset = _oracle_mixed_mode_onset_separation(
            k_n, k_s, tau_ic, tau_iic, disp_ratio)
        b_ratio = _oracle_mixed_mode_energy_ratio(k_n, k_s, disp_ratio)
        g_c = _oracle_benzeggagh_kenane_toughness(
            g_ic, g_iic, eta, b_ratio)
        value = _oracle_bilinear_failure_separation(
            k_n, k_s, disp_ratio, delta_onset, g_c)
        best = max(best, value)

    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Benchmark: the task's asymmetric 15/11 blocks and 12-ply budget.
        {
            "setup": """import numpy as np
""",
            "call": "optimal_reinforcement_failure_separation()",
            "gold_call": "_oracle_optimal_reinforcement_failure_separation()",
        },
        # Boundary: zero reinforcement leaves one feasible allocation.
        {
            "setup": """import numpy as np
""",
            "call": ("optimal_reinforcement_failure_separation((13, 9), 0, 0.1875, 0.02286,"
                     " 4700.0, 1715.0, 30.0, 60.0, 0.212, 0.774, 2.1, 0.75)"),
            "gold_call": ("_oracle_optimal_reinforcement_failure_separation((13, 9), 0, 0.1875,"
                          " 0.02286, 4700.0, 1715.0, 30.0, 60.0, 0.212, 0.774, 2.1, 0.75)"),
        },
        # Symmetric represented blocks with an even reinforcement budget.
        {
            "setup": """import numpy as np
""",
            "call": ("optimal_reinforcement_failure_separation((9, 9), 4, 0.2, 0.02, 4000.0,"
                     " 1500.0, 30.0, 60.0, 0.212, 0.774, 2.1, 0.75)"),
            "gold_call": ("_oracle_optimal_reinforcement_failure_separation((9, 9), 4, 0.2, 0.02,"
                          " 4000.0, 1500.0, 30.0, 60.0, 0.212, 0.774, 2.1, 0.75)"),
        },
        # Boundary: in pure opening every allocation has the same answer.
        {
            "setup": """import numpy as np
EXPECTED = 2.0 * 0.305 / 32.6
""",
            "call": ("optimal_reinforcement_failure_separation((7, 12), 5, 0.125, 0.0254,"
                     " 3400.0, 1300.0, 32.6, 98.0, 0.305, 2.77, 2.05, 0.0)"),
            "gold_call": "EXPECTED",
        },
        # Edge: one represented layer in each arm under an asymmetric budget.
        {
            "setup": """import numpy as np
""",
            "call": ("optimal_reinforcement_failure_separation((1, 1), 3, 0.25, 0.02286,"
                     " 1850.0, 560.0, 32.6, 98.0, 0.305, 2.77, 2.05, 0.4)"),
            "gold_call": ("_oracle_optimal_reinforcement_failure_separation((1, 1), 3, 0.25,"
                          " 0.02286, 1850.0, 560.0, 32.6, 98.0, 0.305, 2.77, 2.05, 0.4)"),
        },
        # Invalid: a negative reinforcement budget.
        {
            "setup": """import numpy as np
def run_model():
    try:
        optimal_reinforcement_failure_separation((15, 11), -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_optimal_reinforcement_failure_separation((15, 11), -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # Invalid: the represented-arm sequence must contain exactly two counts.
        {
            "setup": """import numpy as np
def run_model():
    try:
        optimal_reinforcement_failure_separation((15,), 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_optimal_reinforcement_failure_separation((15,), 12)
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
