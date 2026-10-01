"""
Integrate the exclusive probability for an ordered shower history containing an arbitrary number of designated emissions.

For a history with $N$ designated emissions at times $0<t_1<*\d*ots<t_N<L$, with total hazard $*\l*ambda_k$ between consecutive emissions ($t_0=0$, $t_{N+1}=L$) and designated rates $r_1,*\d*ots,r_N$, the exclusive-history probability is


$$

P_N=*\B*ig(*\p*rod_{i=1}^{N} r_i*\B*ig)*\i*nt_{0<t_1<*\d*ots<t_N<L}*\e*xp*\B*ig[-*\s*um_{k=0}^{N}*\l*ambda_k*\,*(t_{k+1}-t_k)*\B*ig]dt_1*\c*dots dt_N.


$$

The result must be accurate to a relative error of $10^{-11}$ for any hazards, including hazards that are exactly equal or that differ by as little as $10^{-12}$.

Returns
-------
float, the dimensionless exclusive-history probability
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_exclusive_history_probability(
    segment_hazards: "np.ndarray",
    split_rates: "np.ndarray",
    length: float,
) -> float:
    """Integrate a general ordered exclusive shower-history probability.

    Parameters
    ----------
    segment_hazards : np.ndarray
        One-dimensional array of shape (N+1,) containing the total
        inelastic hazards lambda_0, ..., lambda_N in GeV for the
        propagation segments before, between, and after emissions.
    split_rates : np.ndarray
        One-dimensional array of shape (N,) containing the designated
        emission rates r_1, ..., r_N in GeV.
    length : float
        Total propagation length in GeV^(-1).

    Returns
    -------
    probability : float
        Dimensionless exclusive-history probability.
    """
    return probability

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_integrate_exclusive_history_probability(
    segment_hazards: "np.ndarray",
    split_rates: "np.ndarray",
    length: float,
) -> float:
    segment_hazards = np.asarray(segment_hazards, dtype=float)
    split_rates = np.asarray(split_rates, dtype=float)

    n_emissions = split_rates.size
    generator = -np.diag(segment_hazards)
    if n_emissions > 0:
        generator[
            np.arange(n_emissions),
            np.arange(1, n_emissions + 1),
        ] = split_rates

    # Incorporating transition rates avoids a separately overflowing product.
    return float(expm(generator * float(length))[0, n_emissions])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.0740146296576149, 0.519412843255201],
    dtype=float,
)
split_rates = np.array(
    [0.0139125134083925],
    dtype=float,
)
length = 2.0""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.1, 0.4, 0.9],
    dtype=float,
)
split_rates = np.array(
    [0.7, 0.6],
    dtype=float,
)
length = 1.5""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.05, 0.3, 0.8, 1.2],
    dtype=float,
)
split_rates = np.array(
    [0.9, 0.7, 0.5],
    dtype=float,
)
length = 2.0""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.2, 0.200000001, 0.200000002, 0.7],
    dtype=float,
)
split_rates = np.array(
    [0.9, 0.8, 0.7],
    dtype=float,
)
length = 2.0""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.5, 0.5, 0.5],
    dtype=float,
)
split_rates = np.array(
    [1.0, 1.0],
    dtype=float,
)
length = 1.0""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.12, 0.120000000001, 0.9],
    dtype=float,
)
split_rates = np.array(
    [0.8, 0.6],
    dtype=float,
)
length = 1.7""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.array(
    [0.300, 0.302],
    dtype=float,
)
split_rates = np.array(
    [0.070],
    dtype=float,
)
length = 2.0""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
        {
            "setup": """import numpy as np
segment_hazards = np.full(105, 1000.0)
split_rates = np.full(104, 1000.0)
length = 0.104""",
            "call": "integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "gold_call": "_oracle_integrate_exclusive_history_probability(segment_hazards.copy(), split_rates.copy(), length)",
            "tol": 1e-11,
        },
    ]
