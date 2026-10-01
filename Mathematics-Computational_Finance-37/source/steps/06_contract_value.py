"""
Compute the fair initial value of the 3-year surrenderable equity-linked contract via backward obstacle recursion.

Given the (S1,S2,S3) equity-index joint law (already marginalized over the volatility coordinates in the previous step) and the two survival probabilities (from the GTFK step, evaluated at one and two years), the contract's fair initial value is obtained by a backward obstacle recursion over the three policy years. At each of the first two anniversaries, the surviving policyholder's continuation value subtracts the further contribution D as a real cost (paid to keep the contract going), separately from crediting that same D to the fund itself -- the source paper's own continuation-value definition treats the premium as an expense weighed against the alternative of surrendering, not merely as an amount that grows the fund forward. Without this subtraction, continuation would trivially dominate surrender in every scenario, making the surrender option meaningless. At maturity (year 3), the death and survival benefits use the identical accumulating guarantee formula, so no mortality split (and no premium subtraction, since no continuation decision is made there) is needed. At anniversary 2, the alive branch has genuine surrender-versus-continue optionality that the death branch lacks, so the year-2 survival/death split (derived from the two given survival probabilities) is load-bearing. At anniversary 1, the same structure repeats using the one-year survival probability directly.

Returns
-------
U0 : float, the contract's fair initial value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def contract_value(joint123: np.ndarray, S_grid: np.ndarray, D: float, gm: float,
                    alpha_s: float, Q1: float, Q2: float) -> float:
    """Compute the fair initial value of the 3-year surrenderable
    equity-linked contract via backward obstacle recursion.

    Parameters
    ----------
    joint123 : np.ndarray
        (nS, nS, nS) joint law over (S1, S2, S3), already marginalized over
        the volatility coordinates.
    S_grid : np.ndarray
        1D array of nS candidate index levels (index level 1.0 at
        inception).
    D : float
        Contribution credited at inception and at each continuation
        (> 0).
    gm : float
        Guarantee accumulation rate per year.
    alpha_s : float
        Surrender fraction of the current fund value, in (0, 1].
    Q1 : float
        Survival probability to anniversary 1 (in (0, 1]).
    Q2 : float
        Survival probability to anniversary 2 (in (0, 1], and <= Q1).

    Returns
    -------
    U0 : float
        The contract's fair initial value.

    Raises
    ------
    ValueError
        If Q1 or Q2 is not in (0, 1], if Q2 > Q1, or if D <= 0.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_contract_value(joint123: np.ndarray, S_grid: np.ndarray, D: float, gm: float,
                            alpha_s: float, Q1: float, Q2: float) -> float:
    import numpy as np
    joint123 = np.asarray(joint123, dtype=float)
    S_grid = np.asarray(S_grid, dtype=float)
    if not (0 < Q1 <= 1) or not (0 < Q2 <= 1):
        raise ValueError("Q1 and Q2 must be in (0, 1]")
    if Q2 > Q1 + 1e-12:
        raise ValueError("Q2 must be <= Q1")
    if D <= 0:
        raise ValueError("D must be positive")

    F0, S0 = 0.0, 1.0
    s0, d0 = Q1, 1.0 - Q1
    s1, d1 = Q2 / Q1, 1.0 - Q2 / Q1
    G1 = D * np.exp(gm * 1)
    G2 = D * np.exp(gm * 2) + D * np.exp(gm * 1)
    G3 = D * np.exp(gm * 3) + D * np.exp(gm * 2) + D * np.exp(gm * 1)

    p1 = joint123.sum(axis=(1, 2))

    F1_plus_inception = F0 + D
    U0 = 0.0
    for i1 in range(len(S_grid)):
        if p1[i1] < 1e-14:
            continue
        S1 = S_grid[i1]
        F1_minus = F1_plus_inception * (S1 / S0)
        B1 = max(F1_minus, G1)
        surrender1 = alpha_s * F1_minus

        w_s2 = joint123[i1, :, :].sum(axis=1)
        w_s2 = w_s2 / w_s2.sum()
        F1_plus = F1_minus + D
        C1 = 0.0
        for i2 in range(len(S_grid)):
            if w_s2[i2] < 1e-14:
                continue
            S2 = S_grid[i2]
            F2_minus = F1_plus * (S2 / S1)
            B2 = max(F2_minus, G2)
            surrender2 = alpha_s * F2_minus

            w_s3 = joint123[i1, i2, :]
            w_s3 = w_s3 / w_s3.sum()
            F2_plus = F2_minus + D
            C2 = 0.0
            for i3 in range(len(S_grid)):
                if w_s3[i3] < 1e-14:
                    continue
                S3 = S_grid[i3]
                F3_minus = F2_plus * (S3 / S2)
                C2 += w_s3[i3] * max(F3_minus, G3)
            U2 = max(C2 - D, surrender2)
            C1 += w_s2[i2] * (s1 * U2 + d1 * B2)
        U1 = max(C1 - D, surrender1)

        U0 += p1[i1] * (s0 * U1 + d0 * B1)
    return U0

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal case: a small synthetic 2-point index/vol instance ---
        {
            "setup": """import numpy as np
joint123 = np.array([0.11408519359808582, 0.1486707537952537, 0.1253000809216158, 0.11326817393468144, 0.088067694135561, 0.13426592896451614, 0.09096390915257357, 0.18537826549771252]).reshape(2, 2, 2)
S_grid = np.array([0.8, 1.2])
def run_model():
    return contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 0.9, 0.85)
def run_gold():
    return _oracle_contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 0.9, 0.85)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-6,
        },
        # --- Boundary case: certain survival (Q1=Q2=1.0), same instance ---
        {
            "setup": """import numpy as np
joint123 = np.array([0.11408519359808582, 0.1486707537952537, 0.1253000809216158, 0.11326817393468144, 0.088067694135561, 0.13426592896451614, 0.09096390915257357, 0.18537826549771252]).reshape(2, 2, 2)
S_grid = np.array([0.8, 1.2])
def run_model():
    return contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 1.0, 1.0)
def run_gold():
    return _oracle_contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 1.0, 1.0)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-6,
        },
        # --- Edge case: Q2 > Q1 should raise ---
        {
            "setup": """import numpy as np
joint123 = np.array([0.11408519359808582, 0.1486707537952537, 0.1253000809216158, 0.11326817393468144, 0.088067694135561, 0.13426592896451614, 0.09096390915257357, 0.18537826549771252]).reshape(2, 2, 2)
S_grid = np.array([0.8, 1.2])
def run_model():
    try:
        contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 0.8, 0.9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_contract_value(joint123, S_grid, 100.0, 0.02, 1.0, 0.8, 0.9)
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
