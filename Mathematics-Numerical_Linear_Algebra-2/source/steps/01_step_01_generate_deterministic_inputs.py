"""
Generate the numerical inputs required by the subsequent randomized preconditioning subproblems and return them as a single state. Use one NumPy default_rng(seed), draw $G\in\mathbb{R}^{p\times p}$, then $C\in\mathbb{R}^{p\times m}$, then $\Omega\in\mathbb{R}^{m\times k}$, all via standard_normal, and form $E=GG^T+pI_p$.

The randomized preconditioning construction operates on matrix data associated with the coupled constraint system. This benchmark uses a deterministic numerical instance so that the same state is reproduced for every evaluation.

Returns
-------
Tuple[np.ndarray, np.ndarray, np.ndarray] — $(E,C,\Omega)$ with shapes $(p,p)$, $(p,m)$, $(m,k)$, as native NumPy float arrays.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def generate_deterministic_inputs(
    seed: int, p: int, m: int, k: int
) -> tuple:
    r"""
    Deterministically generate the numerical inputs required by the
    subsequent randomized preconditioning subproblems.

    Parameters
    ----------
    seed : int
        Non-negative RNG seed.
    p : int
        Dimension parameter for the SPD block.
    m : int
        Column dimension of the constraint matrix.
    k : int
        Sketch size.

    Returns
    -------
    tuple
        $(E,C,\Omega)$.

    Raises
    ------
    ValueError
        If seed is negative, $p<1$, $m<p$, or $k<1$.
    """
    return E, C, Omega

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_generate_deterministic_inputs(
    seed: int, p: int, m: int, k: int
) -> tuple:
    if not (isinstance(seed, (int, np.integer)) and seed >= 0):
        raise ValueError("seed must be a non-negative integer")

    if not (isinstance(p, (int, np.integer)) and p >= 1):
        raise ValueError("p must be a positive integer")

    if not (isinstance(m, (int, np.integer)) and m >= p):
        raise ValueError("m must be an integer >= p")

    if not (isinstance(k, (int, np.integer)) and k >= 1):
        raise ValueError("k must be a positive integer")

    rng = np.random.default_rng(int(seed))

    G = rng.standard_normal((int(p), int(p)))
    E = G @ G.T + float(p) * np.eye(int(p), dtype=float)

    C = rng.standard_normal((int(p), int(m)))
    Omega = rng.standard_normal((int(m), int(k)))

    return (
        E.astype(float),
        C.astype(float),
        Omega.astype(float),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: exact benchmark configuration ---
        {
            "setup": """seed, p, m, k = 2026, 10, 80, 9
""",
            "call": "np.concatenate([a.ravel() for a in generate_deterministic_inputs(seed, p, m, k)])",
            "gold_call": "np.concatenate([a.ravel() for a in _oracle_generate_deterministic_inputs(seed, p, m, k)])",
        },

        # --- Valid: minimal p=1 ---
        {
            "setup": """seed, p, m, k = 7, 1, 5, 1
""",
            "call": "np.concatenate([a.ravel() for a in generate_deterministic_inputs(seed, p, m, k)])",
            "gold_call": "np.concatenate([a.ravel() for a in _oracle_generate_deterministic_inputs(seed, p, m, k)])",
        },

        # --- Valid: square C ---
        {
            "setup": """seed, p, m, k = 3, 6, 6, 6
""",
            "call": "np.concatenate([a.ravel() for a in generate_deterministic_inputs(seed, p, m, k)])",
            "gold_call": "np.concatenate([a.ravel() for a in _oracle_generate_deterministic_inputs(seed, p, m, k)])",
        },

        # --- Invalid: m < p ---
        {
            "setup": """seed, p, m, k = 1, 10, 5, 3

def run_model():
    try:
        generate_deterministic_inputs(seed, p, m, k)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_generate_deterministic_inputs(seed, p, m, k)
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
