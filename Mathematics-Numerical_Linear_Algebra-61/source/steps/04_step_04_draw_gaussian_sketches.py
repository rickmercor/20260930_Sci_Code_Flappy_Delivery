"""
Draw $n_{\mathrm{steps}}$ independent Gaussian sketches of shape $(m_r, q)$ from default_rng(seed) and pack them in order. Require $m_r$, $q$, $n_{\mathrm{steps}} \ge 1$.

The complementary residual is accessed only through a short sketch. A fixed generator stream makes those three embeddings deterministic and independent of the constraint-row projector.

Returns
-------
1d ndarray: packed (n_steps, m_r, q) sketches
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def draw_gaussian_sketches(m_r: int, q: int, n_steps: int, seed: int) -> np.ndarray:
    """Pack $n_{\mathrm{steps}}$ independent Gaussian matrices of shape $(m_r, q)$.

    Parameters
    ----------
    m_r : int
        Complementary row count, $m_r \ge 1$.
    q : int
        Sketch width, $q \ge 1$.
    n_steps : int
        Number of sketches, $n_{\mathrm{steps}} \ge 1$.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        Concatenation of $(n_{\mathrm{steps}}, m_r, q)$ and the sketches in order.

    Raises
    ------
    ValueError
        If $m_r$, $q$, or $n_{\mathrm{steps}}$ is not an integer, or if
        any of them is less than 1.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_draw_gaussian_sketches(m_r, q, n_steps, seed):
    for name, val in (("m_r", m_r), ("q", q), ("n_steps", n_steps)):
        if not isinstance(val, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    m_r, q, n_steps = int(m_r), int(q), int(n_steps)
    if min(m_r, q, n_steps) < 1:
        raise ValueError("require m_r, q, n_steps >= 1")
    rng = np.random.default_rng(int(seed))
    pieces = [np.array([float(n_steps), float(m_r), float(q)])]
    for _ in range(n_steps):
        pieces.append(rng.standard_normal((m_r, q)).ravel())
    return np.concatenate(pieces)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m_r, q, n_steps, seed = 9, 2, 3, 11
""",
            "call": "draw_gaussian_sketches(m_r, q, n_steps, seed)",
            "gold_call": "_oracle_draw_gaussian_sketches(m_r, q, n_steps, seed)",
        },
        {
            "setup": """import numpy as np
m_r, q, n_steps, seed = 4, 2, 2, 4
""",
            "call": "draw_gaussian_sketches(m_r, q, n_steps, seed)",
            "gold_call": "_oracle_draw_gaussian_sketches(m_r, q, n_steps, seed)",
        },
        {
            "setup": """import numpy as np
m_r, q, n_steps, seed = 2, 1, 1, 0
""",
            "call": "draw_gaussian_sketches(m_r, q, n_steps, seed)",
            "gold_call": "_oracle_draw_gaussian_sketches(m_r, q, n_steps, seed)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        draw_gaussian_sketches(3, 0, 1, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_sketches(3, 0, 1, 0)
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
def run_model():
    try:
        draw_gaussian_sketches(2, 1, 0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_sketches(2, 1, 0, 0)
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
