"""
Draw three independent standard_normal test matrices from default_rng(seed), in order Omega of shape (n, s_width), Psi of shape (d, m), and Phi of shape (n, l). Require min(l, d) > s_width >= 1 and s_width <= min(m, n). Pack them as one vector: six shape integers followed by the three matrices in row-major order.

A one-pass reconstruction needs a short right sketch for the rangefinder, a left sketch for the corange, and a wider right sketch that can stand in for A. Drawing them from one seeded generator in a fixed order makes the later sketches deterministic.

Returns
-------
1D ndarray: packed (Omega, Psi, Phi) with a 6-entry shape header
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def draw_gaussian_test_matrices(
    m: int, n: int, s_width: int, d: int, l: int, seed: int
) -> np.ndarray:
    """Draw Omega (n, s_width), Psi (d, m), Phi (n, l) and pack them.

    Parameters
    ----------
    m, n : int
        Dimensions of A.
    s_width : int
        Rangefinder sketch width.
    d : int
        Corange sketch height, must exceed s_width.
    l : int
        Amplifier sketch width, must exceed s_width.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    pack : np.ndarray
        One-dimensional packing of (Omega, Psi, Phi).

    Raises
    ------
    ValueError
        If `m`, `n`, `s_width`, `d`, or `l` is not an integer; if
        `m >= 1` and `n >= 1` do not both hold; if `s_width >= 1` does
        not hold; if `min(l, d) > s_width` does not hold; or if
        `s_width <= min(m, n)` does not hold.
    """
    return np.zeros(6)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _pack_three(A, B, C):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    header = np.array(
        [A.shape[0], A.shape[1], B.shape[0], B.shape[1], C.shape[0], C.shape[1]],
        dtype=float,
    )
    return np.concatenate([header, A.ravel(), B.ravel(), C.ravel()])


def _unpack_three(pack):
    pack = np.asarray(pack, dtype=float).reshape(-1)
    if pack.size < 6:
        raise ValueError("pack is too short")
    mA, nA, mB, nB, mC, nC = [int(round(float(x))) for x in pack[:6]]
    if min(mA, nA, mB, nB, mC, nC) < 1:
        raise ValueError("packed shapes must be positive")
    need = 6 + mA * nA + mB * nB + mC * nC
    if pack.size != need:
        raise ValueError("pack length does not match header")
    i = 6
    A = pack[i : i + mA * nA].reshape(mA, nA)
    i += mA * nA
    B = pack[i : i + mB * nB].reshape(mB, nB)
    i += mB * nB
    C = pack[i:].reshape(mC, nC)
    return A, B, C


def _oracle_draw_gaussian_test_matrices(m, n, s_width, d, l, seed):
    ints = (m, n, s_width, d, l)
    if not all(isinstance(v, (int, np.integer)) for v in ints):
        raise ValueError("m, n, s_width, d, and l must be integers")
    m, n, s_width, d, l = (int(v) for v in ints)
    if m < 1 or n < 1:
        raise ValueError("require m >= 1 and n >= 1")
    if s_width < 1:
        raise ValueError("require s_width >= 1")
    if not (l > s_width and d > s_width):
        raise ValueError("require min(l, d) > s_width")
    if s_width > n or s_width > m:
        raise ValueError("require s_width <= min(m, n)")
    rng = np.random.default_rng(int(seed))
    Omega = rng.standard_normal((n, s_width))
    Psi = rng.standard_normal((d, m))
    Phi = rng.standard_normal((n, l))
    return _pack_three(Omega, Psi, Phi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, d, l, seed = 12, 10, 4, 7, 8, 11
""",
            "call": "draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
            "gold_call": "_oracle_draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
        },
        {
            "setup": """import numpy as np
m, n, s_width, d, l, seed = 6, 5, 2, 4, 3, 1
""",
            "call": "draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
            "gold_call": "_oracle_draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
        },
        {
            "setup": """import numpy as np
m, n, s_width, d, l, seed = 2, 2, 1, 2, 2, 3
""",
            "call": "draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
            "gold_call": "_oracle_draw_gaussian_test_matrices(m, n, s_width, d, l, seed)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        draw_gaussian_test_matrices(4, 4, 2, 2, 3, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_test_matrices(4, 4, 2, 2, 3, 0)
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
        draw_gaussian_test_matrices(3, 3, 0, 2, 2, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_draw_gaussian_test_matrices(3, 3, 0, 2, 2, 0)
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
