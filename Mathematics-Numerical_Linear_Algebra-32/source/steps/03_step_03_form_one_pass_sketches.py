"""
Unpack the test-matrix pack into Omega, Psi, and Phi. Form the rangefinder Y = A Omega, the corange sketch W = Psi A, and the wider amplifier Z = A Phi. Return those three sketches packed with the same 6-entry shape header.

All three sketches are linear images of A, so they can be accumulated from a single visit to the data. The short rangefinder is the object that will be amplified; the left sketch and wider right sketch are retained for the corange solve and the Gram step.

Returns
-------
1D ndarray: packed (Y, W, Z) with a 6-entry shape header
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def form_one_pass_sketches(A: np.ndarray, test_pack: np.ndarray) -> np.ndarray:
    """Return the packed sketches (Y, W, Z) = (A Omega, Psi A, A Phi).

    Raises
    ------
    ValueError
        If `test_pack` is too short or its packed shapes are not
        positive or do not match its own length header; if `A` is not
        2D or is empty; or if `Omega`, `Psi`, or `Phi` (unpacked from
        `test_pack`) have shapes incompatible with `A` (`Omega` must
        have `A`'s number of rows... i.e. `n` rows, `Psi` must have
        `A`'s number of columns i.e. `m` columns, and `Phi` must have
        `n` rows).
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


def _oracle_form_one_pass_sketches(A, test_pack):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    m, n = A.shape
    if m < 1 or n < 1:
        raise ValueError("A must be nonempty")
    Omega, Psi, Phi = _unpack_three(test_pack)
    if Omega.shape[0] != n:
        raise ValueError("Omega must have n rows")
    if Psi.shape[1] != m:
        raise ValueError("Psi must have m columns")
    if Phi.shape[0] != n:
        raise ValueError("Phi must have n rows")
    Y = A @ Omega
    W = Psi @ A
    Z = A @ Phi
    return _pack_three(Y, W, Z)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, d, l = 12, 10, 4, 7, 8
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
A = U @ np.diag(s) @ V.T
rng = np.random.default_rng(11)
Omega = rng.standard_normal((n, s_width))
Psi = rng.standard_normal((d, m))
Phi = rng.standard_normal((n, l))
header = np.array([Omega.shape[0], Omega.shape[1], Psi.shape[0], Psi.shape[1], Phi.shape[0], Phi.shape[1]], dtype=float)
test_pack = np.concatenate([header, Omega.ravel(), Psi.ravel(), Phi.ravel()])
""",
            "call": "form_one_pass_sketches(A, test_pack)",
            "gold_call": "_oracle_form_one_pass_sketches(A, test_pack)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]], dtype=float)
Omega = np.array([[1.0], [0.0]], dtype=float)
Psi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
Phi = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
header = np.array([2.0, 1.0, 2.0, 3.0, 2.0, 2.0])
test_pack = np.concatenate([header, Omega.ravel(), Psi.ravel(), Phi.ravel()])
""",
            "call": "form_one_pass_sketches(A, test_pack)",
            "gold_call": "_oracle_form_one_pass_sketches(A, test_pack)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]], dtype=float)
Omega = np.array([[1.0]], dtype=float)
Psi = np.array([[3.0]], dtype=float)
Phi = np.array([[4.0, 5.0]], dtype=float)
header = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
test_pack = np.concatenate([header, Omega.ravel(), Psi.ravel(), Phi.ravel()])
""",
            "call": "form_one_pass_sketches(A, test_pack)",
            "gold_call": "_oracle_form_one_pass_sketches(A, test_pack)",
        },
        {
            "setup": """import numpy as np
A = np.ones((2, 2))
header = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 2.0])
test_pack = np.concatenate([header, np.ones(3), np.ones(2), np.ones(4)])
def run_model():
    try:
        form_one_pass_sketches(A, test_pack)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_form_one_pass_sketches(A, test_pack)
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
A = np.array([1.0, 2.0])
test_pack = np.ones(10)
def run_model():
    try:
        form_one_pass_sketches(A, test_pack)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_form_one_pass_sketches(A, test_pack)
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
