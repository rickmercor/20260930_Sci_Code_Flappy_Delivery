"""
Sample a deterministic orthonormal basis for the prescribed complex invariant subspace of the polar factor, together with the conditioning of that sample.

The step multiplies a fixed complex map of $P$ by one real Gaussian sample of shape $(2n,n)$ and orthonormalizes the product by reduced QR. Reduced QR leaves every column free up to a unit-modulus factor, so a fixed generator and a fixed diagonal-phase convention are what make the returned basis reproducible. The moduli of the QR diagonal measure how far the sampled range sits from rank deficiency, so they are returned with the basis and later score one sample against another.

Returns
-------
one complex (2n + 1, n) array with the phase-fixed basis in rows :2n and the QR diagonal moduli in row 2n; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_positive_imaginary_basis(P: np.ndarray, seed: int) -> np.ndarray:
    """Return a canonical thin-QR basis of ``range(P + i I)``.

    Raises ``ValueError`` unless every one of the following holds: ``P`` is real
    rather than complex; ``P`` is a two-dimensional square array of even order at
    least two; every entry of ``P`` is finite; ``P`` is skew-symmetric to an
    absolute tolerance of ``1e-10``; ``P`` is orthogonal to the same tolerance, so
    a skew-symmetric factor that is not orthogonal is rejected rather than
    sampled; ``seed`` is an integer, not a bool; and no modulus of the reduced-QR
    diagonal falls to ``100`` machine epsilons, which would leave the sampled
    range numerically rank deficient.

    Parameters
    ----------
    P : np.ndarray
        Finite real skew-symmetric orthogonal array of shape ``(2n, 2n)``.
    seed : int
        Seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(2n + 1, n)``.  Rows ``:2n`` hold the orthonormal
        reduced-QR factor of ``(P + 1j * I) @ G``, where ``G`` is
        ``numpy.random.default_rng(seed).standard_normal((2n, n))``, with each
        column multiplied by the phase of the matching diagonal entry of ``R`` so
        that every ``R[j, j]`` becomes positive real.  Row ``2n`` holds the moduli
        ``abs(R[j, j])`` as real values in a complex row.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sample_positive_imaginary_basis(P: np.ndarray, seed: int) -> np.ndarray:
    """Reference seeded invariant-subspace basis with its QR diagonal moduli."""
    raw = np.asarray(P)
    if np.iscomplexobj(raw):
        raise ValueError("P must be real")
    polar = np.asarray(raw, dtype=float)
    if (
        polar.ndim != 2
        or polar.shape[0] != polar.shape[1]
        or polar.shape[0] < 2
        or polar.shape[0] % 2
    ):
        raise ValueError("P must be an even-order square matrix")
    if not np.all(np.isfinite(polar)):
        raise ValueError("P must contain only finite entries")
    identity = np.eye(polar.shape[0])
    if not np.allclose(polar + polar.T, 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("P must be skew-symmetric")
    if not np.allclose(polar.T @ polar, identity, atol=1e-10, rtol=0.0):
        raise ValueError("P must be orthogonal")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    n = polar.shape[0] // 2
    sample = np.random.default_rng(int(seed)).standard_normal((2 * n, n))
    basis, triangular = np.linalg.qr(
        (polar + 1j * identity) @ sample,
        mode="reduced",
    )
    diagonal = np.diag(triangular)
    moduli = np.abs(diagonal)
    if np.any(moduli <= 100.0 * np.finfo(float).eps):
        raise ValueError("the sampled range is numerically rank deficient")
    phases = diagonal / moduli
    return np.vstack((basis * phases, moduli.astype(complex)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return two dimensions, a changed seed, and an invalid factor."""
    return [
        {
            "setup": """import numpy as np
P = np.array([[0.0, -1.0], [1.0, 0.0]])
seed = 7
""",
            "call": "sample_positive_imaginary_basis(P, seed)",
            "gold_call": "_oracle_sample_positive_imaginary_basis(P, seed)",
        },
        {
            "setup": """import numpy as np
I = np.eye(3)
P = np.block([[np.zeros((3,3)), -I], [I, np.zeros((3,3))]])
seed = 260812153
""",
            "call": "sample_positive_imaginary_basis(P, seed)",
            "gold_call": "_oracle_sample_positive_imaginary_basis(P, seed)",
        },
        {
            "setup": """import numpy as np
I = np.eye(2)
P = np.block([[np.zeros((2,2)), -I], [I, np.zeros((2,2))]])
seed = 314159
""",
            "call": "sample_positive_imaginary_basis(P, seed)",
            "gold_call": "_oracle_sample_positive_imaginary_basis(P, seed)",
        },
        {
            "setup": """import numpy as np
P = np.zeros((3, 3))
def run_model():
    try:
        sample_positive_imaginary_basis(P, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, 3)
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
P = np.array([[0.0, -1.0], [1.0, np.nan]])
def run_model():
    try:
        sample_positive_imaginary_basis(P, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, 3)
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
P = np.array([[0.0, -1.0], [1.0, 0.0]])
def run_model():
    try:
        sample_positive_imaginary_basis(P, True)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, True)
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
P = np.eye(2)
def run_model():
    try:
        sample_positive_imaginary_basis(P, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, 3)
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
P = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
def run_model():
    try:
        sample_positive_imaginary_basis(P, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, 3)
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
P = np.array([[0.0, -2.0], [2.0, 0.0]])
def run_model():
    try:
        sample_positive_imaginary_basis(P, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sample_positive_imaginary_basis(P, 1)
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
