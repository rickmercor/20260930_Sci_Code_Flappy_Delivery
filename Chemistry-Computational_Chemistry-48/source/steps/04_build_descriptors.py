"""
Build the MILC descriptor matrices from charged-state and zero-charge

solvation quantities and partial molar volumes.

The two states remain independent: zero-charge descriptors come from the

zero-charge 3D-RISM evaluation, not from copying or charge-wiping the

charged state. Recover the paper's descriptor sets and column order from

the source.

Returns
-------
return full_descriptors, sub_descriptors, hnc_descriptors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_descriptors(
    charged_terms: np.ndarray,
    zero_terms: np.ndarray,
    pmv_charged: np.ndarray,
    pmv_zero: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the paper's MILC descriptor matrices.

    Parameters
    ----------
    charged_terms : np.ndarray
        Charged-state solvation terms, shape (n_conformers, 3).
    zero_terms : np.ndarray
        Zero-charge solvation terms, shape (n_conformers, 3).
    pmv_charged : np.ndarray
        Charged-state partial molar volumes, one value per conformer.
    pmv_zero : np.ndarray
        Zero-charge partial molar volumes, one value per conformer.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The paper's Full, Sub, and HNC descriptor matrices.

    Raises
    ------
    ValueError
        If charged and zero-charge inputs have mismatched conformer
        counts, or if numerical values are invalid.
    """
    return full_descriptors, sub_descriptors, hnc_descriptors

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_descriptors(
    charged_terms,
    zero_terms,
    pmv_charged,
    pmv_zero,
):
    charged_terms = np.asarray(charged_terms, dtype=float)
    zero_terms = np.asarray(zero_terms, dtype=float)
    v = np.asarray(pmv_charged, dtype=float).reshape(-1)
    v0 = np.asarray(pmv_zero, dtype=float).reshape(-1)

    if charged_terms.ndim != 2 or charged_terms.shape[1] != 3:
        raise ValueError("charged_terms must have shape (n_conformers, 3)")
    if zero_terms.ndim != 2 or zero_terms.shape[1] != 3:
        raise ValueError("zero_terms must have shape (n_conformers, 3)")
    if charged_terms.shape != zero_terms.shape:
        raise ValueError("charged and zero-charge terms must have the same shape")
    n = charged_terms.shape[0]
    if v.shape != (n,) or v0.shape != (n,):
        raise ValueError("PMV arrays must have one entry per conformer")
    for name, arr in (
        ("charged_terms", charged_terms),
        ("zero_terms", zero_terms),
        ("pmv_charged", v),
        ("pmv_zero", v0),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    sc_hnc, sc_kh, gf = charged_terms.T
    sc_hnc0, sc_kh0, gf0 = zero_terms.T

    full = np.column_stack([sc_hnc, sc_kh, gf, v, sc_hnc0, sc_kh0, gf0, v0])
    sub = np.column_stack([sc_kh, gf, sc_kh0, gf0, v0])
    hnc = np.column_stack([sc_hnc, sc_hnc0, v0])
    return full, sub, hnc

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Test case 1: Distinct charged and zero-charge states
charged_terms = np.array([
    [1.0, 0.5, 0.2],
    [2.0, 1.5, 0.8],
])
zero_terms = np.array([
    [0.3, 0.1, -0.2],
    [0.4, 0.2, -0.1],
])
pmv_charged = np.array([10.0, 12.0])
pmv_zero = np.array([9.0, 11.0])

def run_model():
    return [
        np.asarray(x).tolist()
        for x in build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]

def run_gold():
    return [
        np.asarray(x).tolist()
        for x in _oracle_build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 2: Single conformer
charged_terms = np.array([[0.7, 0.4, 0.1]])
zero_terms = np.array([[0.2, 0.05, -0.3]])
pmv_charged = np.array([8.5])
pmv_zero = np.array([7.2])

def run_model():
    return [
        np.asarray(x).tolist()
        for x in build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]

def run_gold():
    return [
        np.asarray(x).tolist()
        for x in _oracle_build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Test case 3: Invalid - mismatched conformer counts
charged_terms = np.array([[1.0, 0.5, 0.2], [2.0, 1.5, 0.8]])
zero_terms = np.array([[0.3, 0.1, -0.2]])
pmv_charged = np.array([10.0, 12.0])
pmv_zero = np.array([9.0])

def run_model():
    try:
        build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Single-conformer PMV as Python floats; gold reshapes to length 1
charged_terms = np.array([[0.7, 0.4, 0.1]])
zero_terms = np.array([[0.2, 0.05, -0.3]])
pmv_charged = 8.5
pmv_zero = 7.2

def run_model():
    return [
        np.asarray(x).tolist()
        for x in build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]

def run_gold():
    return [
        np.asarray(x).tolist()
        for x in _oracle_build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

# Column PMV (n, 1): one value per conformer, not a 1-D vector
charged_terms = np.array([
    [1.0, 0.5, 0.2],
    [2.0, 1.5, 0.8],
])
zero_terms = np.array([
    [0.3, 0.1, -0.2],
    [0.4, 0.2, -0.1],
])
pmv_charged = np.array([[10.0], [12.0]])
pmv_zero = np.array([[9.0], [11.0]])

def run_model():
    return [
        np.asarray(x).tolist()
        for x in build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]

def run_gold():
    return [
        np.asarray(x).tolist()
        for x in _oracle_build_descriptors(
            charged_terms, zero_terms, pmv_charged, pmv_zero
        )
    ]
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
