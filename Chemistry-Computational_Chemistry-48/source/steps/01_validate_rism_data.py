"""
Validate supplied charged and zero-charge 3D-RISM correlation data.

The validation stage checks that the supplied total and direct correlation

fields are numerically finite and mutually compatible, and that their

dimensions follow the required representation

(n_conformers, n_sites, n_x, n_y, n_z).




The charged and zero-charge states must contain the same number of

conformers, solvent sites, and spatial grid points. The number of conformer

labels must agree with the conformer dimension.




This stage performs validation only. It does not perform a new 3D-RISM

calculation, solve a closure relation, or modify the supplied correlation

fields.




Conformer labels are identifiers and are not required to be unique.

Returns
-------
return None
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def validate_rism_data(
    charged_h: np.ndarray,
    charged_c: np.ndarray,
    zero_h: np.ndarray,
    zero_c: np.ndarray,
    conformer_labels: np.ndarray,
) -> None:
    """
    Validate supplied charged and zero-charge 3D-RISM correlation data.

    Parameters
    ----------
    charged_h : np.ndarray
        Charged-state total correlation fields with shape
        (n_conformers, n_sites, nx, ny, nz).
    charged_c : np.ndarray
        Charged-state direct correlation fields with the same shape
        as charged_h.
    zero_h : np.ndarray
        Zero-charge total correlation fields with the same shape
        as charged_h.
    zero_c : np.ndarray
        Zero-charge direct correlation fields with the same shape
        as charged_c.
    conformer_labels : np.ndarray
        Labels identifying the supplied conformers.

    Returns
    -------
    None
        Returns None when the supplied data are valid.

    Raises
    ------
    ValueError
        If dimensions, conformer counts, or numerical values are invalid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_validate_rism_data(
    charged_h,
    charged_c,
    zero_h,
    zero_c,
    conformer_labels,
):
    arrays = {
        "charged_h": charged_h,
        "charged_c": charged_c,
        "zero_h": zero_h,
        "zero_c": zero_c,
    }

    for name, arr in arrays.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array")

        if arr.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "(n_conformers, n_sites, nx, ny, nz)"
            )

        if any(dim <= 0 for dim in arr.shape):
            raise ValueError(f"{name} contains an empty dimension")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    if charged_h.shape != charged_c.shape:
        raise ValueError(
            "charged_h and charged_c must have identical shapes"
        )

    if zero_h.shape != zero_c.shape:
        raise ValueError(
            "zero_h and zero_c must have identical shapes"
        )

    if charged_h.shape != zero_h.shape:
        raise ValueError(
            "charged and zero-charge fields must have identical shapes"
        )

    labels = np.asarray(conformer_labels)

    if labels.ndim != 1:
        raise ValueError("conformer_labels must be one-dimensional")

    if len(labels) != charged_h.shape[0]:
        raise ValueError(
            "number of conformer labels must match number of conformers"
        )

    if len(labels) == 0:
        raise ValueError("at least one conformer is required")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications."""
    return [
        {
            "setup": """import numpy as np

# Test case 1: Valid 5D charged and zero-charge data
shape = (2, 2, 3, 3, 3)

charged_h = np.random.default_rng(1).normal(size=shape)
charged_c = np.random.default_rng(2).normal(size=shape)
zero_h = np.random.default_rng(3).normal(size=shape)
zero_c = np.random.default_rng(4).normal(size=shape)

conformer_labels = np.array(["conf_1", "conf_2"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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

# Test case 2: Invalid - charged h/c shape mismatch
charged_h = np.ones((2, 2, 3, 3, 3))
charged_c = np.ones((2, 2, 3, 3, 4))
zero_h = np.ones((2, 2, 3, 3, 3))
zero_c = np.ones((2, 2, 3, 3, 3))
conformer_labels = np.array(["a", "b"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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

# Test case 3: Invalid - non-finite correlation value
shape = (2, 2, 3, 3, 3)

charged_h = np.ones(shape)
charged_c = np.ones(shape)
zero_h = np.ones(shape)
zero_c = np.ones(shape)

charged_h[0, 0, 0, 0, 0] = np.nan

conformer_labels = np.array(["a", "b"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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

# Test case 4: Invalid - wrong dimensionality
shape = (2, 2, 3, 3)

charged_h = np.ones(shape)
charged_c = np.ones(shape)
zero_h = np.ones(shape)
zero_c = np.ones(shape)

conformer_labels = np.array(["a", "b"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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

# Test case 5: Invalid - label count mismatch
shape = (2, 1, 2, 2, 2)

charged_h = np.ones(shape)
charged_c = np.ones(shape)
zero_h = np.ones(shape)
zero_c = np.ones(shape)

conformer_labels = np.array(["only_one"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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

# Test case 6: Repeated labels are valid identifiers
shape = (2, 1, 2, 2, 2)

charged_h = np.ones(shape)
charged_c = np.ones(shape)
zero_h = np.ones(shape)
zero_c = np.ones(shape)

conformer_labels = np.array(["same", "same"])

def run_model():
    try:
        validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_validate_rism_data(
            charged_h, charged_c,
            zero_h, zero_c,
            conformer_labels
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
    ]
