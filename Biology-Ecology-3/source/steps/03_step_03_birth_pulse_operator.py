"""
Step 3: build the birth-pulse operator.

Reproduction is a short annual pulse. It adds each female's newborns to the population vector and leaves the existing individuals in place. The records store the raw fecundities (average female offspring per female, before any survival adjustment) in a matrix whose only non-zero row is the first one. The pulse operator is the identity plus that matrix. Entry ``[0, a]`` holds the fecundity of class `$a$`, every diagonal entry is one, and everything else is zero.

Returns
-------
np.eye(n, dtype=float) + raw : np.ndarray, shape (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def birth_pulse_operator(
    fecundity: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Build the birth-pulse operator ``(I + R)``.

    Parameters
    ----------
    fecundity : sequence of float
        Annual female fecundities in age order, one per class, as estimated by step 01.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` birth-pulse matrix: the identity plus the raw fecundity matrix,
        i.e. ones on the diagonal and the fecundities in the first row.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two non-negative
        fecundities.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_birth_pulse_operator(
    fecundity: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for birth_pulse_operator."""
    import numpy as np

    r = np.asarray(fecundity, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("fecundity must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(r)):
        raise ValueError("fecundities must be finite")
    if np.any(r < 0.0):
        raise ValueError("fecundities must be non-negative")

    n = r.size
    raw = np.zeros((n, n), dtype=float)
    raw[0, :] = r
    return np.eye(n, dtype=float) + raw

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        birth_pulse_operator(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_birth_pulse_operator(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a four-class vector with non-breeding young classes.
        {"setup": "import numpy as np\nr = np.array([0.0, 0.0, 1.5, 2.5])\n",
         "call": "birth_pulse_operator(r)",
         "gold_call": "_oracle_birth_pulse_operator(r)"},
        # Normal: a hump-shaped pattern with a smaller terminal fecundity.
        {"setup": "import numpy as np\nr = np.array([0.0, 0.0, 0.42, 0.81, 0.65, 0.43])\n",
         "call": "birth_pulse_operator(r)",
         "gold_call": "_oracle_birth_pulse_operator(r)"},
        # Boundary: all fecundities zero leaves the identity.
        {"setup": "import numpy as np\nr = np.zeros(3)\n",
         "call": "birth_pulse_operator(r)",
         "gold_call": "_oracle_birth_pulse_operator(r)"},
        # Invalid: a negative fecundity.
        {"setup": "import numpy as np\nr = np.array([0.0, -0.5, 1.0])\n" + invalid,
         "call": "run_model(fecundity=r)",
         "gold_call": "run_gold(fecundity=r)"},
        # Invalid: a scalar input.
        {"setup": "import numpy as np\nr = 1.0\n" + invalid,
         "call": "run_model(fecundity=r)",
         "gold_call": "run_gold(fecundity=r)"},
    ]
