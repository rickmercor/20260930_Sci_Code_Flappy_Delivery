"""
Step 5: fractional powers of the survival operator.

For a census taken M months after the birth pulse, the study splits each year's survival around the pulse. The population survives the first M months at one point of the annual cycle and the remaining 12 - M months at another. Each stretch is applied as a fractional power of the annual survival operator. The survival operator is diagonal, so its fractional powers are taken entry by entry. The survival of class a over M months is s_a raised to M/12.

This step returns the two diagonal operators, `$S^(M/12)$` and `$S^(1 - M/12)$`. The assembly step places them around the birth pulse and the ageing operator.

Returns
-------
S_a, S_b : tuple of np.ndarray, each shape (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fractional_survival_powers(
    survival: "numpy.typing.ArrayLike",
    months: float,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Fractional powers of the annual survival operator.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class (see step 01).
    months : float
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S_a, S_b)``: the diagonal operators ``S^(months/12)`` and
        ``S^(1 - months/12)``, whose product is the annual survival operator ``S``.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two probabilities in
        [0, 1], or if `months` is not finite or lies outside [0, 12].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fractional_survival_powers(
    survival: "numpy.typing.ArrayLike",
    months: float,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for fractional_survival_powers."""
    import numpy as np

    s = np.asarray(survival, dtype=float)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("survival must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(s)):
        raise ValueError("survival probabilities must be finite")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("survival probabilities must lie in [0, 1]")
    if not np.isfinite(months) or months < 0.0 or months > 12.0:
        raise ValueError("months must be finite and lie in [0, 12]")

    exponent = float(months) / 12.0
    S_a, S_b = np.diag(s ** exponent), np.diag(s ** (1.0 - exponent))
    return S_a, S_b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        fractional_survival_powers(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_fractional_survival_powers(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    base = "import numpy as np\ns = np.array([0.699, 0.774, 0.947, 0.947, 0.634])\n"
    return [
        # Normal: the graded census, 11 months after the pulse.
        {"setup": base,
         "call": "fractional_survival_powers(s, 11)",
         "gold_call": "_oracle_fractional_survival_powers(s, 11)"},
        # Normal: the paper's example configuration, 9 months after the pulse.
        {"setup": base,
         "call": "fractional_survival_powers(s, 9)",
         "gold_call": "_oracle_fractional_survival_powers(s, 9)"},
        # Boundary: M = 0 gives the identity and the full annual operator.
        {"setup": base,
         "call": "fractional_survival_powers(s, 0)",
         "gold_call": "_oracle_fractional_survival_powers(s, 0)"},
        # Boundary: M = 12 swaps the two roles.
        {"setup": base,
         "call": "fractional_survival_powers(s, 12)",
         "gold_call": "_oracle_fractional_survival_powers(s, 12)"},
        # Invalid: months beyond the annual cycle.
        {"setup": base + invalid,
         "call": "run_model(survival=s, months=13)",
         "gold_call": "run_gold(survival=s, months=13)"},
        # Invalid: negative months.
        {"setup": base + invalid,
         "call": "run_model(survival=s, months=-1)",
         "gold_call": "run_gold(survival=s, months=-1)"},
        # Invalid: non-finite months.
        {"setup": base + invalid,
         "call": "run_model(survival=s, months=float('nan'))",
         "gold_call": "run_gold(survival=s, months=float('nan'))"},
        # Invalid: survival outside [0, 1].
        {"setup": "import numpy as np\ns = np.array([0.5, 1.2])\n" + invalid,
         "call": "run_model(survival=s, months=11)",
         "gold_call": "run_gold(survival=s, months=11)"},
    ]
