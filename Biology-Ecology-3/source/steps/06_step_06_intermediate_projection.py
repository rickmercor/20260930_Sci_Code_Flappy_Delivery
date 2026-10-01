"""
Step 6: assemble the projection matrix for an intermediate census.

A census taken M months after the birth pulse sees animals between pulses. The study therefore splits the annual cycle around the pulse and composes the operators in that order:

1. survival over the elapsed M months, 

2. the birth pulse together with the transition to the next age class, 

3. survival over the remaining 12 - M months.

L = S^(M/12) (I + R) T S^(1 - M/12).

At M = 0 this collapses to the post-breeding matrix `$(I + R) U$`, because `$T S = U$`. The newborn class is observable at an intermediate census (the youngest animals are M months old), so the matrix keeps the life table's full class count. The pre-breeding matrix is not the M = 12 limit of this formula. This step returns L for a supplied census month.

Returns
-------
L : np.ndarray, shape (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intermediate_projection(
    survival: "numpy.typing.ArrayLike",
    fecundity: "numpy.typing.ArrayLike",
    months: float = 11,
) -> "numpy.ndarray":
    """Projection matrix for a census taken `months` after the birth pulse.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class (see step 01).
    fecundity : sequence of float
        Annual female fecundities in age order, one per class (see step 01).
    months : float, optional
        Months elapsed since the birth pulse at the census, from 0 to 12 inclusive.
        The task's graded census uses the default, 11 months.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` matrix ``S^(M/12) (I + R) T S^(1 - M/12)``: survival over the
        elapsed months, then the birth pulse and the transition, then survival over the
        remaining months. At M = 0 it equals the post-breeding matrix ``(I + R) U``.

    Raises
    ------
    ValueError
        If the two inputs do not share a length of at least two classes, or if either
        fails the validation of steps 02, 03 and 05.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_intermediate_projection(
    survival: "numpy.typing.ArrayLike",
    fecundity: "numpy.typing.ArrayLike",
    months: float = 11,
) -> "numpy.ndarray":
    """Reference implementation for intermediate_projection."""
    import numpy as np

    transition = _oracle_transition_matrix(survival)
    survival_operator, ageing = _oracle_survival_ageing_decomposition(transition)
    survival_elapsed, survival_remaining = _oracle_fractional_survival_powers(
        np.diag(survival_operator), months)
    pulse = _oracle_birth_pulse_operator(fecundity)
    if pulse.shape[0] != ageing.shape[0]:
        raise ValueError("survival and fecundity must cover the same age classes")
    L = survival_elapsed @ pulse @ ageing @ survival_remaining
    return L

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        intermediate_projection(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_intermediate_projection(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    small = ("import numpy as np\n"
             "s = np.array([0.5, 0.6, 0.7, 0.0])\n"
             "r = np.array([0.0, 1.5, 2.5, 2.5])\n")
    identity = small + (
        "T = np.zeros((4, 4))\n"
        "for a in range(3):\n"
        "    T[a + 1, a] = 1.0\n"
        "T[3, 3] = 1.0\n"
        "P = np.eye(4)\n"
        "P[0, :] += r\n"
        "POST = P @ (T @ np.diag(s))\n"
    )
    return [
        # Normal: a five-class table at the graded census month.
        {"setup": ("import numpy as np\n"
                   "s = np.array([0.699, 0.774, 0.947, 0.947, 0.634])\n"
                   "r = np.array([0.0, 0.0, 0.846, 0.734, 0.496])\n"),
         "call": "intermediate_projection(s, r, 11)",
         "gold_call": "_oracle_intermediate_projection(s, r, 11)"},
        # Normal: a nine-month census, the paper's example configuration.
        {"setup": ("import numpy as np\n"
                   "s = np.array([0.699, 0.774, 0.947, 0.947, 0.634])\n"
                   "r = np.array([0.0, 0.0, 0.846, 0.734, 0.496])\n"),
         "call": "intermediate_projection(s, r, 9)",
         "gold_call": "_oracle_intermediate_projection(s, r, 9)"},
        # Anchor: at M = 0 the matrix must equal the post-breeding (I + R) U.
        {"setup": identity,
         "call": "intermediate_projection(s, r, 0)",
         "gold_call": "POST"},
        # Boundary: the smallest usable age structure.
        {"setup": ("import numpy as np\n"
                   "s = np.array([0.5, 0.4])\n"
                   "r = np.array([0.0, 0.8])\n"),
         "call": "intermediate_projection(s, r, 11)",
         "gold_call": "_oracle_intermediate_projection(s, r, 11)"},
        # Invalid: survival and fecundity vectors of different lengths.
        {"setup": ("import numpy as np\n"
                   "s = np.array([0.5, 0.6, 0.7])\n"
                   "r = np.array([0.0, 1.5])\n" + invalid),
         "call": "run_model(survival=s, fecundity=r, months=11)",
         "gold_call": "run_gold(survival=s, fecundity=r, months=11)"},
        # Invalid: months beyond the annual cycle.
        {"setup": small + invalid,
         "call": "run_model(survival=s, fecundity=r, months=13)",
         "gold_call": "run_gold(survival=s, fecundity=r, months=13)"},
    ]
