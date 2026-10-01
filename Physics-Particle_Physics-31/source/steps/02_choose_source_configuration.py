"""
Choose the source parameter and largest supported leading-principal rank probe.

The source pole/zero classification prescription applies for odd r. This task
uses the smallest positive odd admissible value, so r=1. For transformed
coefficients c_0,...,c_{m-1}, a leading-principal block C_r^(ell) requires
coefficients through c_{r+2(ell-1)}. Return r together with the largest ell
supported by the available transformed sequence.

Parameters
----------
c : np.ndarray
    One-dimensional finite real transformed-coefficient array containing at
    least c_0 and c_1.

Returns
-------
configuration : tuple[int, int]
    (r, probe_size), where r is the smallest positive odd admissible source
    parameter and probe_size is the largest supported leading-principal size.

Raises
------
ValueError
    If c is not a finite real one-dimensional array containing at least two
    transformed coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def choose_source_configuration(c: "np.ndarray") -> "tuple[int, int]":
    """Return the smallest positive odd r and largest supported probe size."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_choose_source_configuration(c: "np.ndarray") -> "tuple[int, int]":
    """Return the smallest positive odd r and largest supported probe size."""
    raw = np.asarray(c)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("c must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError("c must be a finite one-dimensional array of length at least two")

    r = 1
    max_index = coeff.size - 1
    probe_size = 1 + (max_index - r) // 2
    if probe_size < 1:
        raise ValueError("c is too short for the smallest positive odd source parameter")
    return int(r), int(probe_size)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431,-0.331672505490807096899101287418561408111207028195519199,-0.321904529323547068016655331121902730433961992623601257581247])",
            "call": "choose_source_configuration(c)",
            "gold_call": "_oracle_choose_source_configuration(c)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625])",
            "call": "choose_source_configuration(c)",
            "gold_call": "_oracle_choose_source_configuration(c)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.2,0.1])",
            "call": "choose_source_configuration(c)",
            "gold_call": "_oracle_choose_source_configuration(c)",
        },
    ]
