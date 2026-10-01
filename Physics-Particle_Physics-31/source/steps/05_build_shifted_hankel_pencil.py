"""
Build the consecutive Hankel pair used in the source generalized spectral problem.

For a finite spectrum of size d, construct C_r^(d) and C_{r+1}^(d) from the same
transformed sequence. The generalized eigenvalue problem is formed from this ordered pair.

Parameters
----------
c : np.ndarray
    One-dimensional finite real transformed-coefficient array.
r : int
    Nonnegative source shift.
d : int
    Positive finite spectrum size.

Returns
-------
pencil : tuple[np.ndarray, np.ndarray]
    The ordered pair (C_r^(d), C_{r+1}^(d)).

Raises
------
ValueError
    Under any invalid-input condition declared by build_hankel_probe, or if d
    is not a positive integer.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_shifted_hankel_pencil(
    c: "np.ndarray", r: int, d: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the ordered pair (C_r^(d), C_{r+1}^(d))."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_shifted_hankel_pencil(
    c: "np.ndarray", r: int, d: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Return the ordered pair (C_r^(d), C_{r+1}^(d))."""
    if not isinstance(d, (int, np.integer)) or int(d) < 1:
        raise ValueError("d must be a positive integer")
    d = int(d)
    current = _oracle_build_hankel_probe(c, r, d)
    shifted = _oracle_build_hankel_probe(c, r + 1, d)
    return current, shifted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    pack = "lambda out: (np.round(out[0],12),np.round(out[1],12))"
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431,-0.331672505490807096899101287418561408111207028195519199,-0.321904529323547068016655331121902730433961992623601257581247])\npack=" + pack,
            "call": "pack(build_shifted_hankel_pencil(c,1,4))",
            "gold_call": "pack(_oracle_build_shifted_hankel_pencil(c,1,4))",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.5,0.25,0.125,0.0625])\npack=" + pack,
            "call": "pack(build_shifted_hankel_pencil(c,1,1))",
            "gold_call": "pack(_oracle_build_shifted_hankel_pencil(c,1,1))",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.5,0.5,0.40625,0.3125,0.236328125,0.177734375])\npack=" + pack,
            "call": "pack(build_shifted_hankel_pencil(c,1,2))",
            "gold_call": "pack(_oracle_build_shifted_hankel_pencil(c,1,2))",
        },
    ]
