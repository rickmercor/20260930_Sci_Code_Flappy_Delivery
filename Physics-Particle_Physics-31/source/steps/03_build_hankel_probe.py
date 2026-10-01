"""
Build a leading principal Hankel matrix from the transformed EFT sequence.

For transformed coefficients c_k, the source construction uses
[C_r^(ell)]_{ij} = c_{r+i+j}, i,j=0,...,ell-1. This step returns the requested
leading principal block and is used both for the numerical-rank probe and for
the shifted generalized spectral pencil.

Parameters
----------
c : np.ndarray
    One-dimensional finite real transformed-coefficient array.
r : int
    Nonnegative shift.
size : int
    Positive matrix size. The input must contain all coefficients through
    c_{r+2(size-1)}.

Returns
-------
hankel : np.ndarray
    Real symmetric array with shape (size, size).

Raises
------
ValueError
    If c, r, or size is invalid or if c is too short for the requested block.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_hankel_probe(c: "np.ndarray", r: int, size: int) -> "np.ndarray":
    """Build C_r^(size) with entries c[r+i+j]."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_hankel_probe(c: "np.ndarray", r: int, size: int) -> "np.ndarray":
    """Build C_r^(size) with entries c[r+i+j]."""
    raw = np.asarray(c)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("c must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("c must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size == 0 or not np.all(np.isfinite(coeff)):
        raise ValueError("c must be a nonempty finite one-dimensional array")
    if not isinstance(r, (int, np.integer)) or int(r) < 0:
        raise ValueError("r must be a nonnegative integer")
    if not isinstance(size, (int, np.integer)) or int(size) < 1:
        raise ValueError("size must be a positive integer")
    r = int(r)
    size = int(size)
    required = r + 2 * (size - 1)
    if required >= coeff.size:
        raise ValueError("c is too short for the requested Hankel block")

    idx = r + np.add.outer(np.arange(size), np.arange(size))
    return coeff[idx]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431,-0.331672505490807096899101287418561408111207028195519199,-0.321904529323547068016655331121902730433961992623601257581247])",
            "call": "np.round(build_hankel_probe(c,1,5),12)",
            "gold_call": "np.round(_oracle_build_hankel_probe(c,1,5),12)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.5,0.25,0.125,0.0625,0.03125])",
            "call": "np.round(build_hankel_probe(c,0,1),12)",
            "gold_call": "np.round(_oracle_build_hankel_probe(c,0,1),12)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.1,0.12,0.10825,0.087,0.065700625,0.04773825,0.03379793140625])",
            "call": "np.round(build_hankel_probe(c,1,3),12)",
            "gold_call": "np.round(_oracle_build_hankel_probe(c,1,3),12)",
        },
    ]
