"""
Construct the finite Dirac-string source on a periodic cubic lattice using zero-based coordinates 0,...,L-1 and independent plaquettes ordered as (01,02,12). Set Sigma_01(0,0,z)=-charge for z=1,...,R inclusive and set every other entry to zero. Require an even integer L>=4, an integer separation satisfying 1<=R<L/2, and a nonzero integer charge. Raise ValueError for invalid inputs.

Section III A, Eqs. (43), (53), and (54), represents a finite quark-antiquark pair by a connected stack of integer Dirac plaquettes. With Sigma_01(0,0,z)=-charge for z=1,...,R, the endpoints are at z=0 and z=R. The source enters the noncompact field strength as F=curl(B)-2*pi*Sigma. This construction specifies a finite-length source without imposing cylindrical radial boundary conditions.

Returns
-------
A float NumPy array of shape (3,L,L,L), containing the Dirac plaquette source in component order (01,02,12).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_finite_source(
    L: int,
    R: int,
    charge: int = 1,
) -> np.ndarray:
    """Construct the prescribed finite Dirac source.

    Parameters
    ----------
    L : int
        Even cubic side length, at least 4; booleans are excluded.
    R : int
        Separation with 1 <= R < L/2; booleans are excluded.
    charge : int, default 1
        Nonzero signed integer charge; booleans are excluded.

    Returns
    -------
    numpy.ndarray
        Real array of shape (3,L,L,L), plaquettes (01,02,12).
        Only sigma[0,0,0,1:R+1] is nonzero and equals -charge.

    Raises
    ------
    ValueError
        If L, R or charge violates the specified integer/range contract.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_finite_source(
    L: int,
    R: int,
    charge: int = 1,
) -> np.ndarray:
    """Construct the finite Dirac source in plaquette order (01,02,12)."""
    if (
        isinstance(L, (bool, np.bool_))
        or not isinstance(L, (int, np.integer))
        or L < 4
        or L % 2 != 0
    ):
        raise ValueError("L must be an even integer >= 4")

    if (
        isinstance(R, (bool, np.bool_))
        or not isinstance(R, (int, np.integer))
        or not 1 <= R < L // 2
    ):
        raise ValueError("R must be an integer satisfying 1 <= R < L/2")

    if (
        isinstance(charge, (bool, np.bool_))
        or not isinstance(charge, (int, np.integer))
        or charge == 0
    ):
        raise ValueError("charge must be a nonzero integer")

    sigma = np.zeros((3, L, L, L), dtype=float)
    sigma[0, 0, 0, 1:R + 1] = -charge

    return sigma

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'build_finite_source(6, 2, 1)',
      'gold_call': '_oracle_build_finite_source(6, 2, 1)'},
     {'setup': '',
      'call': 'build_finite_source(8, 3, -1)',
      'gold_call': '_oracle_build_finite_source(8, 3, -1)'},
     {'setup': '',
      'call': 'build_finite_source(10, 2, 2)',
      'gold_call': '_oracle_build_finite_source(10, 2, 2)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(build_finite_source, 5, 1, 1)',
      'gold_call': '1.0',
      'tol': 0.0}]
