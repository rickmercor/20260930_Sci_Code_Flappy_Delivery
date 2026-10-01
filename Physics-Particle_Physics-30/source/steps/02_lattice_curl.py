"""
Compute the forward lattice curl of a real gauge-link array B with shape (3,Lx,Ly,Lz). Return the three independent plaquette components in order (01,02,12), using curl(B)_uv(x)=B_u(x)+B_v(x+u)-B_u(x+v)-B_v(x). All spatial shifts wrap periodically. Require finite inputs and each spatial side length at least 2; raise ValueError for invalid inputs. Do not reduce the resulting plaquette values modulo 2*pi.

Section III A, Eq. (41), defines the noncompact plaquette curl from gauge fields stored on links. Directions 0,1,2 correspond to the three spatial array axes. This curl is combined with the Dirac source in the later field equations through F=curl(B)-2*pi*Sigma. Periodic forward differences, component orientation, and retaining the noncompact values are essential for the lattice field equations and Hodge decomposition.

Returns
-------
A real NumPy array of shape (3,Lx,Ly,Lz), containing the noncompact plaquette curl components in order (01,02,12).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def lattice_curl(B: np.ndarray) -> np.ndarray:
    """Compute the periodic noncompact forward lattice curl.

    Parameters
    ----------
    B : numpy.ndarray
        Finite real gauge links of shape (3,Lx,Ly,Lz), with each
        spatial side at least 2. The input is not modified.

    Returns
    -------
    numpy.ndarray
        Real array with the same shape, plaquettes (01,02,12).
        Values use periodic forward differences and are not reduced
        modulo 2*pi.

    Raises
    ------
    ValueError
        If the numeric input has incorrect shape, a side below 2,
        or a nonfinite value.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_lattice_curl(B: np.ndarray) -> np.ndarray:
    """Compute the periodic noncompact curl in order (01,02,12)."""
    B = np.asarray(B, dtype=float)

    if (
        B.ndim != 4
        or B.shape[0] != 3
        or min(B.shape[1:]) < 2
        or not np.all(np.isfinite(B))
    ):
        raise ValueError(
            "B must be finite with shape (3,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    components = []

    for u, v in ((0, 1), (0, 2), (1, 2)):
        forward_u_Bv = np.roll(B[v], -1, axis=u)
        forward_v_Bu = np.roll(B[u], -1, axis=v)

        component = (
            B[u]
            + forward_u_Bv
            - forward_v_Bu
            - B[v]
        )

        components.append(component)

    return np.stack(components, axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'B = np.arange(3 * 4**3, dtype=float).reshape(3, 4, 4, 4) / 100.0',
      'call': 'lattice_curl(B)',
      'gold_call': '_oracle_lattice_curl(B)'},
     {'setup': 'B = np.sin(np.arange(3 * 4 * 6 * 4, dtype=float)).reshape(3, 4, 6, 4)',
      'call': 'lattice_curl(B)',
      'gold_call': '_oracle_lattice_curl(B)'},
     {'setup': 'B = np.zeros((3, 6, 6, 6), dtype=float)\nB[1, 2, 3, 4] = 0.7',
      'call': 'lattice_curl(B)',
      'gold_call': '_oracle_lattice_curl(B)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(lattice_curl, np.zeros((2,4,4,4), dtype=float))',
      'gold_call': '1.0',
      'tol': 0.0}]
