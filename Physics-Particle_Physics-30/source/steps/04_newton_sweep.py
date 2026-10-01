"""
Perform one damped checkerboard block-Newton sweep without modifying the input arrays.



The state channels are (B0,B1,B2,Re chi,Im chi). Visit even site parity (x+y+z)%2=0 first, then odd parity. Within each parity, update B0, B1, and B2 in that order, followed by a joint update of the two scalar components.



Recompute residuals and curvatures before each of these four stages. Within a stage, update all selected sites simultaneously. For a gauge channel, subtract damping*X/dX. For the scalar block, subtract damping*inverse(H)*Y.



Require compatible finite arrays, positive finite masses, even spatial side lengths of at least 2, and 0<damping<=1. Raise ValueError for invalid inputs or a nonpositive Newton curvature/block. Return a new state array.

Section III A, Eqs. (49)–(52), uses Newton updates for the gauge links and a coupled 2-by-2 Newton block for the real and imaginary monopole fields. The paper describes alternating odd and even lattice updates after Eq. (54).



For reproducibility, this task specifies even parity before odd parity, gauge directions 0,1,2 before the scalar block, and a default damping factor of 0.8. The off-diagonal scalar curvature must be retained. Residuals are recomputed between stages so that later stages use the updated fields.

Returns
-------
A new real NumPy array of shape (5,Lx,Ly,Lz), containing the state after one complete checkerboard block-Newton sweep. The input arrays remain unchanged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def newton_sweep(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    damping: float = 0.8,
) -> np.ndarray:
    """Perform one damped checkerboard block-Newton sweep.

    Parameters
    ----------
    state : numpy.ndarray
        Finite real state (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2.
        Spatial sides must be even and at least 2. Not modified.
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz) matching state; not modified.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.
    damping : float, default 0.8
        Finite correction multiplier with 0 < damping <= 1.

    Returns
    -------
    numpy.ndarray
        A new real array of the same shape as state after one sweep.
        Visit even then odd parity; at each parity update B0,B1,B2
        then the coupled c1,c2 block, recomputing between stages.

    Raises
    ------
    ValueError
        For incompatible/nonfinite numeric inputs, invalid side
        lengths, nonpositive masses, invalid damping, nonpositive
        selected gauge curvatures or a non-positive-definite scalar block.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_newton_sweep(
    state: np.ndarray,
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    damping: float = 0.8,
) -> np.ndarray:
    """Perform one specified checkerboard block-Newton sweep."""
    q = np.asarray(state, dtype=float)
    s = np.asarray(sigma, dtype=float)

    if (
        q.ndim != 4
        or q.shape[0] != 5
        or min(q.shape[1:]) < 2
    ):
        raise ValueError(
            "state must have shape (5,Lx,Ly,Lz), "
            "with each spatial side at least 2"
        )

    if s.shape != (3,) + q.shape[1:]:
        raise ValueError("sigma and state have incompatible shapes")

    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("state and sigma must contain finite values")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if any(length % 2 != 0 for length in q.shape[1:]):
        raise ValueError("checkerboard updating requires even side lengths")

    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")

    q = q.copy()
    parity = np.indices(q.shape[1:]).sum(axis=0) % 2

    for selected_parity in (0, 1):
        mask = parity == selected_parity

        for direction in range(3):
            values = _oracle_lattice_equations(
                q, s, mB, mchi
            )

            residual = values[direction]
            curvature = values[5 + direction]

            if np.any(curvature[mask] <= 0.0):
                raise ValueError("nonpositive gauge curvature")

            q[direction][mask] -= (
                damping
                * residual[mask]
                / curvature[mask]
            )

        values = _oracle_lattice_equations(
            q, s, mB, mchi
        )

        Y1 = values[3]
        Y2 = values[4]
        H11 = values[8]
        H12 = values[9]
        H22 = values[10]

        determinant = H11 * H22 - H12 ** 2

        if (
            np.any(H11[mask] <= 0.0)
            or np.any(determinant[mask] <= 0.0)
        ):
            raise ValueError(
                "scalar Newton block must be positive definite"
            )

        correction_real = (
            H22[mask] * Y1[mask]
            - H12[mask] * Y2[mask]
        ) / determinant[mask]

        correction_imag = (
            H11[mask] * Y2[mask]
            - H12[mask] * Y1[mask]
        ) / determinant[mask]

        q[3][mask] -= damping * correction_real
        q[4][mask] -= damping * correction_imag

    return q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 's = np.zeros((3, 4, 4, 4), dtype=float)\n'
               's[0, 0, 0, 1:2] = -1.0\n'
               'q = np.zeros((5, 4, 4, 4), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)',
      'call': 'newton_sweep(q, s, 0.5, 0.5, 0.8)',
      'gold_call': '_oracle_newton_sweep(q, s, 0.5, 0.5, 0.8)'},
     {'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\n'
               's[0, 0, 0, 1:3] = -1.0\n'
               'q = np.zeros((5, 6, 6, 6), dtype=float)\n'
               'q[3] = 1.0\n'
               'q += 0.03 * np.sin(np.arange(q.size)).reshape(q.shape)',
      'call': 'newton_sweep(q, s, 0.4, 0.7, 0.6)',
      'gold_call': '_oracle_newton_sweep(q, s, 0.4, 0.7, 0.6)'},
     {'setup': 's = np.zeros((3, 4, 4, 4), dtype=float)\n'
               's[0, 0, 0, 1:2] = -1.0\n'
               'q = np.zeros((5, 4, 4, 4), dtype=float)\n'
               'q[3] = 1.0',
      'call': 'newton_sweep(q, s, 0.6, 0.3, 0.9)',
      'gold_call': '_oracle_newton_sweep(q, s, 0.6, 0.3, 0.9)'},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n'
               'q=np.zeros((5,4,4,4), dtype=float)\n'
               'q[3]=1.0\n'
               's=np.zeros((3,4,4,4), dtype=float)\n',
      'call': 'raises_value_error(newton_sweep, q, s, 0.5, 0.5, 0.0)',
      'gold_call': '1.0',
      'tol': 0.0}]
