"""
Solve the three-dimensional finite-source DGL field equations using the specified checkerboard Newton sweep.



Initialize B0=B1=B2=0, Re chi=1, and Im chi=0 everywhere. Before each sweep, check the maximum absolute value over the five normalized residual channels (X0,X1,X2,Y1,Y2). Return the state when this maximum is strictly below tolerance.



Use damping=0.8, tolerance=1e-9, and max_sweeps=20000 by default. Check convergence again after the last permitted sweep. Raise ValueError if the solver does not converge.



Require a finite source array of shape (3,Lx,Ly,Lz), even spatial side lengths of at least 4, positive finite masses and tolerance, a positive integer max_sweeps, and 0<damping<=1. Do not modify the source array or use precomputed fields.

Section III A determines finite-length flux-tube configurations by solving the coupled gauge and complex monopole field equations. The trivial initial condition is B_mu=0, Re chi=1, and Im chi=0.



Convergence must be established from the normalized residuals of Eqs. (45)–(48), not solely from an action plateau or iteration count. Each iteration uses the earlier checkerboard block-Newton component, which in turn uses the earlier residual, curvature, and lattice-curl components.



The returned state supplies the fields needed for the subsequent Hodge energy decomposition. No particular sweep count is graded.

Returns
-------
A converged real NumPy array of shape (5,Lx,Ly,Lz), with channels (B0,B1,B2,Re chi,Im chi). Every normalized field-equation residual has absolute value below tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_finite_tube(
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 20000,
    damping: float = 0.8,
) -> np.ndarray:
    """Solve the finite-source stationary DGL equations.

    Parameters
    ----------
    sigma : numpy.ndarray
        Finite real source (3,Lx,Ly,Lz), plaquettes (01,02,12).
        Spatial sides must be even and at least 4. Not modified.
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass.
    tolerance : float, default 1e-9
        Positive finite threshold for all five normalized residuals.
    max_sweeps : int, default 20000
        Positive integer sweep budget; booleans are excluded.
    damping : float, default 0.8
        Finite correction multiplier with 0 < damping <= 1.

    Returns
    -------
    numpy.ndarray
        Real state (5,Lx,Ly,Lz), channels B0,B1,B2,c1,c2, obtained
        from B=0,c1=1,c2=0. The largest normalized residual magnitude
        is strictly below tolerance. Convergence is checked before
        each sweep and after the last allowed sweep.

    Raises
    ------
    ValueError
        For invalid numeric input shapes, side lengths, masses,
        tolerance, sweep budget or damping; for an invalid Newton
        curvature/block; or if convergence is not reached.
    """
    return np.empty(0, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_finite_tube(
    sigma: np.ndarray,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
    max_sweeps: int = 20000,
    damping: float = 0.8,
) -> np.ndarray:
    """Return the converged finite-source DGL state."""
    s = np.asarray(sigma, dtype=float)

    if s.ndim != 4 or s.shape[0] != 3:
        raise ValueError(
            "sigma must have shape (3,Lx,Ly,Lz)"
        )

    if not np.all(np.isfinite(s)):
        raise ValueError("sigma must contain finite values")

    if any(
        length < 4 or length % 2 != 0
        for length in s.shape[1:]
    ):
        raise ValueError(
            "spatial side lengths must be even and at least 4"
        )

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    if (
        isinstance(max_sweeps, (bool, np.bool_))
        or not isinstance(max_sweeps, (int, np.integer))
        or max_sweeps < 1
    ):
        raise ValueError("max_sweeps must be a positive integer")

    if not np.isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0,1]")

    state = np.zeros((5,) + s.shape[1:], dtype=float)
    state[3] = 1.0

    for iteration in range(max_sweeps + 1):
        values = _oracle_lattice_equations(
            state,
            s,
            mB,
            mchi,
        )

        max_residual = float(
            np.max(np.abs(values[:5]))
        )

        if max_residual < tolerance:
            return state

        if iteration < max_sweeps:
            state = _oracle_newton_sweep(
                state,
                s,
                mB,
                mchi,
                damping,
            )

    raise ValueError(
        "solver did not converge within max_sweeps"
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\ns[0, 0, 0, 1:3] = -1.0',
      'call': 'solve_finite_tube(s, 0.5, 0.5)',
      'gold_call': '_oracle_solve_finite_tube(s, 0.5, 0.5)',
      'tol': 1e-07},
     {'setup': 's = np.zeros((3, 8, 8, 8), dtype=float)\ns[0, 0, 0, 1:4] = 1.0',
      'call': 'solve_finite_tube(s, 0.4, 0.6)',
      'gold_call': '_oracle_solve_finite_tube(s, 0.4, 0.6)',
      'tol': 1e-07},
     {'setup': 's = np.zeros((3, 6, 6, 6), dtype=float)\ns[0, 0, 0, 1:2] = -1.0',
      'call': 'solve_finite_tube(s, 0.6, 0.4)',
      'gold_call': '_oracle_solve_finite_tube(s, 0.6, 0.4)',
      'tol': 1e-07},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n'
               'q=np.zeros((5,4,4,4), dtype=float)\n'
               'q[3]=1.0\n'
               's=np.zeros((3,4,4,4), dtype=float)\n',
      'call': 'raises_value_error(solve_finite_tube, s, tolerance=0.0)',
      'gold_call': '1.0',
      'tol': 0.0}]
