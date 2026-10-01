"""
Compute the finite-length solenoidal-potential curvature on a periodic cubic lattice.



Construct unit-charge Dirac sources at separations R-halfspan, R, and R+halfspan. For each source, solve the full three-dimensional DGL equations using the specified vacuum initialization, checkerboard ordering, damping 0.8, and residual tolerance. Use the periodic Green function and corrected Hodge decomposition to obtain each solenoidal potential.



Return the scalar

Q=100*[Vsole(R+halfspan)-2*Vsole(R)+Vsole(R-halfspan)]/[halfspan*pi*mB^2].



Defaults are L=16, R=5, halfspan=2, mB=mchi=0.5, and tolerance=1e-9.



Require an even integer L>=4, integer R, positive integer halfspan, positive finite equal masses, positive finite tolerance, and all three separations satisfying 1<=separation<L/2. Raise ValueError for invalid inputs or solver nonconvergence.

Section III provides finite-length solutions and separates the potential into Coulombic and solenoidal contributions using the periodic lattice Green function.



The requested scalar compares the secant tension between R-halfspan and R with the secant tension between R and R+halfspan. Their difference is expressed as a percentage of the continuum unit-flux Bogomolnyi reference pi*mB^2.



The calculation retains periodic-box effects and the finite-volume Hodge correction. It is not an infinitely long cylindrical tension or a continuum-extrapolated terminal correction. The final oracle must use the preceding oracle functions, reaching every earlier component through its dependency chain.

Returns
-------
One finite Python float containing Q, the difference between adjacent solenoidal secant tensions expressed as a percentage of the continuum reference pi*mB^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_length_curvature(
    L: int = 16,
    R: int = 5,
    halfspan: int = 2,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
) -> float:
    """Compute finite-box solenoidal curvature through all earlier steps.

    Parameters
    ----------
    L : int, default 16
        Even cubic side length at least 4; booleans are excluded.
    R : int, default 5
        Central separation; booleans are excluded.
    halfspan : int, default 2
        Positive integer separation increment; booleans are excluded.
        Each of R-halfspan, R, R+halfspan must lie in [1,L/2).
    mB : float, default 0.5
        Positive finite gauge mass.
    mchi : float, default 0.5
        Positive finite scalar mass, equal to mB in this benchmark.
    tolerance : float, default 1e-9
        Positive finite threshold for all normalized residual channels.

    Returns
    -------
    float
        Q=100*(Vsole(R+halfspan)-2*Vsole(R)+Vsole(R-halfspan))
        /(halfspan*pi*mB**2). This is a finite-box difference of
        secant tensions expressed as a percentage of pi*mB**2.
        Each solve uses damping 0.8 and at most 20000 sweeps.

    Raises
    ------
    ValueError
        For violations of the stated numeric/integer contracts,
        invalid separation ranges, unequal masses, solver failure
        (including invalid Newton curvature) or a nonfinite result.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_length_curvature(
    L: int = 16,
    R: int = 5,
    halfspan: int = 2,
    mB: float = 0.5,
    mchi: float = 0.5,
    tolerance: float = 1.0e-9,
) -> float:
    """Compute the finite-box solenoidal-potential curvature."""
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
    ):
        raise ValueError("R must be an integer")

    if (
        isinstance(halfspan, (bool, np.bool_))
        or not isinstance(halfspan, (int, np.integer))
        or halfspan < 1
    ):
        raise ValueError("halfspan must be a positive integer")

    if (
        not np.isfinite(mB)
        or not np.isfinite(mchi)
        or mB <= 0.0
        or mchi <= 0.0
    ):
        raise ValueError("masses must be positive and finite")

    if mB != mchi:
        raise ValueError(
            "this benchmark requires equal masses mB=mchi"
        )

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")

    separations = (
        R - halfspan,
        R,
        R + halfspan,
    )

    if any(
        not 1 <= separation < L // 2
        for separation in separations
    ):
        raise ValueError(
            "all separations must satisfy 1 <= separation < L/2"
        )

    sources = [
        _oracle_build_finite_source(L, separation, 1)
        for separation in separations
    ]

    G = _oracle_periodic_green(L)
    potentials = []

    for sigma in sources:
        state = _oracle_solve_finite_tube(
            sigma,
            mB=mB,
            mchi=mchi,
            tolerance=tolerance,
            max_sweeps=20000,
            damping=0.8,
        )

        direct_curl = _oracle_lattice_curl(state[:3])
        if (
            direct_curl.shape != sigma.shape
            or not np.all(np.isfinite(direct_curl))
        ):
            raise ValueError("direct curl consistency check failed")

        direct_equations = _oracle_lattice_equations(
            state,
            sigma,
            mB,
            mchi,
        )
        direct_residual = float(
            np.max(np.abs(direct_equations[:5]))
        )
        if direct_residual >= tolerance:
            raise ValueError("stationary residual check failed")

        post_sweep_state = _oracle_newton_sweep(
            state,
            sigma,
            mB,
            mchi,
            0.8,
        )
        post_sweep_equations = _oracle_lattice_equations(
            post_sweep_state,
            sigma,
            mB,
            mchi,
        )
        post_sweep_residual = float(
            np.max(np.abs(post_sweep_equations[:5]))
        )
        if (
            post_sweep_state.shape != state.shape
            or not np.all(np.isfinite(post_sweep_state))
            or post_sweep_residual >= tolerance
        ):
            raise ValueError("post-sweep consistency check failed")

        actions = _oracle_hodge_action(
            state,
            sigma,
            G,
            mB=mB,
            mchi=mchi,
        )

        potentials.append(float(actions[2]))

    curvature = (
        100.0
        * (
            potentials[2]
            - 2.0 * potentials[1]
            + potentials[0]
        )
        / (halfspan * np.pi * mB ** 2)
    )

    if not np.isfinite(curvature):
        raise ValueError("computed curvature is not finite")

    return float(curvature)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '',
      'call': 'run_length_curvature()',
      'gold_call': '_oracle_run_length_curvature()',
      'tol': 1e-06},
     {'setup': '',
      'call': 'run_length_curvature(L=12, R=3, halfspan=1)',
      'gold_call': '_oracle_run_length_curvature(L=12, R=3, halfspan=1)',
      'tol': 1e-06},
     {'setup': '',
      'call': 'run_length_curvature(L=14, R=4, halfspan=1)',
      'gold_call': '_oracle_run_length_curvature(L=14, R=4, halfspan=1)',
      'tol': 1e-06},
     {'setup': 'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'raises_value_error(run_length_curvature, halfspan=0)',
      'gold_call': '1.0',
      'tol': 0.0}]
