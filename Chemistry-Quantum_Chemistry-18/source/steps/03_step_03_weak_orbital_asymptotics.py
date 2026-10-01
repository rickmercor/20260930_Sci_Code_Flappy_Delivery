"""
Step 03: Constants of the weak natural-orbital asymptotics. Constants of the weak natural-orbital asymptotics of a two-electron singlet, from its on-top density.

The natural orbitals of a correlated two-electron state fall into a strongly occupied group and an infinite tail of
weakly occupied ones. The tail is what makes the on-top two-electron density converge so slowly with the size of a
one-electron basis, and the source paper shows that the tail is not arbitrary: its amplitudes, the local weight its
orbitals carry at a point, and the number of orbitals of a given angular momentum among the most occupied ones are all
governed by the on-top density of the same state, through a small set of universal constants.

Two of those constants are integrals of the on-top density itself. Because the tail contributes to a volume sum in one
place and to a radial count in another, the two integrals are taken over different measures and with different powers,
and interchanging them is the usual way of getting the asymptotics wrong. The remaining two are the coefficient of the
algebraic decay of the amplitudes when they are ordered by magnitude and counted with their degeneracies, and the
coefficient that turns a budget of natural orbitals into a count of the spherically symmetric ones among them.

For the pair functions of this task the on-top density is available in closed form, so all four constants are fixed by
the trap frequency and the correlation factor alone. They are the input to every large-basis estimate made later: the
rate at which a truncated natural-orbital expansion approaches the exact reduced on-top density is set by them, and so
is the number of shells that a given orbital budget buys.

Returns
-------
numpy.ndarray [J, I1, a, K]: the two integrals of the on-top density and the two coefficients of the weak natural-orbital asymptotics
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weak_orbital_asymptotics(omega: float, c: float) -> "np.ndarray":
    '''Constants governing the weakly occupied natural orbitals of the pair function Psi.

    Parameters
    ----------
    omega : float
        Trap frequency, omega > 0, of the Gaussian factor exp[-omega (r1^2 + r2^2) / 2].
    c : float
        Coefficient, c >= 0, of the quadratic term of the correlation factor p(s) = 1 + s / 2 + c s^2.

    Returns
    -------
    result : np.ndarray
        Real array [J, I1, a, K] of four constants of the weak natural-orbital asymptotics, for the normalized pair
        function with these parameters.

        J is the integral of the on-top density raised to the power 3/8 over all space, in the volume element
        d^3r = 4 pi r^2 dr. I1 is the integral of the on-top density raised to the power 1/8 over the radius from
        0 to infinity, in the element dr. a is the coefficient of the large-n law |lambda_n| -> a n^(-4/3) obeyed by
        the natural amplitudes ordered by decreasing magnitude and counted with the 2l + 1 degeneracy of each shell.
        K is the coefficient of the large-n law N_s -> K n^(1/3) for the number of l = 0 natural orbitals among the n
        most occupied ones. All four are positive.

    Raises
    ------
    ValueError
        If omega is not positive or c is negative.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_weak_orbital_asymptotics(omega: float, c: float) -> "np.ndarray":
    trap = float(omega)
    quad = float(c)
    if not trap > 0.0:
        raise ValueError("omega must be positive")
    if quad < 0.0:
        raise ValueError("c must be non-negative")
    norm = _oracle_pair_normalization(trap, quad)
    volume_integral = norm ** 0.75 * (4.0 * np.pi / (3.0 * trap)) ** 1.5
    radial_integral = norm ** 0.25 * (np.pi / trap) ** 0.5
    amplitude = (np.sqrt(2.0) * volume_integral / (3.0 * np.pi ** 1.25)) ** (4.0 / 3.0)
    shell_count = (6.0 * radial_integral ** 3 / (np.pi * volume_integral)) ** (1.0 / 3.0)
    return np.array([volume_integral, radial_integral, amplitude, shell_count], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def logged(values):
    v = np.asarray(values, dtype=float).reshape(4)
    return np.log(v)
"""
    return [
        # --- Normal: the harmonium ground state of the benchmark ---
        {"setup": setup, "call": "logged(weak_orbital_asymptotics(0.1, 0.05))",
         "gold_call": "logged(_oracle_weak_orbital_asymptotics(0.1, 0.05))", "tol": 1e-9},
        # --- Normal: a tighter trap, where the on-top density and both integrals change scale ---
        {"setup": setup, "call": "logged(weak_orbital_asymptotics(0.5, 0.05))",
         "gold_call": "logged(_oracle_weak_orbital_asymptotics(0.5, 0.05))", "tol": 1e-9},
        # --- Boundary: an uncorrelated pair function, c = 0, where only the cusp term survives ---
        {"setup": setup, "call": "logged(weak_orbital_asymptotics(0.25, 0.0))",
         "gold_call": "logged(_oracle_weak_orbital_asymptotics(0.25, 0.0))", "tol": 1e-9},
        # --- Boundary: a loose trap with a strong correlation factor, a corner of the documented domain ---
        {"setup": setup, "call": "logged(weak_orbital_asymptotics(0.05, 0.5))",
         "gold_call": "logged(_oracle_weak_orbital_asymptotics(0.05, 0.5))", "tol": 1e-9},
        # --- Error: a negative quadratic coefficient is outside the family ---
        {"setup": "def _probe(fn):\n    try:\n        fn(0.1, -0.5)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(weak_orbital_asymptotics)", "gold_call": "_probe(_oracle_weak_orbital_asymptotics)"},
    ]
