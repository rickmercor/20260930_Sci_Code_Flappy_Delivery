"""
Find a zero of the effective dispersion function of the laminate, $D(K, s) = -1 / \Theta_r(K, s)$, where $\Theta_r$ is the mean-temperature amplitude of the response to a unit heat source (step 03), by the secant method started from the two given complex wavenumbers. Stop when the secant update is at most tolerance times the modulus of the new iterate, returning that iterate without evaluating $D$ there; raise an error if this does not happen within max_iterations updates. Units are those of step 01.

The function $D$ vanishes where the mean response of the laminate to a unit heat source has a pole. The iteration count is the number of secant updates made, including the one that meets the stopping rule, and dispersion_at_guess is $D$ evaluated at first_guess.

Returns
-------
dict holding the complex scalars root and dispersion_at_guess, the value of $D$ at first_guess, and the integer iterations, the number of secant updates made.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_bloch_dispersion_root(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    first_guess: complex,
    second_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    r"""Zero of the dispersion function $D = -1/\Theta_r$ of the source-driven effective medium, by the secant method.

    Parameters
    ----------
    conductivities : np.ndarray
        Layer conductivities $\kappa_j$ in order along $+x$.
    capacities : np.ndarray
        Layer volumetric heat capacities $c_j$.
    thicknesses : np.ndarray
        Layer thicknesses $h_j$.
    laplace_s : complex
        Laplace variable $s$.
    first_guess : complex
        First starting wavenumber $K_0$.
    second_guess : complex
        Second starting wavenumber $K_1$.
    tolerance : float
        Relative stopping tolerance on the secant update.
    max_iterations : int
        Largest number of secant updates.

    Returns
    -------
    dict
        Under the keys root, dispersion_at_guess and iterations.

    Raises
    ------
    ValueError
        When an input is invalid for the forced cell problem, when the guesses coincide or are not finite numbers, when the tolerance is not a finite real number above zero, when the iteration limit is not an integer of at least 1, when two successive values of $D$ coincide, or when the iteration does not converge.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _dispersion(conductivities, capacities, thicknesses, s, K):
    out = _oracle_forced_bloch_cell_response(conductivities, capacities, thicknesses, s, K, 1.0, 0.0, 0.0)  # noqa: F821
    if out["mean_temperature"] == 0:
        raise ValueError("the mean temperature of the heat-source response vanishes")
    return -1.0 / out["mean_temperature"]


def _oracle_effective_bloch_dispersion_root(
    conductivities: np.ndarray,
    capacities: np.ndarray,
    thicknesses: np.ndarray,
    laplace_s: complex,
    first_guess: complex,
    second_guess: complex,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation."""
    if isinstance(tolerance, (bool, complex, np.complexfloating)):
        raise ValueError("tolerance must be a real number")
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError):
        raise ValueError("tolerance must be a real number") from None
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and above zero")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be an integer of at least 1")
    try:
        k0, k1 = complex(first_guess), complex(second_guess)
    except (TypeError, ValueError):
        raise ValueError("the starting wavenumbers must be numbers") from None
    if not (np.isfinite(k0) and np.isfinite(k1)) or k0 == k1:
        raise ValueError("the starting wavenumbers must be finite and distinct")
    s = complex(laplace_s)
    d0 = _dispersion(conductivities, capacities, thicknesses, s, k0)
    first = d0
    d1 = _dispersion(conductivities, capacities, thicknesses, s, k1)
    for iteration in range(1, max_iterations + 1):
        if d1 == d0:
            raise ValueError("two successive values of the dispersion function coincide")
        k2 = k1 - d1 * (k1 - k0) / (d1 - d0)
        if abs(k2 - k1) <= tolerance * abs(k2):
            return {"root": complex(k2), "dispersion_at_guess": complex(first), "iterations": iteration}
        k0, d0 = k1, d1
        k1 = k2
        d1 = _dispersion(conductivities, capacities, thicknesses, s, k1)
    raise ValueError("the secant iteration did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    # flatten to a tuple of plain real terminals, complex numbers split into real and imaginary parts, -0.0 read as 0.0
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, complex):
        return (x.real + 0.0, x.imag + 0.0)
    if isinstance(x, bool):
        return (int(x),)
    if isinstance(x, float):
        return (x + 0.0,)
    return (x,)
"""
    SETUP = """
import numpy as np
def graded():
    return (np.array([1.38, 400.0, 35.0, 148.0]), np.array([1.65, 3.45, 3.06, 1.66]), np.array([0.35, 0.15, 0.30, 0.20]))
def tri():
    return (np.array([1.38, 719.0, 400.0]), np.array([1.65, 1.78, 3.45]), np.array([0.3, 0.4, 0.3]))
def bi():
    return (np.array([1.38, 719.0, 1.38]), np.array([1.65, 1.78, 1.65]), np.array([0.3, 0.4, 0.3]))
def four():
    return (np.array([0.19, 21.9, 148.0, 1.38]), np.array([1.73, 2.36, 1.66, 1.65]), np.array([0.2, 0.3, 0.1, 0.4]))
def trace_root(kap, cap, h, s):
    # Bloch wavenumber from the trace formula, written out here so that the case needs no other step
    T = np.eye(2, dtype=complex)
    for a, c, d in zip(kap, cap, h):
        k = np.sqrt(c * s / a)
        T = np.array([[np.cosh(k * d), a * k * np.sinh(k * d)], [np.sinh(k * d) / (a * k), np.cosh(k * d)]]) @ T
    z = np.arccosh(0.5 * (T[0, 0] + T[1, 1]))
    return -z if z.real < 0 else z
def bloch_case(fn, cell, s, lo, hi, tol, cap_it):
    kap, cap, h = cell()
    kb = trace_root(kap, cap, h, s)
    out = fn(kap, cap, h, s, lo * kb, hi * kb, tol, cap_it)
    return (round(abs(out["root"] - kb), 5), complex(np.round(out["root"].real, 5), np.round(out["root"].imag, 5)),
            complex(np.round(out["dispersion_at_guess"].real, 7), np.round(out["dispersion_at_guess"].imag, 7)),
            int(isinstance(out["iterations"], (int, np.integer)) and 1 <= out["iterations"] <= cap_it))
def homogeneous_case(fn, sign):
    # the root is also a resonance of the particular solution in the single material, so the stopping tolerance is
    # kept at $10^{-5}$: every trial wavenumber, including the accepted iterate, stays at least $10^{-10}$ relative from it
    kap, cap, h = np.array([3.0, 3.0]), np.array([2.0, 2.0]), np.array([0.5, 0.5])
    k = sign * np.sqrt(2.0 * 4j / 3.0)
    out = fn(kap, cap, h, 4j, 0.6 * k, 1.3 * k, 1e-5, 60)
    return (round(abs(out["root"] - k), 6), complex(np.round(out["dispersion_at_guess"].real, 9), np.round(out["dispersion_at_guess"].imag, 9)),
            int(isinstance(out["iterations"], (int, np.integer)) and 1 <= out["iterations"] <= 60))
def verdict(fn, **kw):
    kap, cap, h = tri()
    args = dict(conductivities=kap, capacities=cap, thicknesses=h, laplace_s=10j, first_guess=1.0 + 1.0j, second_guess=2.0 + 1.5j, tolerance=1e-10, max_iterations=50)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT
    return [
        {
            # normal: the four-layer cell of the task at $s = 15 i$, seeded around its Bloch wavenumber
            "setup": SETUP,
            "call": "flat(bloch_case(effective_bloch_dispersion_root, graded, 15j, 0.9, 1.1, 1e-10, 60))",
            "gold_call": "flat(bloch_case(_oracle_effective_bloch_dispersion_root, graded, 15j, 0.9, 1.1, 1e-10, 60))",
        },
        {
            # normal: asymmetric three-layer cell at dimensionless frequency 10
            "setup": SETUP,
            "call": "flat(bloch_case(effective_bloch_dispersion_root, tri, 10j, 0.8, 1.25, 1e-10, 60))",
            "gold_call": "flat(bloch_case(_oracle_effective_bloch_dispersion_root, tri, 10j, 0.8, 1.25, 1e-10, 60))",
        },
        {
            # boundary: a homogeneous cell converges to $\sqrt{c s / \kappa}$
            "setup": SETUP,
            "call": "flat(homogeneous_case(effective_bloch_dispersion_root, 1.0))",
            "gold_call": "flat(homogeneous_case(_oracle_effective_bloch_dispersion_root, 1.0))",
        },
        {
            # boundary: the homogeneous cell seeded on the other side returns the backward root
            "setup": SETUP,
            "call": "flat(homogeneous_case(effective_bloch_dispersion_root, -1.0))",
            "gold_call": "flat(homogeneous_case(_oracle_effective_bloch_dispersion_root, -1.0))",
        },
        {
            # edge: a four-layer cell at a damped Laplace variable, seeded far from the root
            "setup": SETUP,
            "call": "flat(bloch_case(effective_bloch_dispersion_root, four, 2.0 + 12j, 0.5, 0.7, 1e-10, 80))",
            "gold_call": "flat(bloch_case(_oracle_effective_bloch_dispersion_root, four, 2.0 + 12j, 0.5, 0.7, 1e-10, 80))",
        },
        {
            # invalid input: a first guess that is not a number
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, first_guess=None)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, first_guess=None)",
        },
        {
            # invalid input: a complex tolerance
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, tolerance=1e-10 + 0j)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, tolerance=1e-10 + 0j)",
        },
        {
            # invalid input: coinciding guesses
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, second_guess=1.0 + 1.0j)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, second_guess=1.0 + 1.0j)",
        },
        {
            # invalid input: a zero tolerance
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, tolerance=0.0)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, tolerance=0.0)",
        },
        {
            # invalid input: a non-integer iteration limit
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, max_iterations=2.5)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, max_iterations=2.5)",
        },
        {
            # invalid input: an iteration limit too small to converge
            "setup": SETUP,
            "call": "verdict(effective_bloch_dispersion_root, max_iterations=2)",
            "gold_call": "verdict(_oracle_effective_bloch_dispersion_root, max_iterations=2)",
        },
    ]
