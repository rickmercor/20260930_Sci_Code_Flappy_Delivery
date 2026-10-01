"""
Locate the least-damped root of the exact kinetic dispersion relation of an electrostatic Maxwellian plasma at a prescribed normalised wave number.

Closing the Poisson equation with the Maxwellian density response turns the dispersion relation into a single transcendental equation in the normalised phase speed, all of whose roots lie below the real axis. The root whose imaginary part is closest to zero is the one that survives longest and therefore dictates the long-time macroscopic behaviour.

Returns
-------
complex: the least-damped normalised phase speed with positive real part, as a native Python complex.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_kinetic_root(wavenumber: float) -> complex:
    """Locate the least-damped kinetic root at one normalised wave number.

    The dispersion relation is the statement that the Maxwellian kinetic
    response function of sub-problem 01 evaluated at the normalised phase speed
    equals minus the square of the normalised wave number.

    Parameters
    ----------
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium. It must be at least 0.1: below that the damping of the
        least-damped root falls under the resolution of double precision, the
        root can no longer be located reliably, and the function is not defined.

    Returns
    -------
    root : complex
        The normalised phase speed satisfying the dispersion relation that has
        a strictly positive real part and, among all such solutions, the
        largest (least negative) imaginary part.

    Raises
    ------
    ValueError
        If ``wavenumber`` is not a finite number, or is smaller than 0.1.
    """
    return root  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_kinetic_root(wavenumber: float) -> complex:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _faddeeva(argument, terms=48):
        """Weideman rational evaluation of the Faddeeva function."""
        half = 2 * terms
        full = 2 * half
        index = np.arange(-half + 1, half)
        scale = np.sqrt(terms / np.sqrt(2.0))
        node = scale * np.tan(index * np.pi / full)
        weight = np.exp(-node ** 2) * (scale ** 2 + node ** 2)
        coefficients = np.real(np.fft.fft(np.fft.fftshift(
            np.append(0.0, weight)))) / full
        coefficients = np.flipud(coefficients[1:terms + 1])
        upper = argument.imag >= 0.0
        folded = np.where(upper, argument, -argument)
        mapped = (scale + 1j * folded) / (scale - 1j * folded)
        series = np.polyval(coefficients, mapped)
        value = (2.0 * series / (scale - 1j * folded) ** 2
                 + (1.0 / np.sqrt(np.pi)) / (scale - 1j * folded))
        return np.where(upper, value, 2.0 * np.exp(-folded ** 2) - value)

    # Below a tenth of the Debye wave number the least-damped root is damped by
    # an exponentially small amount that double precision cannot resolve, and
    # the tracking returns a spurious root, so the domain is closed there.
    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) >= 0.1):
        raise ValueError("wavenumber must be a finite number of at least 0.1")
    wavenumber = float(wavenumber)

    def _response(point):
        speed = np.asarray([point], dtype=complex)
        dispersion = 1j * np.sqrt(np.pi) * _faddeeva(speed)
        return complex(1.0 + speed[0] * dispersion[0]), complex(dispersion[0])

    # The long-wavelength limit is where the least-damped branch is easiest to
    # identify, so the root is tracked there from the Bohm-Gross estimate and
    # then continued in the wave number up to the requested one.
    start = min(wavenumber, 0.05)
    grid = np.linspace(start, wavenumber,
                       max(2, int(np.ceil(wavenumber / 0.02)) + 1))
    point = (np.sqrt(1.0 + 3.0 * start ** 2) / (np.sqrt(2.0) * start)) - 0.001j
    for step_wavenumber in grid:
        offset = step_wavenumber ** 2
        for _ in range(200):
            response, dispersion = _response(point)
            # d/dzeta of the response, using Z'(zeta) = -2 (1 + zeta Z(zeta)).
            slope = dispersion - 2.0 * point * response
            if slope == 0.0:
                break
            correction = (response + offset) / slope
            point = point - correction
            if abs(correction) <= 1.0e-15 * max(1.0, abs(point)):
                break
    return complex(point)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark wave number of the study (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(z, s):
    z = complex(z)
    return round(float((abs(z.real) + abs(z.imag) + z.real * np.cos(1.0)
                        + z.imag * np.cos(2.0)) / s), 9)
wavenumber = 0.4
""",
            "call": "sig(solve_kinetic_root(wavenumber), 1.0)",
            "gold_call": "sig(_oracle_solve_kinetic_root(wavenumber), 1.0)",
        },
        # --- Valid: the short-wavelength end of the band, where the root is
        #     strongly damped and sits far from the Bohm-Gross estimate ---
        {
            "setup": """import numpy as np
def sig(z, s):
    z = complex(z)
    return round(float((abs(z.real) + abs(z.imag) + z.real * np.cos(1.0)
                        + z.imag * np.cos(2.0)) / s), 9)
wavenumber = 0.6
""",
            "call": "sig(solve_kinetic_root(wavenumber), 1.0)",
            "gold_call": "sig(_oracle_solve_kinetic_root(wavenumber), 1.0)",
        },
        # --- Valid: a wave number beyond the band, where the damping rate is
        #     comparable with the real frequency ---
        {
            "setup": """import numpy as np
def sig(z, s):
    z = complex(z)
    return round(float((abs(z.real) + abs(z.imag) + z.real * np.cos(1.0)
                        + z.imag * np.cos(2.0)) / s), 9)
wavenumber = 1.0
""",
            "call": "sig(solve_kinetic_root(wavenumber), 1.0)",
            "gold_call": "sig(_oracle_solve_kinetic_root(wavenumber), 1.0)",
        },
        # --- Boundary: the long-wavelength end of the band, where the root is
        #     only marginally damped and the imaginary part is tiny ---
        {
            "setup": """import numpy as np
def sig(z, s):
    z = complex(z)
    return round(float((abs(z.real) + abs(z.imag) + z.real * np.cos(1.0)
                        + z.imag * np.cos(2.0)) / s), 9)
wavenumber = 0.2
""",
            "call": "sig(solve_kinetic_root(wavenumber), 1.0)",
            "gold_call": "sig(_oracle_solve_kinetic_root(wavenumber), 1.0)",
        },
        # --- Edge: an integer wave number argument, which must be accepted ---
        {
            "setup": """import numpy as np
def sig(z, s):
    z = complex(z)
    return round(float((abs(z.real) + abs(z.imag) + z.real * np.cos(1.0)
                        + z.imag * np.cos(2.0)) / s), 9)
wavenumber = 2
""",
            "call": "sig(solve_kinetic_root(wavenumber), 1.0)",
            "gold_call": "sig(_oracle_solve_kinetic_root(wavenumber), 1.0)",
        },
        # --- Invalid: a vanishing wave number, at which the dispersion relation
        #     has no finite phase speed ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_kinetic_root(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_kinetic_root(0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a wave number below the resolvable domain, where the
        #     damping of the least-damped root is exponentially small ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_kinetic_root(0.05)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_kinetic_root(0.05)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative wave number ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_kinetic_root(-0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_kinetic_root(-0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
