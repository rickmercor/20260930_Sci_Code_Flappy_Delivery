"""
Evaluate the kinetic response function of a Maxwellian electron population at a set of normalised complex phase speeds.

The linear electrostatic response of a Maxwellian plasma is carried by a single analytic function of the normalised phase speed, and every fluid closure considered here is a rational approximation of that one function. Its values below the real axis are what make Landau damping visible to a fluid model.

Returns
-------
np.ndarray of the same shape as zeta, complex: the Maxwellian kinetic response function.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_kinetic_response(zeta: np.ndarray) -> np.ndarray:
    """Evaluate the Maxwellian kinetic response function.

    Parameters
    ----------
    zeta : np.ndarray
        Array of normalised complex phase speeds zeta = omega/(sqrt(2)*abs(k)*v_t),
        where v_t = sqrt(T0/m), T0 is the equilibrium temperature in energy
        units, and m is the electron mass. Real input is accepted and treated
        as having zero imaginary part. Any shape is accepted.

    Returns
    -------
    response : np.ndarray
        Complex array of the same shape as ``zeta``, holding the kinetic
        response function of a Maxwellian equilibrium at each phase speed. The
        function equals one at zero phase speed and behaves as minus one half
        the inverse square of the phase speed when the phase speed is large.

    Raises
    ------
    ValueError
        If ``zeta`` is empty or holds a non-finite entry.
    """
    return response  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_kinetic_response(zeta: np.ndarray) -> np.ndarray:
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
        # The rational form converges only above the real axis; below it the
        # reflection w(z) + w(-z) = 2 exp(-z**2) supplies the continuation.
        upper = argument.imag >= 0.0
        folded = np.where(upper, argument, -argument)
        mapped = (scale + 1j * folded) / (scale - 1j * folded)
        series = np.polyval(coefficients, mapped)
        value = (2.0 * series / (scale - 1j * folded) ** 2
                 + (1.0 / np.sqrt(np.pi)) / (scale - 1j * folded))
        return np.where(upper, value, 2.0 * np.exp(-folded ** 2) - value)

    speeds = np.asarray(zeta, dtype=complex)
    if speeds.size < 1:
        raise ValueError("zeta must hold at least one phase speed")
    if not np.all(np.isfinite(speeds)):
        raise ValueError("zeta must contain only finite entries")

    # The plasma dispersion function is the Faddeeva function up to the factor
    # i*sqrt(pi); the density response is one plus the phase speed times it.
    dispersion = 1j * np.sqrt(np.pi) * _faddeeva(speeds.ravel())
    response = 1.0 + speeds.ravel() * dispersion
    return response.reshape(speeds.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: phase speeds straddling the least-damped Landau roots of the
        #     wave numbers this task uses (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
zeta = np.array([3.761752712 - 0.000194834j,
                 2.733784372 - 0.029746494j,
                 2.271681244 - 0.116898820j,
                 2.002048243 - 0.216883038j,
                 1.821632880 - 0.311257019j])
""",
            "call": "sig(evaluate_kinetic_response(zeta), 1.0)",
            "gold_call": "sig(_oracle_evaluate_kinetic_response(zeta), 1.0)",
        },
        # --- Valid: a mixed grid covering both half planes and a purely real line,
        #     which exercises the continuation across the real axis ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
real = np.linspace(-2.5, 2.5, 7)
imag = np.array([-0.9, -0.25, 0.0, 0.4])
zeta = (real[:, None] + 1j * imag[None, :]).ravel()
""",
            "call": "sig(evaluate_kinetic_response(zeta), 10.0)",
            "gold_call": "sig(_oracle_evaluate_kinetic_response(zeta), 10.0)",
        },
        # --- Boundary: the origin, where the response takes its normalising value ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
zeta = np.array([0.0 + 0.0j])
""",
            "call": "sig(evaluate_kinetic_response(zeta), 1.0)",
            "gold_call": "sig(_oracle_evaluate_kinetic_response(zeta), 1.0)",
        },
        # --- Edge: a large real phase speed, where the response has collapsed onto
        #     its fluid asymptote and the exponentially small Landau part is lost ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
zeta = np.array([14.2 - 1.0e-9j, 25.0 + 0.0j])
""",
            "call": "sig(evaluate_kinetic_response(zeta), 0.001)",
            "gold_call": "sig(_oracle_evaluate_kinetic_response(zeta), 0.001)",
        },
        # --- Edge: a two-dimensional input, whose shape must survive ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    v = np.concatenate([a.real.ravel(), a.imag.ravel()]) if np.iscomplexobj(a) else np.asarray(a, dtype=float).ravel()
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k) + 100.0 * np.asarray(a).ndim) / s), 9)
zeta = np.array([[1.0 - 0.1j, 2.0 - 0.2j], [3.0 - 0.3j, 4.0 - 0.4j]])
""",
            "call": "sig(evaluate_kinetic_response(zeta), 1.0)",
            "gold_call": "sig(_oracle_evaluate_kinetic_response(zeta), 1.0)",
        },
        # --- Invalid: an empty input, which carries no phase speed to evaluate ---
        {
            "setup": """import numpy as np
zeta = np.array([], dtype=complex)
def run_model():
    try:
        evaluate_kinetic_response(zeta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_kinetic_response(zeta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite phase speed ---
        {
            "setup": """import numpy as np
zeta = np.array([1.0 - 0.1j, np.inf + 0.0j])
def run_model():
    try:
        evaluate_kinetic_response(zeta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_kinetic_response(zeta)
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
