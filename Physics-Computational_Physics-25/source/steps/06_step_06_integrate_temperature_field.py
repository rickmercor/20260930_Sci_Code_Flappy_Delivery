"""
March the semi-discrete thermal system from a uniform initial state to the end of the analysis, combining the exact propagators supplied by sub-problem 05 with a quadratured particular integral.

The solution of a linear first-order system over one step is the propagator acting on the state at the start of the step plus the convolution of the propagator with the load over the step. The first term is available to working precision; the second is not, because the load varies within the step and the convolution has no closed form for a general history. Evaluating it by Gaussian quadrature amounts to sampling the load at the quadrature abscissae of the step and weighting each sample by the propagator over the time remaining from that abscissa to the end of the step. Since those remaining times are fixed fractions of the step, the corresponding propagators are the same at every step, so they are built once by the caller and handed to the march, which is what makes the scheme cheap despite each of them costing a full precise-integration sequence.




The accuracy of the whole march is governed by that quadrature and not by the propagator. The integrand is the load history multiplied by an exponentially decaying matrix factor, and the decay rate of the fastest component is the largest eigenvalue magnitude of the state matrix. When the step is comparable with the corresponding time constant the integrand is smooth on the step and a low-order rule is excellent; when the step is many times longer the fast components of the integrand are concentrated near the end of the step, close to an impulse, and a fixed-abscissa rule underestimates them badly. The symptom is characteristic and easy to misread: the slow, deep part of the field looks sensible while the fastest degrees of freedom, which here are the surface nodes carrying the convective exchange, come out far too cold. Because refining the spatial mesh raises the largest eigenvalue as the inverse square of the element size, a step that is adequate on one mesh is not adequate on a finer one, and the scheme therefore has a step-size restriction of accuracy even though it has none of stability.




The load enters here as a fixed spatial vector times a scalar history, which is what the convective patch with a spatially uniform ambient temperature produces. The history used is a saturating exponential approach to a final ambient temperature, so it is smooth and bounded and contributes no quadrature difficulty of its own; the whole burden falls on the matrix factor. Starting from a uniform zero field means the entire response is the particular integral, so any deficiency of the quadrature appears directly in the answer with nothing to mask it.

Returns
-------
np.ndarray of shape (n,), float: the nodal temperature field in degrees Celsius at time step * n_steps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def integrate_temperature_field(step_propagator: np.ndarray, tail_propagators: list,
                                load_influence: np.ndarray, ambient_amplitude: float,
                                ramp_time: float, step: float, n_steps: int) -> np.ndarray:
    """March the semi-discrete thermal system to the end of the analysis.

    The load acting on the system at time t is ``load_influence`` scaled by the
    ambient history ``ambient_amplitude * (1 - exp(-t / ramp_time))``, and the
    field starts from zero everywhere at t = 0.

    Every propagator the march needs is supplied by the caller, which obtains it
    from ``compute_precise_propagator`` of sub-problem 05; none is built here.
    The order of the Gauss-Legendre rule is the length of ``tail_propagators``,
    and its abscissae and weights are those of
    ``numpy.polynomial.legendre.leggauss(len(tail_propagators))``, so entry i of
    ``tail_propagators`` belongs to abscissa i of that rule in the order the
    routine returns them.

    Parameters
    ----------
    step_propagator : np.ndarray
        Square array of shape (n, n) holding the propagator of the system over
        a whole step.
    tail_propagators : list
        Sequence of at least one array, each of shape (n, n), holding the
        propagator over the time remaining from a quadrature abscissa of the
        step to the end of that step, one entry per abscissa in the order
        described above.
    load_influence : np.ndarray
        Vector of length n giving the response rate per unit ambient
        temperature, in K s^-1 K^-1.
    ambient_amplitude : float
        Final ambient temperature of the saturating history, in degrees
        Celsius.
    ramp_time : float
        Time constant of the ambient history in seconds (ramp_time > 0).
    step : float
        Time step in seconds (step > 0), the same step the supplied
        propagators were built over.
    n_steps : int
        Number of steps to march (n_steps >= 0).

    Returns
    -------
    temperature : np.ndarray
        Vector of length n holding the nodal temperatures at the end of the
        analysis, in degrees Celsius.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return temperature  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_integrate_temperature_field(step_propagator: np.ndarray, tail_propagators: list,
                                        load_influence: np.ndarray, ambient_amplitude: float,
                                        ramp_time: float, step: float,
                                        n_steps: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_propagators(step_propagator, tail_propagators):
        """Raise ValueError unless the supplied propagators are square and consistent."""
        whole = np.asarray(step_propagator, dtype=float)
        if whole.ndim != 2 or whole.shape[0] != whole.shape[1] or whole.shape[0] == 0:
            raise ValueError("step_propagator must be a non-empty square two-dimensional array")
        try:
            tails = [np.asarray(tail, dtype=float) for tail in tail_propagators]
        except TypeError:
            raise ValueError("tail_propagators must be a sequence of arrays")
        if len(tails) < 1:
            raise ValueError("tail_propagators must hold at least one propagator")
        if any(tail.shape != whole.shape for tail in tails):
            raise ValueError("every tail propagator must have the shape of step_propagator")
        if not (np.all(np.isfinite(whole)) and all(np.all(np.isfinite(tail)) for tail in tails)):
            raise ValueError("the propagators must contain only finite entries")
        return whole, tails

    whole, tails = _validate_propagators(step_propagator, tail_propagators)
    load = np.asarray(load_influence, dtype=float)
    if load.ndim != 1 or load.shape[0] != whole.shape[0]:
        raise ValueError("load_influence must be a vector of the same order as the propagators")
    if not np.all(np.isfinite(load)):
        raise ValueError("load_influence must contain only finite entries")
    if not (isinstance(ambient_amplitude, (int, float)) and np.isfinite(ambient_amplitude)):
        raise ValueError("ambient_amplitude must be a finite number")
    for name, value in (("ramp_time", ramp_time), ("step", step)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_steps, (int, np.integer)) and not isinstance(n_steps, bool)
            and int(n_steps) >= 0):
        raise ValueError("n_steps must be an integer >= 0")

    ambient_amplitude = float(ambient_amplitude)
    ramp_time = float(ramp_time)
    step = float(step)
    n_steps = int(n_steps)

    # The rule is fixed by how many tail propagators were supplied, one per
    # abscissa, so the caller and the march cannot disagree about its order.
    abscissae, weights = np.polynomial.legendre.leggauss(len(tails))
    driven = [tail @ load for tail in tails]

    temperature = np.zeros(whole.shape[0], dtype=float)
    for index in range(n_steps):
        start = index * step
        particular = np.zeros_like(temperature)
        for weight, node, contribution in zip(weights, abscissae, driven):
            sample = start + 0.5 * step * (1.0 + node)
            ambient = ambient_amplitude * (1.0 - np.exp(-sample / ramp_time))
            particular += weight * ambient * contribution
        temperature = whole @ temperature + 0.5 * step * particular
    return temperature

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    # The propagators are built from a closed-form matrix exponential of the
    # symmetric test systems, so these cases measure the quadrature and the
    # recursion of this step alone and never sub-problem 05.
    return [
        # --- Valid: benchmark march on a small stiff system (normal scenario) ---
        {
            "setup": """import numpy as np
n = 10
diag = np.linspace(2.0e-4, 3.0e-3, n)
off = np.diag(np.linspace(1.0e-5, 4.0e-4, n - 1), 1)
state = -(np.diag(diag) + 0.5 * (off + off.T))
load = np.zeros(n)
load[0] = 2.0e-3
load[1] = 5.0e-4
ambient_amplitude, ramp_time = 80.0, 1.0e5
step, n_steps, n_gauss = 2500.0, 80, 6

def propagators(state, step, n_gauss):
    spectrum, vectors = np.linalg.eigh(state)
    def over(interval):
        return (vectors * np.exp(spectrum * interval)) @ vectors.T
    nodes, _ = np.polynomial.legendre.leggauss(n_gauss)
    return over(step), [over(0.5 * step * (1.0 - node)) for node in nodes]

whole, tails = propagators(state, step, n_gauss)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(integrate_temperature_field(whole, tails, load, ambient_amplitude, ramp_time, step, n_steps))",
            "gold_call": "digest(_oracle_integrate_temperature_field(whole, tails, load, ambient_amplitude, ramp_time, step, n_steps))",
        },
        # --- Valid: coarser quadrature on a longer step, where the rule loses accuracy ---
        {
            "setup": """import numpy as np
state = np.array([[-4.0e-3, 8.0e-4, 0.0],
                  [8.0e-4, -1.2e-3, 2.0e-4],
                  [0.0, 2.0e-4, -3.0e-4]])
load = np.array([3.0e-3, 0.0, 0.0])
ambient_amplitude, ramp_time = 80.0, 1.0e5
step, n_steps, n_gauss = 20000.0, 10, 2

def propagators(state, step, n_gauss):
    spectrum, vectors = np.linalg.eigh(state)
    def over(interval):
        return (vectors * np.exp(spectrum * interval)) @ vectors.T
    nodes, _ = np.polynomial.legendre.leggauss(n_gauss)
    return over(step), [over(0.5 * step * (1.0 - node)) for node in nodes]

whole, tails = propagators(state, step, n_gauss)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(integrate_temperature_field(whole, tails, load, ambient_amplitude, ramp_time, step, n_steps))",
            "gold_call": "digest(_oracle_integrate_temperature_field(whole, tails, load, ambient_amplitude, ramp_time, step, n_steps))",
        },
        # --- Boundary: no steps taken, so the field stays at its initial value ---
        {
            "setup": """import numpy as np
state = np.array([[-1.0e-3, 0.0], [0.0, -2.0e-3]])
load = np.array([1.0e-3, 2.0e-3])
step, n_gauss = 2500.0, 6

def propagators(state, step, n_gauss):
    spectrum, vectors = np.linalg.eigh(state)
    def over(interval):
        return (vectors * np.exp(spectrum * interval)) @ vectors.T
    nodes, _ = np.polynomial.legendre.leggauss(n_gauss)
    return over(step), [over(0.5 * step * (1.0 - node)) for node in nodes]

whole, tails = propagators(state, step, n_gauss)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, step, 0))",
            "gold_call": "digest(_oracle_integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, step, 0))",
        },
        # --- Edge: single node driven far past saturation of the ambient history ---
        {
            "setup": """import numpy as np
state = np.array([[-5.0e-4]])
load = np.array([5.0e-4])
step, n_gauss = 2500.0, 6

def propagators(state, step, n_gauss):
    spectrum, vectors = np.linalg.eigh(state)
    def over(interval):
        return (vectors * np.exp(spectrum * interval)) @ vectors.T
    nodes, _ = np.polynomial.legendre.leggauss(n_gauss)
    return over(step), [over(0.5 * step * (1.0 - node)) for node in nodes]

whole, tails = propagators(state, step, n_gauss)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(integrate_temperature_field(whole, tails, load, 80.0, 1.0e4, step, 400))",
            "gold_call": "digest(_oracle_integrate_temperature_field(whole, tails, load, 80.0, 1.0e4, step, 400))",
        },
        # --- Invalid: load vector inconsistent with the order of the propagators ---
        {
            "setup": """import numpy as np
whole = np.eye(3)
tails = [np.eye(3), np.eye(3)]
load = np.zeros(2)
def run_model():
    try:
        integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, 2500.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, 2500.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive time step ---
        {
            "setup": """import numpy as np
whole = np.eye(2)
tails = [np.eye(2)]
load = np.zeros(2)
def run_model():
    try:
        integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, 0.0, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_temperature_field(whole, tails, load, 80.0, 1.0e5, 0.0, 5)
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
