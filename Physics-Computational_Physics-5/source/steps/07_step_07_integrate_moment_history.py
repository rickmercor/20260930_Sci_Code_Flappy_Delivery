"""
Advance a linear Fourier-space moment system over a fixed number of equal steps with an explicit two-stage second-order Runge-Kutta scheme and record every time level.

An initial-value problem posed on a linear moment system excites all of its modes at once, so what a fluid closure predicts at late times is not the least-damped mode alone but the whole superposition the discrete integrator propagates. Recording every time level is what makes the prediction comparable with a kinetic benchmark point by point rather than only through a fitted decay rate.

Returns
-------
np.ndarray of shape (step_count + 1, 3), complex: the moment state at every time level including the initial one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrate_moment_history(evolution_matrix: np.ndarray,
                             initial_state: np.ndarray,
                             time_step: float,
                             step_count: int) -> np.ndarray:
    """Integrate a linear moment system and record every time level.

    The advance is the explicit two-stage second-order Runge-Kutta scheme, in
    which a trial derivative taken at the current level is used to form an
    intermediate state and the increment is then taken from the derivative
    there.

    Parameters
    ----------
    evolution_matrix : np.ndarray
        Complex array of shape (3, 3) as returned by sub-problem 06.
    initial_state : np.ndarray
        Array of shape (3,) holding the initial normalised density, velocity
        and pressure perturbations, in that order. Real input is accepted.
    time_step : float
        Time increment between successive levels in inverse plasma
        frequencies, strictly positive.
    step_count : int
        Number of increments to take, at least one.

    Returns
    -------
    moment_history : np.ndarray
        Complex array of shape (step_count + 1, 3) whose first row is
        ``initial_state`` and whose remaining rows are the successive levels.

    Raises
    ------
    ValueError
        If ``evolution_matrix`` is not a finite array of shape (3, 3), if
        ``initial_state`` is not a finite array of shape (3,), if ``time_step``
        is not a finite number greater than zero, or if ``step_count`` is not an
        integer greater than zero.
    """
    return moment_history  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_integrate_moment_history(evolution_matrix: np.ndarray,
                                     initial_state: np.ndarray,
                                     time_step: float,
                                     step_count: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    matrix = np.asarray(evolution_matrix, dtype=complex)
    if matrix.shape != (3, 3):
        raise ValueError("evolution_matrix must be an array of shape (3, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("evolution_matrix must contain only finite entries")

    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (3,):
        raise ValueError("initial_state must be an array of shape (3,)")
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite entries")

    if not (isinstance(time_step, (int, float, np.floating, np.integer))
            and not isinstance(time_step, bool) and np.isfinite(time_step)
            and float(time_step) > 0.0):
        raise ValueError("time_step must be a finite number greater than zero")
    time_step = float(time_step)

    if (isinstance(step_count, bool)
            or not isinstance(step_count, (int, np.integer))
            or int(step_count) < 1):
        raise ValueError("step_count must be an integer greater than zero")
    step_count = int(step_count)

    history = np.empty((step_count + 1, 3), dtype=complex)
    history[0] = state
    for level in range(step_count):
        trial = matrix @ state
        state = state + time_step * (matrix @ (state + 0.5 * time_step * trial))
        history[level + 1] = state
    return history

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the wave-number-dependent closure at the benchmark wave
        #     number, run for a short window (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
evolution_matrix = np.array([
    [0.0 + 0.0j, -0.565685424949j, 0.0 + 0.0j],
    [-1.767766952966j, 0.0 + 0.0j, -0.282842712475j],
    [0.441724628000 + 0.0j, -1.231670049000j, -0.441724628000 + 0.0j]])
initial_state = np.array([0.02, 0.0, 0.02])
time_step = 0.005
step_count = 200
""",
            "call": "sig(integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 100.0)",
            "gold_call": "sig(_oracle_integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 100.0)",
        },
        # --- Valid: a purely real decaying operator with a coarse step, where the
        #     second-order truncation error is visible ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
evolution_matrix = np.array([[-0.5, 0.2, 0.0],
                             [0.0, -0.3, 0.1],
                             [0.4, 0.0, -0.9]], dtype=complex)
initial_state = np.array([1.0, -2.0, 0.5])
time_step = 0.25
step_count = 40
""",
            "call": "sig(integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 10.0)",
            "gold_call": "sig(_oracle_integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 10.0)",
        },
        # --- Boundary: a single step, which returns only the initial level and
        #     one advance ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
evolution_matrix = np.array([[0.0, -1.0j, 0.0],
                             [-1.0j, 0.0, -0.5j],
                             [0.3, -2.0j, -0.3]], dtype=complex)
initial_state = np.array([0.02, 0.0, 0.02])
time_step = 0.005
step_count = 1
""",
            "call": "sig(integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 1.0)",
            "gold_call": "sig(_oracle_integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 1.0)",
        },
        # --- Edge: a purely oscillatory operator with no damping anywhere in its
        #     spectrum, where the exact solution neither grows nor decays, so any
        #     dissipative or amplifying bias in the integrator shows up as drift
        #     in the recorded amplitude ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
evolution_matrix = np.array([[0.0 + 0.0j, -1.0j, 0.0 + 0.0j],
                             [-1.0j, 0.0 + 0.0j, -0.5j],
                             [0.0 + 0.0j, -2.0j, 0.0 + 0.0j]], dtype=complex)
initial_state = np.array([0.02, -0.01, 0.03])
time_step = 0.005
step_count = 50
""",
            "call": "sig(integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 1.0)",
            "gold_call": "sig(_oracle_integrate_moment_history(evolution_matrix, initial_state, time_step, step_count), 1.0)",
        },
        # --- Invalid: a non-positive number of steps ---
        {
            "setup": """import numpy as np
evolution_matrix = np.zeros((3, 3), dtype=complex)
initial_state = np.array([0.02, 0.0, 0.02])
def run_model():
    try:
        integrate_moment_history(evolution_matrix, initial_state, 0.005, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_moment_history(evolution_matrix, initial_state, 0.005, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive time step ---
        {
            "setup": """import numpy as np
evolution_matrix = np.zeros((3, 3), dtype=complex)
initial_state = np.array([0.02, 0.0, 0.02])
def run_model():
    try:
        integrate_moment_history(evolution_matrix, initial_state, -0.005, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_moment_history(evolution_matrix, initial_state, -0.005, 10)
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
