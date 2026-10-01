"""
With the baseline frequencies already in hand, the same cell is now struck and watched. A brief traction pulse is applied to one end face of the bar and the displacement of a single interior point is written down for a long time. A periodic solver admits no boundary condition, so the traction cannot be prescribed at a face; it enters as a singular body force, the traction vector multiplied by the discrete surface delta function of the loaded face. On the grid that delta is the reciprocal of the voxel size on the one specimen voxel layer adjacent to the face and vanishes everywhere else. The excitation reaches the equations nowhere else.

The pulse is two Gaussians of equal width, their centres one width apart, the later subtracted from the earlier. That construction gives it zero time integral, so the bar picks up no appreciable net velocity, and it spreads energy across a band wide enough to set every resonance of interest ringing.

The march is the implicit Newmark scheme. One step runs as follows: form the predictor from the current displacement, velocity and acceleration; solve for the new displacement with the routine of the previous stage, the right-hand side being the Newmark coefficient times the body force plus the voxel_density times the predictor; recover the new acceleration as the new displacement minus the predictor, divided by the Newmark coefficient; and update the velocity from the old and new accelerations weighted by the second Newmark parameter. Displacement, velocity and acceleration all start at zero, the body force of a step is sampled at the end time of that step, and the receptor displacement is written down after each step, the state at time zero included. Should any implicit solve miss the requested relative residual the march must stop, because that displacement would enter the next predictor and spoil the whole record; the routine raises a RuntimeError instead.

Returns
-------
dict, the receptor displacement record of the pulsed bar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def march_pulsed_bar(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    time_step: float,
    n_steps: int,
    newmark_beta: float,
    newmark_gamma: float,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    struck_voxel: int,
    receptor_voxel: int,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Strike the embedded bar and write down the receptor displacement step by step.

    Parameters
    ----------
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero at every voxel.
    voxel_edge : float
        Edge of one voxel in metre.
    time_step : float
        Length of one time step in second.
    n_steps : int
        How many time steps to take.
    newmark_beta : float
        The first Newmark parameter, somewhere in (0, 1].
    newmark_gamma : float
        The second Newmark parameter, somewhere in (0, 1].
    traction_amplitude : float
        Height of the traction pulse in pascal.
    gaussian_width : float
        Width of one Gaussian in second.
    gaussian_centre : float
        Where the earlier Gaussian is centred, in second.
    traction_axis : np.ndarray
        Aim of the traction, a non-zero vector of three entries, normalised inside the routine.
    struck_voxel : int
        The specimen voxel that touches the struck face.
    receptor_voxel : int
        The voxel whose displacement is written down.
    residual_tolerance : float
        Relative residual demanded of the linear solver.
    iteration_ceiling : int
        Ceiling on the linear solver iteration count.

    Returns
    -------
    dict
        Under the keys record, displacement, velocity, acceleration and mean_iterations.
        record : np.ndarray of shape (n_steps + 1, 3), float64. Row t holds the three
        displacement components of the receptor voxel after t steps; row 0 is the rest state.
        displacement, velocity, acceleration : np.ndarray of shape (3, n_total), float64, the
        state of every voxel after the last step, one row per component like face_modulus.
        mean_iterations : float, linear-solver iterations averaged over the steps.

    Raises
    ------
    ValueError
        When face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,) or fails to stay above zero, when n_total comes out even, when voxel_edge, time_step or gaussian_width fails to sit above zero, when n_steps is not an integer of one or more, when a Newmark parameter falls outside (0, 1], when traction_axis is not a non-zero vector of three entries, or when struck_voxel or receptor_voxel is not an integer addressing a voxel of the cell.
    RuntimeError
        When the implicit solve of a step misses the requested relative residual.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _counted(value, label):
    """Return a count as a native int once it is known to be an integer of one or more."""
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError("%s wants an integer of one or more" % label)
    return int(value)


def _gaussian_doublet(time, amplitude, width, centre):
    """Two Gaussians of one width, their centres one width apart, the later subtracted from the earlier."""
    leading = np.exp(-(time - centre) ** 2 / (2.0 * width ** 2))
    trailing = np.exp(-(time - centre - width) ** 2 / (2.0 * width ** 2))
    return amplitude * (leading - trailing)


def _voxel_index(value, label, n_total):
    """Return an integer voxel index once it is known to address a voxel of the cell."""
    if not isinstance(value, (int, np.integer)):
        raise ValueError("%s wants an integer" % label)
    index = int(value)
    if not 0 <= index < n_total:
        raise ValueError("%s falls outside the cell" % label)
    return index


def _oracle_march_pulsed_bar(
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    time_step: float,
    n_steps: int,
    newmark_beta: float,
    newmark_gamma: float,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    struck_voxel: int,
    receptor_voxel: int,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation."""
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if links.ndim != 2 or links.shape[0] != 3:
        raise ValueError("face_modulus wants three rows, one per displacement component")
    n_total = links.shape[1]
    if mass.shape != (n_total,) or mass.min() <= 0.0:
        raise ValueError("density wants one entry per voxel, every one of them above zero")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    increment = float(time_step)
    width = float(gaussian_width)
    if min(edge, increment, width) <= 0.0:
        raise ValueError("voxel_size, time_step and gaussian_width all want values above zero")
    steps = _counted(n_steps, "n_steps")
    beta = float(newmark_beta)
    gamma = float(newmark_gamma)
    if not 0.0 < beta <= 1.0 or not 0.0 < gamma <= 1.0:
        raise ValueError("both Newmark parameters want a value in (0, 1]")
    aim = np.asarray(traction_axis, dtype=float)
    if aim.shape != (3,) or np.linalg.norm(aim) == 0.0:
        raise ValueError("pulse_direction wants a non-zero vector of three entries")
    source = _voxel_index(struck_voxel, "loaded_voxel", n_total)
    probe = _voxel_index(receptor_voxel, "receptor_voxel", n_total)

    unit = aim / np.linalg.norm(aim)
    coefficient = beta * increment * increment
    lag = increment * increment * (0.5 - beta)
    amplitude = float(traction_amplitude)
    centre = float(gaussian_centre)
    target = float(residual_tolerance)

    displacement = np.zeros((3, n_total))
    velocity = np.zeros((3, n_total))
    acceleration = np.zeros((3, n_total))
    body_force = np.zeros((3, n_total))
    history = np.zeros((steps + 1, 3))
    spent = 0
    for step in range(1, steps + 1):
        forecast = displacement + increment * velocity + lag * acceleration
        traction = _gaussian_doublet(step * increment, amplitude, width, centre)
        body_force[:, source] = traction * unit / edge
        outcome = _oracle_invert_implicit_step(  # noqa: F821
            coefficient * body_force + mass * forecast, links, mass, edge, coefficient,
            target, iteration_ceiling,
        )
        if float(outcome["residual"]) > target:
            raise RuntimeError("step %d left the implicit system unconverged" % step)
        spent += outcome["iterations"]
        advanced = outcome["solution"]
        rate = (advanced - forecast) / coefficient
        velocity = velocity + increment * ((1.0 - gamma) * acceleration + gamma * rate)
        displacement = advanced
        acceleration = rate
        history[step] = displacement[:, probe]
    return {
        "record": history,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "mean_iterations": spent / steps,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
LINKS = np.zeros((3, 33))
LINKS[:, :28] = 1.0e11
MASS = np.full(33, 4506.3)
def verdict(fn, loaded=0, receptor=33):
    try:
        fn(LINKS, MASS, 2.5e-5, 2.0e-9, 10, 0.25, 0.5, 1.0e6, 2.0e-8, 8.0e-8,
           np.array([1.0, 0.0, 0.0]), loaded, receptor, 1e-10, 100)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": ("(verdict(march_pulsed_bar), "
                     "verdict(march_pulsed_bar, loaded=-0.5, receptor=10), "
                     "verdict(march_pulsed_bar, loaded=0, receptor=1.5), "
                     "verdict(march_pulsed_bar, loaded=0, receptor=10))"),
            "gold_call": ("(verdict(_oracle_march_pulsed_bar), "
                          "verdict(_oracle_march_pulsed_bar, loaded=-0.5, receptor=10), "
                          "verdict(_oracle_march_pulsed_bar, loaded=0, receptor=1.5), "
                          "verdict(_oracle_march_pulsed_bar, loaded=0, receptor=10))"),
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
draw = np.random.default_rng(5)
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = draw.uniform(1.3e11, 2.7e11, BODY)
LINKS[1, :BODY] = draw.uniform(3.6e10, 5.5e10, BODY)
LINKS[2] = LINKS[1]
MASS = np.full(COUNT, 4506.3)
AIM = np.array([1.0, 0.0, 0.0])
def digest(out):
    trace = out["record"]
    scale = float(np.abs(trace).max())
    # the closing entries are absolute displacements in picometres, so a trace rescaled by any
    # factor, or a solver reporting a normalised history, cannot reproduce them
    return (trace.shape, int(np.all(trace[0] == 0.0)),
            int(np.all(trace[:, 1] == 0.0)), int(np.all(trace[:, 2] == 0.0)),
            int(out["mean_iterations"] < 30),
            round(float(trace[600, 0] / scale), 7), round(float(trace[900, 0] / scale), 7),
            round(float(scale * 1e12), 6), round(float(trace[900, 0] * 1e12), 6),
            round(float(out["displacement"][0, 0] * 1e12), 6))
""",
            "call": ("digest(march_pulsed_bar(LINKS, MASS, EDGE, 2.0e-9, 1000, 0.25, 0.5, 1.0e6, "
                     "2.0e-8, 8.0e-8, AIM, 0, BODY - 1, 1e-10, 500))"),
            "gold_call": ("digest(_oracle_march_pulsed_bar(LINKS, MASS, EDGE, 2.0e-9, 1000, 0.25, 0.5, 1.0e6, "
                          "2.0e-8, 8.0e-8, AIM, 0, BODY - 1, 1e-10, 500))"),
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = 1.77e11
LINKS[1, :BODY] = 4.1e10
LINKS[2, :BODY] = 4.1e10
MASS = np.full(COUNT, 4506.3)
AIM = np.array([0.0, 3.0, 4.0])
STEP, TAKEN = 2.0e-9, 1400
def traction(t):
    return 1.0e6 * (np.exp(-(t - 8.0e-8) ** 2 / (2 * 4.0e-16)) - np.exp(-(t - 1.0e-7) ** 2 / (2 * 4.0e-16)))
def momentum_digest(out):
    # zero initial acceleration plus the trapezoidal Newmark rule makes the total momentum the
    # trapezoidal sum of the traction across the steps, the value at time zero left out
    sampled = traction(np.arange(TAKEN + 1) * STEP)
    sampled[0] = 0.0
    impulse = STEP * (sampled.sum() - 0.5 * sampled[0] - 0.5 * sampled[-1])
    carried = (MASS * out["velocity"]).sum(axis=1) * EDGE
    predicted = impulse * np.array([0.0, 0.6, 0.8])
    trace = out["record"]
    scale = float(np.abs(trace).max())
    landing = int(np.argmax(np.abs(trace[:, 2])))
    # the two transverse components share one modulus and are driven in the ratio 0.6 to 0.8, so
    # the combination below is round-off at the largest recorded displacement
    return (int(np.all(trace[:, 0] == 0.0)), landing,
            round(float(np.abs(carried - predicted).max() / np.abs(predicted).max()), 8),
            round(float((trace[landing, 1] - 0.75 * trace[landing, 2]) / scale), 10),
            round(float(trace[landing, 2] / scale), 7), round(float(scale * 1e12), 6))
""",
            "call": ("momentum_digest(march_pulsed_bar(LINKS, MASS, EDGE, STEP, TAKEN, 0.25, 0.5, "
                     "1.0e6, 2.0e-8, 8.0e-8, AIM, 0, BODY - 1, 1e-12, 500))"),
            "gold_call": ("momentum_digest(_oracle_march_pulsed_bar(LINKS, MASS, EDGE, STEP, TAKEN, 0.25, 0.5, "
                          "1.0e6, 2.0e-8, 8.0e-8, AIM, 0, BODY - 1, 1e-12, 500))"),
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
draw = np.random.default_rng(19)
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = draw.uniform(1.3e11, 2.7e11, BODY)
LINKS[1, :BODY] = draw.uniform(3.6e10, 5.5e10, BODY)
LINKS[2] = LINKS[1]
MASS = np.full(COUNT, 4506.3)
AIM = np.array([1.0, 0.0, 0.0])
def under_cap(fn, ceiling):
    # one conjugate gradient iteration cannot reach 1e-12 here, so no step may be advanced
    try:
        out = fn(LINKS, MASS, EDGE, 2.0e-9, 60, 0.25, 0.5, 1.0e6, 2.0e-8, 8.0e-8, AIM, 0,
                 BODY - 1, 1e-12, ceiling)
    except RuntimeError:
        return (1, 0.0)
    except ValueError:
        return (2, 0.0)
    return (0, round(float(np.abs(out["displacement"][:, 0]).max() * 1e12), 6))
""",
            "call": "(under_cap(march_pulsed_bar, 1), under_cap(march_pulsed_bar, 500))",
            "gold_call": "(under_cap(_oracle_march_pulsed_bar, 1), under_cap(_oracle_march_pulsed_bar, 500))",
        },
    ]
