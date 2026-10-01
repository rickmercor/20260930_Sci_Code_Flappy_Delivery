"""
Every step of an implicit time march ends in the same linear system, and this stage both states that system and solves it. The operator is the voxel voxel_density on the diagonal plus a coefficient times the discrete stiffness operator of the padded cell, so a field $u$ is carried to $d u + c Q(u)$: $d$ is the voxel voxel_density, $c$ gathers the Newmark parameter with the square of the time step, and $Q(u)$ is the force the extended links exert back on the voxels. $Q$ is evaluated through discrete Fourier transforms, never assembled: extend every link with the symbol of the one-voxel forward difference, weight each link by its own modulus, and gather with the conjugate symbol, which pairs the difference with its own adjoint and leaves $Q$ symmetric and positive semi-definite.

Positive voxel_density everywhere makes the whole operator real, symmetric and positive definite, which is the setting the conjugate gradient method was written for, and each iteration reaches the operator only by applying it to the current search direction.

A preconditioner is needed because the padding leaves an enormous stiffness contrast. Take the same operator for a uniform reference medium carrying the volume-averaged link modulus and the volume-averaged voxel_density of the entire cell, padding included, and invert it exactly; that inverse is diagonal in the transformed domain, its multiplier at a given frequency being the reciprocal of the averaged voxel_density plus the coefficient times the averaged modulus times the squared magnitude of the difference symbol, applied to each component with the average of its own row. Since the preconditioned right-hand side already solves a uniform problem exactly, it makes the natural starting iterate. Iteration stops when the Euclidean norm of the residual across the three components drops below the residual_tolerance times the norm of the right-hand side.

Exhausting the iteration cap with the residual still above that threshold is a failure, not an answer. An unconverged displacement would be swallowed by the time integrator and would contaminate everything after it, so the routine raises a RuntimeError rather than handing back whichever iterate it happens to hold. The image of the accepted solution under the operator is returned alongside it, so that the residual can be checked without rebuilding the operator.

Returns
-------
dict, the implicit step solution with its operator image, iteration count and residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invert_implicit_step(
    rhs: np.ndarray,
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    newmark_coefficient: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Build the implicit elastodynamic operator and invert it by preconditioned conjugate gradients.

    Parameters
    ----------
    rhs : np.ndarray
        Right-hand side of the system, shape (3, n_total).
    face_modulus : np.ndarray
        Modulus of every link in pascal, one row per component, shape (3, n_total).
    voxel_density : np.ndarray
        Voxel voxel_density, shape (n_total,), above zero at every voxel.
    voxel_edge : float
        Edge of one voxel in metre.
    newmark_coefficient : float
        The Newmark parameter multiplied by the squared time step, not negative.
    residual_tolerance : float
        Relative residual at which the iteration is allowed to stop.
    iteration_ceiling : int
        Ceiling on the iteration count.

    Returns
    -------
    dict
        Under the keys solution, operator_image, iterations and residual.

    Raises
    ------
    ValueError
        When rhs or face_modulus is not shaped (3, n_total), when voxel_density is not shaped (n_total,) or fails to stay above zero, when n_total comes out even, when voxel_edge or residual_tolerance fails to sit above zero, when newmark_coefficient drops below zero, or when iteration_ceiling is not an integer of one or more.
    RuntimeError
        When the iteration ceiling is reached while the relative residual is still above the residual_tolerance.
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


def _link_difference_symbol(n_total, voxel_edge):
    """Transform multiplier that shifts a field by one voxel and subtracts, over an odd cell."""
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    modes = np.fft.fftfreq(n_total, d=1.0 / n_total)
    return (np.exp(2j * np.pi * modes / n_total) - 1.0) / voxel_edge


def _stiffness_action(field, face_row, symbol):
    """Extend every link, weight it by its own modulus, and gather the forces back onto the voxels."""
    extension = np.fft.ifft(symbol * np.fft.fft(field, axis=-1), axis=-1).real
    force = np.fft.fft(face_row * extension, axis=-1)
    return np.fft.ifft(np.conj(symbol) * force, axis=-1).real


def _reference_inverse(face_modulus, voxel_density, symbol, coefficient):
    """Per-component multiplier that inverts the operator of the volume-averaged reference medium."""
    row_average = face_modulus.mean(axis=1, keepdims=True)
    return 1.0 / (voxel_density.mean() + coefficient * row_average * np.abs(symbol) ** 2)


def _oracle_invert_implicit_step(
    rhs: np.ndarray,
    face_modulus: np.ndarray,
    voxel_density: np.ndarray,
    voxel_edge: float,
    newmark_coefficient: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation."""
    load = np.asarray(rhs, dtype=float)
    links = np.asarray(face_modulus, dtype=float)
    mass = np.asarray(voxel_density, dtype=float)
    if load.ndim != 2 or load.shape[0] != 3:
        raise ValueError("rhs wants three rows, one per displacement component")
    if links.shape != load.shape:
        raise ValueError("face_modulus wants the same shape as rhs")
    n_total = load.shape[1]
    if mass.shape != (n_total,) or mass.min() <= 0.0:
        raise ValueError("density wants one entry per voxel, every one of them above zero")
    if n_total % 2 == 0:
        raise ValueError("an even voxel count is outside this configuration")
    edge = float(voxel_edge)
    target = float(residual_tolerance)
    if edge <= 0.0 or target <= 0.0:
        raise ValueError("voxel_size and residual_tolerance both want values above zero")
    coefficient = float(newmark_coefficient)
    if coefficient < 0.0:
        raise ValueError("beta_dt2 must not be negative")
    cap = _counted(iteration_ceiling, "max_iterations")

    symbol = _link_difference_symbol(n_total, edge)
    multiplier = _reference_inverse(links, mass, symbol, coefficient)

    def _smooth(vector):
        return np.fft.ifft(multiplier * np.fft.fft(vector, axis=-1), axis=-1).real

    def _forward(vector):
        return mass * vector + coefficient * _stiffness_action(vector, links, symbol)

    scale = np.linalg.norm(load)
    if scale == 0.0:
        blank = np.zeros_like(load)
        return {"solution": blank, "operator_image": blank.copy(), "iterations": 0, "residual": 0.0}

    estimate = _smooth(load)
    image = _forward(estimate)
    defect = load - image
    smoothed = _smooth(defect)
    search = smoothed.copy()
    energy = float(np.sum(defect * smoothed))
    iterations = 0
    for _ in range(cap):
        if np.linalg.norm(defect) <= target * scale:
            break
        mapped = _forward(search)
        alpha = energy / float(np.sum(search * mapped))
        estimate = estimate + alpha * search
        image = image + alpha * mapped
        defect = defect - alpha * mapped
        smoothed = _smooth(defect)
        refreshed = float(np.sum(defect * smoothed))
        search = smoothed + (refreshed / energy) * search
        energy = refreshed
        iterations += 1

    achieved = float(np.linalg.norm(defect) / scale)
    if achieved > target:
        raise RuntimeError(
            "conjugate gradients exhausted the iteration ceiling at a relative residual of %g" % achieved
        )
    return {"solution": estimate, "operator_image": image, "iterations": iterations, "residual": achieved}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
COUNT, EDGE = 33, 1.0e-5
LINKS = np.ones((3, COUNT)) * np.array([[1.8e11], [4.1e10], [4.1e10]])
MASS = np.full(COUNT, 4506.3)
draw = np.random.default_rng(11)
LOAD = draw.standard_normal((3, COUNT))
def digest(out):
    # the preconditioner inverts a uniform medium exactly, so nothing is left to iterate on
    slip = float(np.linalg.norm(out["operator_image"] - LOAD) / np.linalg.norm(LOAD))
    return (int(out["iterations"] <= 1), int(out["residual"] < 1e-12), int(slip < 1e-12),
            round(float(out["solution"].sum() * 1e3), 6))
""",
            "call": "digest(invert_implicit_step(LOAD, LINKS, MASS, EDGE, 1e-19, 1e-10, 100))",
            "gold_call": "digest(_oracle_invert_implicit_step(LOAD, LINKS, MASS, EDGE, 1e-19, 1e-10, 100))",
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
draw = np.random.default_rng(3)
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = draw.uniform(1.3e11, 2.7e11, BODY)
LINKS[1, :BODY] = draw.uniform(3.6e10, 5.5e10, BODY)
LINKS[2] = LINKS[1]
MASS = np.full(COUNT, 4506.3)
LOAD = draw.standard_normal((3, COUNT)) * 1e-6
COEF = 0.25 * (2.0e-9) ** 2
def digest(out):
    # the returned image must reproduce the load, and a hand-built operator must reproduce the image
    symbol = (np.exp(2j * np.pi * np.fft.fftfreq(COUNT, d=1.0 / COUNT) / COUNT) - 1.0) / EDGE
    field = out["solution"]
    extension = np.fft.ifft(symbol * np.fft.fft(field, axis=-1), axis=-1).real
    hand = MASS * field + COEF * np.fft.ifft(np.conj(symbol) * np.fft.fft(LINKS * extension, axis=-1), axis=-1).real
    return (round(float(field.sum() * 1e6), 6), round(float(field[0, BODY - 1] * 1e9), 6),
            int(np.linalg.norm(out["operator_image"] - LOAD) / np.linalg.norm(LOAD) < 1e-9),
            int(float(np.abs(hand - out["operator_image"]).max() / np.abs(LOAD).max()) < 1e-9),
            int(out["iterations"] < 60))
""",
            "call": "digest(invert_implicit_step(LOAD, LINKS, MASS, EDGE, COEF, 1e-11, 500))",
            "gold_call": "digest(_oracle_invert_implicit_step(LOAD, LINKS, MASS, EDGE, COEF, 1e-11, 500))",
        },
        {
            "setup": """import numpy as np
MASS = np.full(33, 4506.3)
MASS[30:] = 0.0
def verdict(fn):
    try:
        fn(np.ones((3, 33)), np.full((3, 33), 1e11), MASS, 1e-5, 1e-19, 1e-10, 100)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "verdict(invert_implicit_step)",
            "gold_call": "verdict(_oracle_invert_implicit_step)",
        },
        {
            "setup": """import numpy as np
COUNT, BODY, EDGE = 141, 129, 5.0e-3 / 129
draw = np.random.default_rng(23)
LINKS = np.zeros((3, COUNT))
LINKS[0, :BODY] = draw.uniform(1.3e11, 2.7e11, BODY)
LINKS[1, :BODY] = draw.uniform(3.6e10, 5.5e10, BODY)
LINKS[2] = LINKS[1]
MASS = np.full(COUNT, 4506.3)
LOAD = draw.standard_normal((3, COUNT)) * 1e-6
COEF = 0.25 * (2.0e-9) ** 2
def under_cap(fn, ceiling):
    # one iteration cannot reach 1e-11 on this contrast, so nothing may be handed back
    try:
        out = fn(LOAD, LINKS, MASS, EDGE, COEF, 1e-11, ceiling)
    except RuntimeError:
        return (1, 0.0)
    except ValueError:
        return (2, 0.0)
    return (0, round(float(out["residual"]), 12))
""",
            "call": "(under_cap(invert_implicit_step, 1), under_cap(invert_implicit_step, 400))",
            "gold_call": "(under_cap(_oracle_invert_implicit_step, 1), under_cap(_oracle_invert_implicit_step, 400))",
        },
    ]
