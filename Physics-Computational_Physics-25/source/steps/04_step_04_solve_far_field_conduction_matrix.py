"""
Solve the matrix quadratic of the scaling-surface far field for the steady-state conduction matrix that condenses the whole exterior onto the interface.

Once the exterior has been mapped onto a surface mesh and a radial coordinate, its semi-analytical equation is a second-order ordinary differential system in that coordinate whose coefficients are the three matrices of the previous step. The object wanted is not the temperature field in the exterior but the linear relation between the interface temperatures and the heat flowing outwards through the interface. Substituting that relation into the radial system removes the temperatures entirely and leaves a matrix quadratic for the relation itself, in which the radial-derivative matrix appears inverted between two copies of the sought operator shifted by the cross matrix, the circumferential matrix appears as an additive term, and the rescaling of the previous step contributes a further linear term in the sought operator. Reaching that quadratic is the whole point of the semi-analytical treatment: it converts an unbounded domain into a dense matrix of the size of the interface, with no elements outside the interface at all.

The quadratic has more than one solution and only one is physical. Writing the radial system in first-order form as a state equation for the pair of interface temperatures and interface fluxes produces a matrix whose eigenvalues come in pairs of equal magnitude and opposite sign, an antisymmetry inherited from the self-adjointness of conduction. Modes with positive exponents grow as the radial coordinate increases and represent energy arriving from infinity; modes with negative exponents decay and represent energy leaving. For a bounded domain, the growing set is the admissible one; for an unbounded domain, it is the decaying set. Assembling the eigenvector blocks of the chosen half and taking the flux block multiplied by the inverse of the temperature block yields the sought operator. Choosing the wrong half returns a matrix that satisfies the same quadratic exactly and is therefore invisible to any residual check, but is negative semi-definite instead of positive definite, so the far field would feed heat into the near field rather than absorb it, and the assembled system would lose the definiteness on which the time integration depends.

Two structural properties provide the checks. The operator must be symmetric, because the exterior problem is self-adjoint and asymmetry can only come from the eigenvector algebra; small asymmetry from finite-precision eigen-solution is legitimately removed by averaging with the transpose. And the operator must be positive definite, with the sum of all its entries equal to the total steady conductance from a uniformly heated interface out to infinity. This quantity can be estimated independently from the geometry of the exterior. A degenerate case worth understanding is a prismatic exterior, one whose two surfaces are congruent so that the rescaling factor vanishes: there the operator becomes singular, correctly reflecting that a prism of constant cross-section extending to infinity carries no steady heat at all, and only a diverging exterior gives a nonsingular steady conductance.

Returns
-------
np.ndarray of shape (m, m), float: the symmetric steady-state far-field conduction matrix on the interface, in W/K.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_far_field_conduction_matrix(radial_matrix: np.ndarray, cross_matrix: np.ndarray,
                                      circumferential_matrix: np.ndarray) -> np.ndarray:
    """Solve for the steady-state conduction matrix of the unbounded far field.

    Parameters
    ----------
    radial_matrix : np.ndarray
        Rescaled radial-derivative coefficient matrix of shape (m, m),
        symmetric positive definite.
    cross_matrix : np.ndarray
        Cross coefficient matrix of shape (m, m); it is not symmetric.
    circumferential_matrix : np.ndarray
        Rescaled circumferential coefficient matrix of shape (m, m),
        symmetric positive semi-definite.

    Returns
    -------
    conduction : np.ndarray
        Symmetric positive definite matrix of shape (m, m) relating the
        interface temperatures to the steady heat flow into the far field,
        in W K^-1.

    Raises
    ------
    ValueError
        Admissibility of the coefficients is the routine's duty to check
        rather than a property it may assume. The argument descriptions
        above are therefore requirements, and this is raised if any matrix
        is not square, if the three do not share one order, if any entry is
        not finite, if ``radial_matrix`` is singular or is not positive
        definite, or if the coefficients admit no decaying half-spectrum.
    """
    return conduction  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_solve_far_field_conduction_matrix(radial_matrix: np.ndarray,
                                              cross_matrix: np.ndarray,
                                              circumferential_matrix: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _validate_coefficient_block(name, matrix, order):
        """Raise ValueError unless matrix is a finite square array of the given order."""
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(f"{name} must be a square two-dimensional array")
        if order is not None and array.shape[0] != order:
            raise ValueError("the coefficient matrices must all have the same order")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        return array

    radial = _validate_coefficient_block("radial_matrix", radial_matrix, None)
    order = radial.shape[0]
    cross = _validate_coefficient_block("cross_matrix", cross_matrix, order)
    circumferential = _validate_coefficient_block(
        "circumferential_matrix", circumferential_matrix, order)
    if np.linalg.matrix_rank(radial) < order:
        raise ValueError("radial_matrix must be non-singular")
    if np.min(np.linalg.eigvalsh(0.5 * (radial + radial.T))) <= 0.0:
        raise ValueError("radial_matrix must be positive definite")

    inverse_radial = np.linalg.inv(radial)
    identity = np.eye(order)
    # State matrix of the first-order radial system for interface temperatures
    # and interface fluxes. The half-identity shifts split the linear term that
    # the characteristic-length rescaling contributes to the matrix quadratic.
    state = np.block([
        [-inverse_radial @ cross.T + 0.5 * identity, -inverse_radial],
        [cross @ inverse_radial @ cross.T - circumferential,
         cross @ inverse_radial - 0.5 * identity]])

    values, vectors = np.linalg.eig(state)
    # The exterior is unbounded, so the admissible modes are the decaying half.
    decaying = np.argsort(values.real)[:order]
    if np.max(values.real[decaying]) >= 0.0:
        raise ValueError("the coefficient matrices do not admit a decaying half-spectrum")

    temperature_block = vectors[:order, decaying]
    flux_block = vectors[order:, decaying]
    conduction = (flux_block @ np.linalg.inv(temperature_block)).real
    return 0.5 * (conduction + conduction.T)

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
    return [
        # --- Valid: diverging exterior of a small square interface (normal scenario) ---
        {
            "setup": """import numpy as np
def coefficients(width, depth, n_lateral, k_far, contraction, xi):
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    lateral = np.linspace(0.0, width, n_lateral + 1)
    count = (n_lateral + 1) ** 2
    boundary = np.empty((count, 3))
    for j in range(n_lateral + 1):
        for i in range(n_lateral + 1):
            boundary[i + (n_lateral + 1) * j] = (lateral[i], lateral[j], depth)
    scaling = boundary.copy()
    scaling[:, 0] = 0.5 * width + contraction * (boundary[:, 0] - 0.5 * width)
    scaling[:, 1] = 0.5 * width + contraction * (boundary[:, 1] - 0.5 * width)
    scaling[:, 2] = 0.0
    faces = [[i + (n_lateral + 1) * j, (i + 1) + (n_lateral + 1) * j,
              (i + 1) + (n_lateral + 1) * (j + 1), i + (n_lateral + 1) * (j + 1)]
             for j in range(n_lateral) for i in range(n_lateral)]
    e0 = np.zeros((count, count)); e1 = np.zeros((count, count)); e2 = np.zeros((count, count))
    for face in faces:
        offset = boundary[face] - scaling[face]
        for r in gp:
            for s in gp:
                shape = 0.25 * (1.0 + signs[:, 0] * r) * (1.0 + signs[:, 1] * s)
                nat = np.empty((2, 4))
                nat[0] = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * s)
                nat[1] = 0.25 * (1.0 + signs[:, 0] * r) * signs[:, 1]
                swept = scaling[face] + xi * offset
                jac = np.vstack([shape @ offset, nat[0] @ swept, nat[1] @ swept])
                det = np.linalg.det(jac); inv = np.linalg.inv(jac)
                b1 = np.outer(inv[:, 0], shape)
                b2 = np.outer(inv[:, 1], nat[0]) + np.outer(inv[:, 2], nat[1])
                e0[np.ix_(face, face)] += k_far * (b1.T @ b1) * det
                e1[np.ix_(face, face)] += k_far * (b2.T @ b1) * det
                e2[np.ix_(face, face)] += k_far * (b2.T @ b2) * det
    rb, rs = width, contraction * width
    a0 = (rb - rs) / (rs + (rb - rs) * xi)
    return a0 * e0, e1, e2 / a0

e0, e1, e2 = coefficients(120.0, 40.0, 2, 100.0, 0.5, 1.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_far_field_conduction_matrix(e0, e1, e2))",
            "gold_call": "digest(_oracle_solve_far_field_conduction_matrix(e0, e1, e2))",
        },
        # --- Valid: benchmark interface, four elements per side ---
        {
            "setup": """import numpy as np
def coefficients(width, depth, n_lateral, k_far, contraction, xi):
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    lateral = np.linspace(0.0, width, n_lateral + 1)
    count = (n_lateral + 1) ** 2
    boundary = np.empty((count, 3))
    for j in range(n_lateral + 1):
        for i in range(n_lateral + 1):
            boundary[i + (n_lateral + 1) * j] = (lateral[i], lateral[j], depth)
    scaling = boundary.copy()
    scaling[:, 0] = 0.5 * width + contraction * (boundary[:, 0] - 0.5 * width)
    scaling[:, 1] = 0.5 * width + contraction * (boundary[:, 1] - 0.5 * width)
    scaling[:, 2] = 0.0
    faces = [[i + (n_lateral + 1) * j, (i + 1) + (n_lateral + 1) * j,
              (i + 1) + (n_lateral + 1) * (j + 1), i + (n_lateral + 1) * (j + 1)]
             for j in range(n_lateral) for i in range(n_lateral)]
    e0 = np.zeros((count, count)); e1 = np.zeros((count, count)); e2 = np.zeros((count, count))
    for face in faces:
        offset = boundary[face] - scaling[face]
        for r in gp:
            for s in gp:
                shape = 0.25 * (1.0 + signs[:, 0] * r) * (1.0 + signs[:, 1] * s)
                nat = np.empty((2, 4))
                nat[0] = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * s)
                nat[1] = 0.25 * (1.0 + signs[:, 0] * r) * signs[:, 1]
                swept = scaling[face] + xi * offset
                jac = np.vstack([shape @ offset, nat[0] @ swept, nat[1] @ swept])
                det = np.linalg.det(jac); inv = np.linalg.inv(jac)
                b1 = np.outer(inv[:, 0], shape)
                b2 = np.outer(inv[:, 1], nat[0]) + np.outer(inv[:, 2], nat[1])
                e0[np.ix_(face, face)] += k_far * (b1.T @ b1) * det
                e1[np.ix_(face, face)] += k_far * (b2.T @ b1) * det
                e2[np.ix_(face, face)] += k_far * (b2.T @ b2) * det
    rb, rs = width, contraction * width
    a0 = (rb - rs) / (rs + (rb - rs) * xi)
    return a0 * e0, e1, e2 / a0

e0, e1, e2 = coefficients(120.0, 40.0, 4, 100.0, 0.5, 1.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_far_field_conduction_matrix(e0, e1, e2))",
            "gold_call": "digest(_oracle_solve_far_field_conduction_matrix(e0, e1, e2))",
        },
        # --- Boundary: single interface element, so the operator is four by four ---
        {
            "setup": """import numpy as np
def coefficients(width, depth, n_lateral, k_far, contraction, xi):
    signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    gp = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    lateral = np.linspace(0.0, width, n_lateral + 1)
    count = (n_lateral + 1) ** 2
    boundary = np.empty((count, 3))
    for j in range(n_lateral + 1):
        for i in range(n_lateral + 1):
            boundary[i + (n_lateral + 1) * j] = (lateral[i], lateral[j], depth)
    scaling = boundary.copy()
    scaling[:, 0] = 0.5 * width + contraction * (boundary[:, 0] - 0.5 * width)
    scaling[:, 1] = 0.5 * width + contraction * (boundary[:, 1] - 0.5 * width)
    scaling[:, 2] = 0.0
    faces = [[i + (n_lateral + 1) * j, (i + 1) + (n_lateral + 1) * j,
              (i + 1) + (n_lateral + 1) * (j + 1), i + (n_lateral + 1) * (j + 1)]
             for j in range(n_lateral) for i in range(n_lateral)]
    e0 = np.zeros((count, count)); e1 = np.zeros((count, count)); e2 = np.zeros((count, count))
    for face in faces:
        offset = boundary[face] - scaling[face]
        for r in gp:
            for s in gp:
                shape = 0.25 * (1.0 + signs[:, 0] * r) * (1.0 + signs[:, 1] * s)
                nat = np.empty((2, 4))
                nat[0] = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * s)
                nat[1] = 0.25 * (1.0 + signs[:, 0] * r) * signs[:, 1]
                swept = scaling[face] + xi * offset
                jac = np.vstack([shape @ offset, nat[0] @ swept, nat[1] @ swept])
                det = np.linalg.det(jac); inv = np.linalg.inv(jac)
                b1 = np.outer(inv[:, 0], shape)
                b2 = np.outer(inv[:, 1], nat[0]) + np.outer(inv[:, 2], nat[1])
                e0[np.ix_(face, face)] += k_far * (b1.T @ b1) * det
                e1[np.ix_(face, face)] += k_far * (b2.T @ b1) * det
                e2[np.ix_(face, face)] += k_far * (b2.T @ b2) * det
    rb, rs = width, contraction * width
    a0 = (rb - rs) / (rs + (rb - rs) * xi)
    return a0 * e0, e1, e2 / a0

e0, e1, e2 = coefficients(80.0, 20.0, 1, 50.0, 0.25, 1.0)

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_far_field_conduction_matrix(e0, e1, e2))",
            "gold_call": "digest(_oracle_solve_far_field_conduction_matrix(e0, e1, e2))",
        },
        # --- Edge: analytically diagonal coefficients with a single degree of freedom ---
        {
            "setup": """import numpy as np
e0 = np.array([[2.0]])
e1 = np.array([[0.75]])
e2 = np.array([[1.5]])

def digest(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "digest(solve_far_field_conduction_matrix(e0, e1, e2))",
            "gold_call": "digest(_oracle_solve_far_field_conduction_matrix(e0, e1, e2))",
        },
        # --- Invalid: mismatched coefficient orders ---
        {
            "setup": """import numpy as np
a = np.eye(3)
b = np.eye(2)
def run_model():
    try:
        solve_far_field_conduction_matrix(a, b, a)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_far_field_conduction_matrix(a, b, a)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: indefinite radial-derivative matrix ---
        {
            "setup": """import numpy as np
bad = np.array([[-1.0, 0.0], [0.0, 2.0]])
other = np.eye(2)
def run_model():
    try:
        solve_far_field_conduction_matrix(bad, other, other)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_far_field_conduction_matrix(bad, other, other)
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
