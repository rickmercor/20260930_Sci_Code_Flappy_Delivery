"""
Assemble the three radial coefficient matrices of the unbounded far field from the scaling-surface coordinate map, frozen at a prescribed radial coordinate.

The far field is represented without discretising it, by mapping the whole exterior onto one surface mesh and one extra coordinate. Two geometrically similar surfaces are chosen, a scaling surface and a boundary surface, and every point of the exterior is written as the scaling-surface point plus the radial coordinate times the offset between corresponding points of the two surfaces; the boundary surface is recovered at radial coordinate one, and letting the coordinate grow without bound sweeps out the exterior. Classical formulations collapse the scaling surface to a single point, which makes the geometric Jacobian factor into a radial part times a surface part; keeping the scaling surface a surface destroys that factorisation, which is the price of being able to represent exteriors that no single point sees star-like, and the consequence is that the coefficient matrices below depend on the radial coordinate instead of being constants.

The physical gradient is expressed in the mapped coordinates through the inverse of that Jacobian, whose three columns are the gradients of the three mapped coordinates. Splitting the interpolated temperature gradient accordingly gives one operator that multiplies the radial derivative of the surface temperatures and one that multiplies the surface temperatures themselves, the first built from the radial gradient times the shape functions and the second from the two circumferential gradients times the circumferential shape-function derivatives. The three coefficient matrices are then the three distinct products of those two operators with the conductivity between them, integrated over the surface against the Jacobian determinant. The first is a mass-like matrix and positive definite; the third is a Laplacian-like matrix, positive semi-definite and singular on constant temperatures; the second is the cross term and is not symmetric, and it is the cross term that carries the geometric divergence of the exterior, so dropping it or symmetrizing it destroys the far field's ability to absorb a steady flux.

A characteristic length is attached to the mapping through the square roots of the two surface areas, interpolated linearly in the radial coordinate, and its logarithmic derivative provides the factor by which the three matrices are rescaled: the radial-derivative matrix is multiplied by it and the circumferential matrix divided by it. That rescaling is what converts the radial equation into the form whose steady limit involves the surface operator itself rather than only its derivative, and it is the reason a purely prismatic exterior, for which the factor vanishes, transmits no steady heat at all while a diverging one does. Because the matrices retain a radial dependence, a value of the radial coordinate must be nominated at which they are evaluated and thereafter held constant; the natural choice is the boundary surface itself, where the surface mesh actually lives.

Returns
-------
np.ndarray of shape (3, m, m), float: the rescaled far-field coefficient matrices E0, E1 and E2 of the scaling-surface formulation, frozen at the given radial coordinate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def assemble_scaling_surface_coefficients(width: float, depth: float, n_lateral: int,
                                          far_conductivity: float, contraction: float,
                                          radial_coordinate: float) -> np.ndarray:
    """Assemble the radial coefficient matrices of the scaling-surface far field.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth at which the boundary surface sits, in metres (depth > 0).
    n_lateral : int
        Number of boundary-surface elements along each lateral direction
        (n_lateral >= 1).
    far_conductivity : float
        Isotropic thermal conductivity of the far-field medium in
        W m^-1 K^-1 (far_conductivity > 0).
    contraction : float
        Ratio by which the boundary square is contracted about the vertical
        centre axis to form the scaling surface, 0 <= contraction < 1.
    radial_coordinate : float
        Radial coordinate at which the coefficient matrices are frozen
        (radial_coordinate > 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (3, m, m) with m = (n_lateral + 1) ** 2, holding the
        rescaled radial-derivative matrix, the cross matrix and the
        circumferential matrix in that order, in SI units.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the two
        surfaces do not define a diverging exterior.
    """
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_assemble_scaling_surface_coefficients(width: float, depth: float, n_lateral: int,
                                                  far_conductivity: float, contraction: float,
                                                  radial_coordinate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    _QUAD_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    _GAUSS_2PT = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    def _quad_shape(r, s):
        """Return shape functions and natural derivatives of the bilinear quadrilateral."""
        sr, ss = _QUAD_SIGNS[:, 0], _QUAD_SIGNS[:, 1]
        shape = 0.25 * (1.0 + sr * r) * (1.0 + ss * s)
        grad = np.empty((2, 4))
        grad[0] = 0.25 * sr * (1.0 + ss * s)
        grad[1] = 0.25 * (1.0 + sr * r) * ss
        return shape, grad

    def _surface_meshes(width, depth, n_lateral, contraction):
        """Return the boundary-surface nodes, the scaling-surface nodes and the faces.

        The boundary surface is the base of the near-field box at z = depth, carrying
        the quadrilaterals the near-field mesh induces there. The scaling surface is
        that same square contracted about the vertical axis through the box centre
        and laid on the free surface z = 0. Surface nodes are numbered i + (n+1) j.
        """
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
        faces = []
        for j in range(n_lateral):
            for i in range(n_lateral):
                faces.append([i + (n_lateral + 1) * j, (i + 1) + (n_lateral + 1) * j,
                              (i + 1) + (n_lateral + 1) * (j + 1), i + (n_lateral + 1) * (j + 1)])
        return boundary, scaling, np.array(faces, dtype=int)

    for name, value in (("width", width), ("depth", depth),
                        ("far_conductivity", far_conductivity),
                        ("radial_coordinate", radial_coordinate)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(n_lateral, (int, np.integer)) and not isinstance(n_lateral, bool)
            and int(n_lateral) >= 1):
        raise ValueError("n_lateral must be an integer >= 1")
    if not (isinstance(contraction, (int, float)) and np.isfinite(contraction)
            and 0.0 <= float(contraction) < 1.0):
        raise ValueError("contraction must be a finite number in the interval [0, 1)")

    width = float(width)
    depth = float(depth)
    n_lateral = int(n_lateral)
    far_conductivity = float(far_conductivity)
    contraction = float(contraction)
    radial_coordinate = float(radial_coordinate)

    boundary, scaling, faces = _surface_meshes(width, depth, n_lateral, contraction)
    count = boundary.shape[0]
    radial = np.zeros((count, count), dtype=float)
    cross = np.zeros((count, count), dtype=float)
    circumferential = np.zeros((count, count), dtype=float)

    for face in faces:
        offset = boundary[face] - scaling[face]
        for r in _GAUSS_2PT:
            for s in _GAUSS_2PT:
                shape, natural = _quad_shape(r, s)
                # Position of the swept surface at the frozen radial coordinate.
                swept = scaling[face] + radial_coordinate * offset
                jacobian = np.vstack([shape @ offset, natural[0] @ swept, natural[1] @ swept])
                detj = np.linalg.det(jacobian)
                inverse = np.linalg.inv(jacobian)
                # Columns of the inverse Jacobian are the gradients of the mapped
                # coordinates; the first drives the radial derivative and the other
                # two drive the circumferential derivatives.
                b_radial = np.outer(inverse[:, 0], shape)
                b_circ = (np.outer(inverse[:, 1], natural[0])
                          + np.outer(inverse[:, 2], natural[1]))
                radial[np.ix_(face, face)] += \
                    far_conductivity * (b_radial.T @ b_radial) * detj
                cross[np.ix_(face, face)] += \
                    far_conductivity * (b_circ.T @ b_radial) * detj
                circumferential[np.ix_(face, face)] += \
                    far_conductivity * (b_circ.T @ b_circ) * detj

    # Characteristic length of the mapping and its logarithmic radial derivative.
    root_boundary = np.sqrt(width ** 2)                    # square root of the boundary area
    root_scaling = np.sqrt((contraction * width) ** 2)     # square root of the scaling area
    length = root_scaling + (root_boundary - root_scaling) * radial_coordinate
    divergence = (root_boundary - root_scaling) / length
    if divergence <= 0.0:
        raise ValueError("the scaling and boundary surfaces must define a diverging exterior")

    return np.stack([divergence * radial, cross, circumferential / divergence])

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
        # --- Valid: benchmark far field frozen at the boundary surface (normal scenario) ---
        {
            "setup": """import numpy as np
width, depth, n_lateral = 120.0, 40.0, 4
far_conductivity, contraction, radial_coordinate = 100.0, 0.5, 1.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
            "gold_call": "digest(_oracle_assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
        },
        # --- Valid: softer far-field medium, frozen further out ---
        {
            "setup": """import numpy as np
width, depth, n_lateral = 120.0, 40.0, 4
far_conductivity, contraction, radial_coordinate = 50.0, 0.5, 4.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
            "gold_call": "digest(_oracle_assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
        },
        # --- Boundary: scaling surface collapsed to a point, the classical scaling centre ---
        {
            "setup": """import numpy as np
width, depth, n_lateral = 120.0, 40.0, 2
far_conductivity, contraction, radial_coordinate = 89.0, 0.0, 1.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
            "gold_call": "digest(_oracle_assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
        },
        # --- Edge: barely diverging exterior on a single element ---
        {
            "setup": """import numpy as np
width, depth, n_lateral = 60.0, 10.0, 1
far_conductivity, contraction, radial_coordinate = 66.0, 0.95, 1.0

def digest(values):
    # Each returned block is reduced on its own scale and raised to its own
    # power, so that a relative error in one block moves the result by that
    # relative amount however small that block is against the others. The
    # weight is an outer product of two different position functions, which
    # makes the reduction sensitive to a transposed or reordered block.
    total = 1.0
    for index, block in enumerate(np.asarray(values, dtype=float)):
        entries = 1.0e6 * np.asarray(block, dtype=float)
        weight = np.outer(np.arange(1.0, entries.shape[0] + 1.0),
                          np.sqrt(np.arange(1.0, entries.shape[1] + 1.0)))
        weight = weight / weight.sum()
        moments = np.sqrt(np.sum(weight * entries ** 2)) + 0.5 * np.sum(weight * entries)
        total *= (entries.size + moments) ** (index + 1.0)
    return float(total)
""",
            "call": "digest(assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
            "gold_call": "digest(_oracle_assemble_scaling_surface_coefficients(width, depth, n_lateral, far_conductivity, contraction, radial_coordinate))",
        },
        # --- Invalid: contraction outside its admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_scaling_surface_coefficients(120.0, 40.0, 4, 100.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_scaling_surface_coefficients(120.0, 40.0, 4, 100.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive far-field conductivity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        assemble_scaling_surface_coefficients(120.0, 40.0, 4, 0.0, 0.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_scaling_surface_coefficients(120.0, 40.0, 4, 0.0, 0.5, 1.0)
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
