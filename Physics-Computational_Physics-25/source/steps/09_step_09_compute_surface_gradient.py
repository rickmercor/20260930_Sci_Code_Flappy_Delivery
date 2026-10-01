"""
Measure the steepness of the thermal front in the uppermost stratum as the largest magnitude of the depth derivative of temperature over the centroids of that stratum's elements.

The depth derivative of temperature just below a heated surface is the quantity an engineer reads off a temperature contour plot and uses as a proxy for thermal severity, and it is also, through Fourier's law, the local heat flux divided by the conductivity. Two thermal properties of the surface stratum bear on it and they need not act in the same direction: the conductivity divides the flux, so a more conductive layer carries the same heat at a milder gradient, while the diffusivity fixes how deep the front has run and therefore how warm the surface has become and how much flux the film still delivers. Which of the two dominates is a property of the configuration and is not to be assumed. The elastic properties of the stratum play no part whatever in this number, which is precisely why comparing its variation with the variation of peak stress is a meaningful test of whether stress can be inferred from a thermal picture.




Evaluating the derivative at element centroids rather than at nodes matters for the same reason it mattered for stress. Within a trilinear element the temperature gradient is a low-order function recovered by differentiating the interpolation, and it is discontinuous across element faces, so a nodal value would depend on which neighbouring elements were averaged and with what weights. The centroid value is unambiguous and, for the vertical direction of a regular mesh, reduces to the difference between the mean temperature of the element's upper face and that of its lower face divided by the element thickness. Restricting the search to the uppermost stratum keeps the measurement about the surface layer and prevents it from being captured by a strong internal gradient at a deeper material interface, where a jump in conductivity forces a jump in the gradient to keep the flux continuous.




Taking the largest magnitude over that stratum rather than a value at one nominated point makes the measure independent of where the heated patch happens to sit on the mesh, and picks up the same place in every configuration, namely directly beneath the heated region where the front is steepest. It does inherit a resolution dependence: the near-surface temperature profile is a boundary layer whose thickness shrinks as the front is young, so a mesh that does not resolve it reports a gradient averaged over the element rather than the true surface value. That is a property of the discretisation and not of the stratification, and it cancels out of a comparison between stratifications computed on the same mesh, which is how the measure is used.

Returns
-------
float: the largest magnitude of the depth derivative of temperature over the centroids of the uppermost stratum's elements, in K/m, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def compute_surface_gradient(width: float, depth: float, n_lateral: int, n_depth: int,
                             temperature: np.ndarray) -> float:
    """Measure the steepest depth gradient of temperature in the uppermost stratum.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    temperature : np.ndarray
        Vector of length n_nodes holding the nodal temperatures.

    Returns
    -------
    gradient : float
        Largest magnitude of the depth derivative of temperature over the
        centroids of the uppermost stratum's elements, in K m^-1, as a native
        Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return gradient  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_compute_surface_gradient(width: float, depth: float, n_lateral: int, n_depth: int,
                                     temperature: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    _HEX_SIGNS = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                           [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])

    def _node_id(i, j, k, n_lateral):
        """Return the global index of the node at grid position (i, j, k)."""
        return i + (n_lateral + 1) * (j + (n_lateral + 1) * k)

    def _hex_shape(r, s, t):
        """Return shape functions and natural derivatives of the trilinear brick."""
        sr, ss, st = _HEX_SIGNS[:, 0], _HEX_SIGNS[:, 1], _HEX_SIGNS[:, 2]
        shape = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * (1.0 + st * t)
        grad = np.empty((3, 8))
        grad[0] = 0.125 * sr * (1.0 + ss * s) * (1.0 + st * t)
        grad[1] = 0.125 * (1.0 + sr * r) * ss * (1.0 + st * t)
        grad[2] = 0.125 * (1.0 + sr * r) * (1.0 + ss * s) * st
        return shape, grad

    def _build_mesh(width, depth, n_lateral, n_depth):
        """Return nodes, element connectivity and the stratum index of each element."""
        lateral = np.linspace(0.0, width, n_lateral + 1)
        vertical = np.linspace(0.0, depth, n_depth + 1)
        nodes = np.empty(((n_lateral + 1) ** 2 * (n_depth + 1), 3))
        for k in range(n_depth + 1):
            for j in range(n_lateral + 1):
                for i in range(n_lateral + 1):
                    nodes[_node_id(i, j, k, n_lateral)] = (lateral[i], lateral[j], vertical[k])
        elements = []
        for k in range(n_depth):
            for j in range(n_lateral):
                for i in range(n_lateral):
                    elements.append([
                        _node_id(i, j, k, n_lateral), _node_id(i + 1, j, k, n_lateral),
                        _node_id(i + 1, j + 1, k, n_lateral), _node_id(i, j + 1, k, n_lateral),
                        _node_id(i, j, k + 1, n_lateral), _node_id(i + 1, j, k + 1, n_lateral),
                        _node_id(i + 1, j + 1, k + 1, n_lateral), _node_id(i, j + 1, k + 1, n_lateral)])
        elements = np.array(elements, dtype=int)
        centre_depth = nodes[elements][:, :, 2].mean(axis=1)
        stratum = np.minimum((centre_depth / (depth / 4.0)).astype(int), 3)
        return nodes, elements, stratum

    def _validate_mesh_arguments(width, depth, n_lateral, n_depth):
        """Raise ValueError when the near-field geometry or discretisation is inadmissible."""
        for name, value in (("width", width), ("depth", depth)):
            if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
                raise ValueError(f"{name} must be a finite number > 0")
        for name, value in (("n_lateral", n_lateral), ("n_depth", n_depth)):
            if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                    and int(value) >= 1):
                raise ValueError(f"{name} must be an integer >= 1")
        if int(n_depth) % 4 != 0:
            raise ValueError("n_depth must be a multiple of 4 so that every element lies in one stratum")

    _validate_mesh_arguments(width, depth, n_lateral, n_depth)

    width = float(width)
    depth = float(depth)
    n_lateral = int(n_lateral)
    n_depth = int(n_depth)

    nodes, elements, stratum = _build_mesh(width, depth, n_lateral, n_depth)
    field = np.asarray(temperature, dtype=float)
    if field.ndim != 1 or field.shape[0] != nodes.shape[0]:
        raise ValueError("temperature must be a vector with one entry per mesh node")
    if not np.all(np.isfinite(field)):
        raise ValueError("temperature must contain only finite entries")

    gradient = 0.0
    _, natural = _hex_shape(0.0, 0.0, 0.0)
    for index, connectivity in enumerate(elements):
        if stratum[index] != 0:
            continue
        cartesian = np.linalg.solve(natural @ nodes[connectivity], natural)
        gradient = max(gradient, abs(float((cartesian @ field[connectivity])[2])))
    return float(gradient)

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the step itself returns one, and the
    # invalid cases return a status code.
    return [
        # --- Valid: benchmark mesh with a decaying depth profile (normal scenario) ---
        {
            "setup": """import numpy as np
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 8
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = 70.0 * np.exp(-levels / 12.0)

def digest(value):
    return float(1.0 + 1.0e6 * float(value))
""",
            "call": "digest(compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Valid: laterally varying field so the steepest element is not unique by depth ---
        {
            "setup": """import numpy as np
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 4
lateral = np.linspace(0.0, width, n_lateral + 1)
gx, gy = np.meshgrid(lateral, lateral, indexing='ij')
plane = np.exp(-(((gx - 60.0) ** 2 + (gy - 60.0) ** 2) / 2500.0)).T.ravel()
levels = np.linspace(0.0, depth, n_depth + 1)
temperature = np.concatenate([80.0 * plane * np.exp(-z / 9.0) for z in levels])

def digest(value):
    return float(1.0 + 1.0e6 * float(value))
""",
            "call": "digest(compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Boundary: the coarsest admissible mesh, so the uppermost stratum
        #     holds a single element and the search over it has one candidate ---
        {
            "setup": """import numpy as np
width, depth, n_lateral, n_depth = 120.0, 40.0, 1, 4
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = 60.0 * np.exp(-levels / 15.0)

def digest(value):
    return float(1.0 + 1.0e6 * float(value))
""",
            "call": "digest(compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Edge: a buried hot zone whose own steepest gradient lies well below
        #     the uppermost stratum, so the answer is the milder surface gradient
        #     of the decaying tail rather than the largest gradient in the box ---
        {
            "setup": """import numpy as np
width, depth, n_lateral, n_depth = 120.0, 40.0, 4, 8
levels = np.repeat(np.linspace(0.0, depth, n_depth + 1), (n_lateral + 1) ** 2)
temperature = (50.0 * np.exp(-((levels - 25.0) ** 2) / 50.0)
               + 30.0 * np.exp(-levels / 25.0))

def digest(value):
    return float(1.0 + 1.0e6 * float(value))
""",
            "call": "digest(compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
            "gold_call": "digest(_oracle_compute_surface_gradient(width, depth, n_lateral, n_depth, temperature))",
        },
        # --- Invalid: temperature vector of the wrong length ---
        {
            "setup": """import numpy as np
bad = np.zeros(11)
def run_model():
    try:
        compute_surface_gradient(120.0, 40.0, 4, 4, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_surface_gradient(120.0, 40.0, 4, 4, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: depth discretisation incompatible with four strata ---
        {
            "setup": """import numpy as np
field = np.zeros(5 ** 2 * 8)
def run_model():
    try:
        compute_surface_gradient(120.0, 40.0, 4, 7, field)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_surface_gradient(120.0, 40.0, 4, 7, field)
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
