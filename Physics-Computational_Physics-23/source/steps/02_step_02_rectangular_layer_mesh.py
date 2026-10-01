"""
Lay out the quadratic boundary elements of one rectangular layer together with the interior points at which that layer is sampled.

A boundary element model of a rectangular layer stores only its outline, discretised into three-node quadratic elements whose middle node sits at the midpoint of a straight segment, traversed anticlockwise so that the outward normal and the swept area of the layer both carry a positive sign. Interior sample points are placed at the centres of a uniform rectangular grid, which for an ultra-thin layer leaves every sample point a small fraction of an element length away from the long faces.

Returns
-------
tuple of two np.ndarray, float: the (n_elements, 3, 2) anticlockwise quadratic boundary elements and the (n_x * n_y, 2) interior sample points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rectangular_layer_mesh(length: float, y_bottom: float, y_top: float,
                           n_long: int, n_short: int, n_x: int, n_y: int) -> tuple:
    """Discretise the outline of one rectangular layer and sample its interior.

    Parameters
    ----------
    length : float
        Extent of the layer along x (``length > 0``); the layer occupies
        ``0 <= x <= length``.
    y_bottom : float
        Lower bound of the layer along y.
    y_top : float
        Upper bound of the layer along y (``y_top > y_bottom``).
    n_long : int
        Number of boundary elements on each of the two faces of length
        ``length`` (``n_long >= 1``).
    n_short : int
        Number of boundary elements on each of the two faces of height
        ``y_top - y_bottom`` (``n_short >= 1``).
    n_x : int
        Number of interior sample columns along x (``n_x >= 1``).
    n_y : int
        Number of interior sample rows along y (``n_y >= 1``).

    Returns
    -------
    result : tuple
        ``(elements, interior)``. ``elements`` is a float array of shape
        ``(2 * n_long + 2 * n_short, 3, 2)`` holding, for each boundary element
        in anticlockwise order starting from the corner ``(0, y_bottom)`` and
        running first along the face at ``y_bottom``, the ``(x, y)``
        coordinates of its first end node, its middle node and its second end
        node. ``interior`` is a float array of shape ``(n_x * n_y, 2)`` holding
        the interior sample points, which are the cell centres of a uniform
        ``n_x`` by ``n_y`` grid over the layer, ordered with the x index
        varying slowest.

    Raises
    ------
    ValueError
        If ``length`` is not a finite real number > 0, if ``y_bottom`` or
        ``y_top`` is not finite, if ``y_top`` is not greater than ``y_bottom``,
        or if any of ``n_long``, ``n_short``, ``n_x``, ``n_y`` is not an
        integer >= 1.

    """
    return (elements, interior)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rectangular_layer_mesh(length: float, y_bottom: float, y_top: float,
                                   n_long: int, n_short: int, n_x: int, n_y: int) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("length", length), ("y_bottom", y_bottom), ("y_top", y_top)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(length) <= 0.0:
        raise ValueError("length must be a finite number > 0")
    if float(y_top) <= float(y_bottom):
        raise ValueError("y_top must be greater than y_bottom")
    counts = {"n_long": n_long, "n_short": n_short, "n_x": n_x, "n_y": n_y}
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    span = float(length)
    low = float(y_bottom)
    high = float(y_top)

    def _face(start, end, count):
        # Subdivide one straight face into quadratic elements whose middle node
        # sits at the midpoint, keeping the traversal direction of the face.
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        out = []
        for index in range(int(count)):
            first = start + (end - start) * (index / float(count))
            second = start + (end - start) * ((index + 1) / float(count))
            out.append(np.array([first, 0.5 * (first + second), second], dtype=float))
        return out

    # Anticlockwise traversal of the rectangle keeps the swept area positive
    # when it is seen from any point inside the layer.
    corners = [(0.0, low), (span, low), (span, high), (0.0, high)]
    elements = []
    elements += _face(corners[0], corners[1], n_long)
    elements += _face(corners[1], corners[2], n_short)
    elements += _face(corners[2], corners[3], n_long)
    elements += _face(corners[3], corners[0], n_short)

    columns = (np.arange(int(n_x), dtype=float) + 0.5) * span / float(n_x)
    rows = low + (np.arange(int(n_y), dtype=float) + 0.5) * (high - low) / float(n_y)
    grid_x, grid_y = np.meshgrid(columns, rows, indexing="ij")
    interior = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    return (np.asarray(elements, dtype=float), np.asarray(interior, dtype=float))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- The outline of a unit square has a known perimeter, and its single
        # interior point is the centre. ---
        {
            "setup": """import numpy as np
def summarize(result):
    elements, interior = result
    perimeter = np.sum(np.linalg.norm(elements[:, 2, :] - elements[:, 0, :], axis=1))
    return float(perimeter + 100.0 * interior[0, 0] + 10000.0 * interior[0, 1])
""",
            "call": "summarize(rectangular_layer_mesh(1.0, 0.0, 1.0, 1, 1, 1, 1))",
            "gold_call": "summarize(_oracle_rectangular_layer_mesh(1.0, 0.0, 1.0, 1, 1, 1, 1))",
        },
        # --- Anticlockwise traversal makes the shoelace area positive. ---
        {
            "setup": """import numpy as np
def summarize(result):
    elements, _ = result
    return float(0.5 * np.sum(elements[:, 0, 0] * elements[:, 2, 1]
                              - elements[:, 2, 0] * elements[:, 0, 1]))
""",
            "call": "summarize(rectangular_layer_mesh(4.0, -0.25, 0.0, 3, 2, 2, 2))",
            "gold_call": "summarize(_oracle_rectangular_layer_mesh(4.0, -0.25, 0.0, 3, 2, 2, 2))",
        },
        # --- Valid: the benchmark coating layout, ten long and two short ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
def summarize(result):
    elements, interior = result
    return float(pin(elements) + 13.0 * pin(interior))
""",
            "call": "summarize(rectangular_layer_mesh(4.0, 0.0, 1e-4, 10, 2, 10, 2))",
            "gold_call": "summarize(_oracle_rectangular_layer_mesh(4.0, 0.0, 1e-4, 10, 2, 10, 2))",
        },
        # --- Valid: the benchmark substrate layout, which sits below y = 0 ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
def summarize(result):
    elements, interior = result
    return float(pin(elements) + 5.0 * pin(interior))
""",
            "call": "summarize(rectangular_layer_mesh(4.0, -1.0, 0.0, 10, 3, 8, 3))",
            "gold_call": "summarize(_oracle_rectangular_layer_mesh(4.0, -1.0, 0.0, 10, 3, 8, 3))",
        },
        # --- Boundary: an extremely thin layer, where the two long faces are
        # separated by far less than one element length ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
""",
            "call": "float(pin(rectangular_layer_mesh(4.0, 0.0, 1e-9, 4, 1, 3, 2)[1]))",
            "gold_call": "float(pin(_oracle_rectangular_layer_mesh(4.0, 0.0, 1e-9, 4, 1, 3, 2)[1]))",
        },
        # --- Edge: a single element per face on a wide short layer ---
        {
            "setup": """import numpy as np
def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    rank = np.arange(1.0, flat.size + 1.0)
    logs = np.log(np.abs(flat) + 1.0)
    scale = np.exp(np.mean(logs))
    order = np.sum(rank * np.sign(flat) * logs) / np.sum(rank)
    spread = np.sum(np.sqrt(rank) * logs ** 2) / np.sum(np.sqrt(rank))
    return float(flat.size * (1.0 + scale + spread + 0.5 * order))
def summarize(result):
    elements, interior = result
    return float(pin(elements[:, 1, :]) + pin(interior))
""",
            "call": "summarize(rectangular_layer_mesh(7.5, -2.0, -1.5, 1, 1, 4, 1))",
            "gold_call": "summarize(_oracle_rectangular_layer_mesh(7.5, -2.0, -1.5, 1, 1, 4, 1))",
        },
        # --- Invalid: an inverted layer whose top lies below its bottom ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rectangular_layer_mesh(4.0, 1.0, 0.0, 2, 2, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rectangular_layer_mesh(4.0, 1.0, 0.0, 2, 2, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a zero element count on the long faces ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        rectangular_layer_mesh(4.0, 0.0, 1.0, 0, 2, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_rectangular_layer_mesh(4.0, 0.0, 1.0, 0, 2, 2, 2)
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
