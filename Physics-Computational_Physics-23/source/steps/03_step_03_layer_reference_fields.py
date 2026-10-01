"""
Evaluate the benchmark temperature field of one layer, its two spatial derivatives and the heat source density that makes it an exact solution.

A manufactured benchmark prescribes the temperature field analytically and then defines the heat source density as whatever the transient conduction balance requires, so that the field is an exact solution of the layer problem and can be used to measure the error of a numerical scheme. The transient balance used here reads as the Laplacian of the temperature equal to the time derivative divided by the thermal conductivity plus the source density.

Returns
-------
np.ndarray, float, shape (n, 4): the temperature, its x derivative, its y derivative and the heat source density at each sample point.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layer_reference_fields(points: "np.ndarray", time: float, slope: float,
                           conductivity: float) -> "np.ndarray":
    """Sample the benchmark field of one layer and the source density it implies.

    The benchmark temperature of a layer is
    ``u(x, y, t) = (slope * y + 3 - sin(x)) * exp(t)``, where ``slope`` is the
    coefficient that the interface continuity of temperature and of normal heat
    flux assigns to that layer.

    Parameters
    ----------
    points : np.ndarray
        Float array of shape ``(n, 2)`` holding the ``(x, y)`` coordinates at
        which the fields are wanted.
    time : float
        Time at which the fields are wanted (finite).
    slope : float
        Coefficient multiplying ``y`` in the benchmark field of this layer
        (finite).
    conductivity : float
        Thermal conductivity of this layer (``conductivity > 0``).

    Returns
    -------
    fields : np.ndarray
        Float array of shape ``(n, 4)`` whose columns are, in order, the
        temperature, its derivative with respect to x, its derivative with
        respect to y, and the heat source density of the transient conduction
        balance at the sample points.

    Raises
    ------
    ValueError
        If ``points`` is not a finite real array of shape ``(n, 2)`` with
        ``n >= 1``, if ``time`` or ``slope`` is not a finite real number, or if
        ``conductivity`` is not a finite real number > 0.

    """
    return fields  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_layer_reference_fields(points: "np.ndarray", time: float, slope: float,
                                   conductivity: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sample = np.asarray(points, dtype=float)
    if sample.ndim != 2 or sample.shape[0] < 1 or sample.shape[1] != 2:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    if not np.all(np.isfinite(sample)):
        raise ValueError("points must be finite")
    for name, value in (("time", time), ("slope", slope), ("conductivity", conductivity)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(conductivity) <= 0.0:
        raise ValueError("conductivity must be a finite number > 0")

    x_coord = sample[:, 0]
    y_coord = sample[:, 1]
    growth = np.exp(float(time))
    tilt = float(slope)

    # The benchmark field and the two derivatives that build the boundary flux.
    temperature = (tilt * y_coord + 3.0 - np.sin(x_coord)) * growth
    d_dx = -np.cos(x_coord) * growth
    d_dy = np.full_like(x_coord, tilt) * growth

    # The field grows like exp(t), so its time derivative equals the field
    # itself, and its Laplacian comes only from the sine in x. The source
    # density is what the transient balance leaves over.
    laplacian = np.sin(x_coord) * growth
    source = laplacian - temperature / float(conductivity)

    return np.column_stack([temperature, d_dx, d_dy, source])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned: at the origin at time zero the field, its two derivatives
        # and the source are all closed forms, obtained without the oracle.
        {
            "setup": """import numpy as np
k = 2.0
slope = 7.5
""",
            "call": ("float(np.dot(layer_reference_fields(np.zeros((1, 2)), 0.0, slope, k)[0],"
                     " np.array([1.0, 10.0, 100.0, 1000.0])))"),
            "gold_call": ("float(np.dot(_oracle_layer_reference_fields(np.zeros((1, 2)), 0.0, slope, k)[0],"
                          " np.array([1.0, 10.0, 100.0, 1000.0])))"),
        },
        # --- Pinned: the transient balance must close, so the Laplacian of the
        # field minus the time derivative over the conductivity has to return
        # the source column at any sample point.
        {
            "setup": """import numpy as np
pts = np.array([[0.7, -0.4], [2.9, 0.15], [4.0, -0.9]])
t = 1.3
slope = 1.0
k = 15.0
""",
            "call": "float(np.sum(layer_reference_fields(pts.copy(), t, slope, k)[:, 3]))",
            "gold_call": "float(np.sum(_oracle_layer_reference_fields(pts.copy(), t, slope, k)[:, 3]))",
        },
        # --- Valid: a spread of coating points at a Gauss node time ---
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
pts = np.column_stack([np.linspace(0.2, 3.8, 10), np.full(10, 2.5e-5)])
""",
            "call": "float(pin(layer_reference_fields(pts.copy(), 0.4691007703, 7.5, 2.0)))",
            "gold_call": "float(pin(_oracle_layer_reference_fields(pts.copy(), 0.4691007703, 7.5, 2.0)))",
        },
        # --- Valid: substrate points, where the tilt is unity and the
        # conductivity much larger ---
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
gx, gy = np.meshgrid(np.linspace(0.25, 3.75, 8), np.array([-0.8, -0.5, -0.2]), indexing='ij')
pts = np.column_stack([gx.ravel(), gy.ravel()])
""",
            "call": "float(pin(layer_reference_fields(pts.copy(), 0.9530899230, 1.0, 15.0)))",
            "gold_call": "float(pin(_oracle_layer_reference_fields(pts.copy(), 0.9530899230, 1.0, 15.0)))",
        },
        # --- Boundary: a single point at a negative time, where the exponential
        # damps every column ---
        {
            "setup": """import numpy as np
""",
            "call": ("float(1.0 + 1.0e6 * float(np.dot(layer_reference_fields(np.array([[1.5, -0.25]]), -2.0, 1.0, 0.5)[0],"
                     " np.array([1.0, 2.0, 3.0, 4.0]))))"),
            "gold_call": ("float(1.0 + 1.0e6 * float(np.dot(_oracle_layer_reference_fields(np.array([[1.5, -0.25]]), -2.0, 1.0, 0.5)[0],"
                          " np.array([1.0, 2.0, 3.0, 4.0]))))"),
        },
        # --- Edge: a very small conductivity makes the source dominate ---
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
pts = np.array([[0.05, 0.0], [3.14159, 1.0e-9], [6.2, -3.0]])
""",
            "call": "float(pin(layer_reference_fields(pts.copy(), 0.25, 12.0, 0.01)))",
            "gold_call": "float(pin(_oracle_layer_reference_fields(pts.copy(), 0.25, 12.0, 0.01)))",
        },
        # --- Invalid: a non-positive conductivity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        layer_reference_fields(np.zeros((2, 2)), 0.0, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_layer_reference_fields(np.zeros((2, 2)), 0.0, 1.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: points with three coordinates per row ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        layer_reference_fields(np.zeros((2, 3)), 0.0, 1.0, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_layer_reference_fields(np.zeros((2, 3)), 0.0, 1.0, 2.0)
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
