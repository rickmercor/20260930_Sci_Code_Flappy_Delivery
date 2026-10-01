"""
Build the particle-grid stencil of a staggered grid family and return the kernel weights, their spatial gradients and the node offsets for every particle.

The kernel decides how far a material point reaches into the background grid, and a kernel of unit support radius touches only the corners of the cell that contains the point, which is what keeps the transfer local. Evaluating the kernel and its derivative once per particle, for every grid of a staggered family, supplies every quantity the later transfer, force and tangent assemblies need.

Returns
-------
tuple of four np.ndarray: node lattice indices of shape (n_particles, n_grids, n_slots, 2), kernel weights of shape (n_particles, n_grids, n_slots), weight gradients of shape (n_particles, n_grids, n_slots, 2) and node offsets of shape (n_particles, n_grids, n_slots, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np 

def evaluate_kernel_weights(positions: "np.ndarray", spacing: float,
                            grid_offsets: tuple, kernel_name: str) -> tuple:
    """Evaluate the particle-grid kernel over a family of staggered grids.

    Grid ``k`` of the family carries nodes at the positions
    ``(index + grid_offsets[k]) * spacing`` in each coordinate direction, so an
    offset of 0 gives the unshifted lattice and an offset of 1/2 gives the
    lattice shifted by half a cell in both directions.

    Two one-dimensional kernel profiles are supported, both even in their
    argument ``d`` measured in units of the grid spacing.

    ``"compact"`` is the compact kernel of the compact-kernel material point
    method, ``K(d) = 1 - |d| + sin(2 pi |d|) / (2 pi)`` for ``|d| <= 1`` and
    zero outside, whose derivative is ``sign(d) (cos(2 pi d) - 1)``. Its
    support radius is one spacing, so two nodes per direction and four nodes
    per grid enter in two dimensions. Weights returned for this
    profile are clamped at zero: evaluated in floating point at ``|d| = 1``
    the expression above gives about ``-3.9e-17`` rather than exactly 0,
    because ``sin(2 pi)`` is not exactly representable, and a negative weight
    there would give a node a negative mass. Every returned weight is
    therefore non-negative.

    ``"quadratic"`` is the quadratic B-spline, ``K(d) = 3/4 - d^2`` for
    ``|d| < 1/2``, ``K(d) = (3/2 - |d|)^2 / 2`` for ``1/2 <= |d| <= 3/2`` and
    zero outside. Its support radius is three halves of a spacing, so three
    nodes per direction and nine nodes per grid enter.

    The two-dimensional weight is the product of the one-dimensional profiles
    over the two directions, and the returned weight gradient is its gradient
    with respect to the particle position. The stencil of grid ``k`` starts at the integer
    index ``floor(x / spacing - grid_offsets[k] - shift)`` in each direction,
    with ``shift`` equal to 0 for a support radius of one spacing and to 1/2
    for a support radius of three halves. Local stencil slots are numbered
    ``s = side * a + b``, where ``side`` is the number of nodes per direction,
    ``a`` counts nodes along the first coordinate and ``b`` along the second.

    Parameters
    ----------
    positions : "np.ndarray"
        Array of shape (n_particles, 2) holding the current particle positions
        in metres.
    spacing : float
        Background grid spacing in metres (spacing > 0).
    grid_offsets : tuple
        Offsets of the grids of the family, in units of the spacing; every
        entry must satisfy 0 <= offset < 1 and there must be at least one.
    kernel_name : str
        Either "compact" or "quadratic".

    Returns
    -------
    node_indices : "np.ndarray"
        Integer array of shape (n_particles, n_grids, n_slots, 2) holding the
        lattice indices of the stencil nodes.
    weights : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots) holding the kernel
        weights.
    weight_gradients : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the gradient
        of each weight with respect to the particle position, in m^-1.
    node_offsets : "np.ndarray"
        Array of shape (n_particles, n_grids, n_slots, 2) holding the node
        position minus the particle position, in metres.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return node_indices, weights, weight_gradients, node_offsets  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_kernel_weights(positions: "np.ndarray", spacing: float,
                                    grid_offsets: tuple, kernel_name: str) -> tuple:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    def _profile(name, d):
        """Return the one-dimensional kernel and its derivative in d."""
        magnitude = np.abs(d)
        if name == "compact":
            inside = magnitude <= 1.0
            value = np.where(inside,
                             1.0 - magnitude + np.sin(2.0 * np.pi * magnitude) / (2.0 * np.pi),
                             0.0)
            # The analytic profile is nonnegative and exactly zero at |d|=1.
            # Roundoff in sin(2*pi) can otherwise create a tiny negative nodal
            # mass when a particle lies exactly on a lattice node.
            value = np.maximum(value, 0.0)
            slope = np.where(inside, np.sign(d) * (np.cos(2.0 * np.pi * d) - 1.0), 0.0)
            return value, slope
        near = magnitude < 0.5
        far = (magnitude >= 0.5) & (magnitude <= 1.5)
        value = np.where(near, 0.75 - d * d, np.where(far, 0.5 * (1.5 - magnitude) ** 2, 0.0))
        slope = np.where(near, -2.0 * d, np.where(far, -np.sign(d) * (1.5 - magnitude), 0.0))
        return value, slope

    coordinates = np.asarray(positions, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("positions must have shape (n_particles, 2)")
    if coordinates.shape[0] == 0:
        raise ValueError("positions must hold at least one particle")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("positions must contain only finite entries")
    if not (isinstance(spacing, (int, float)) and np.isfinite(spacing) and float(spacing) > 0.0):
        raise ValueError("spacing must be a finite number > 0")
    offsets = np.asarray(grid_offsets, dtype=float).ravel()
    if offsets.size == 0:
        raise ValueError("grid_offsets must hold at least one offset")
    if not np.all(np.isfinite(offsets)) or np.any(offsets < 0.0) or np.any(offsets >= 1.0):
        raise ValueError("every entry of grid_offsets must satisfy 0 <= offset < 1")
    if kernel_name not in ("compact", "quadratic"):
        raise ValueError("kernel_name must be 'compact' or 'quadratic'")

    spacing = float(spacing)
    side = 2 if kernel_name == "compact" else 3
    shift = 0.0 if side == 2 else 0.5
    n_particles = coordinates.shape[0]
    n_grids = offsets.size
    n_slots = side * side

    node_indices = np.zeros((n_particles, n_grids, n_slots, 2), dtype=np.int64)
    weights = np.zeros((n_particles, n_grids, n_slots))
    weight_gradients = np.zeros((n_particles, n_grids, n_slots, 2))
    node_offsets = np.zeros((n_particles, n_grids, n_slots, 2))

    for grid, offset in enumerate(offsets):
        base = np.floor(coordinates / spacing - offset - shift).astype(np.int64)
        for first in range(side):
            for second in range(side):
                slot = side * first + second
                index = base + np.array([first, second], dtype=np.int64)
                separation = (index + offset) * spacing - coordinates
                reduced = separation / spacing
                value_x, slope_x = _profile(kernel_name, reduced[:, 0])
                value_y, slope_y = _profile(kernel_name, reduced[:, 1])
                node_indices[:, grid, slot, :] = index
                node_offsets[:, grid, slot, :] = separation
                weights[:, grid, slot] = value_x * value_y
                # The reduced coordinate falls as the particle advances, hence
                # the sign that turns the profile slope into a gradient in the
                # particle position.
                weight_gradients[:, grid, slot, 0] = -slope_x * value_y / spacing
                weight_gradients[:, grid, slot, 1] = -value_x * slope_y / spacing

    return node_indices, weights, weight_gradients, node_offsets

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned arrays
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: compact kernel on the staggered pair, benchmark spacing (normal scenario) ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.005, 0.005], [0.031, 0.017], [0.144, 0.098], [0.1913, 0.0377]])
offsets = (0.0, 0.5)

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_kernel_weights(positions, spacing, offsets, 'compact'))",
            "gold_call": "pin_all(_oracle_evaluate_kernel_weights(positions, spacing, offsets, 'compact'))",
        },
        # --- Valid: quadratic B-spline on a single grid, wider stencil ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.005, 0.005], [0.031, 0.017], [0.144, 0.098], [0.1913, 0.0377]])

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_kernel_weights(positions, spacing, (0.0,), 'quadratic'))",
            "gold_call": "pin_all(_oracle_evaluate_kernel_weights(positions, spacing, (0.0,), 'quadratic'))",
        },
        # --- Valid: reproduction properties of the staggered pair against a single grid ---
        # The averaged weights sum to one on either family, but only the pair
        # reproduces the particle position, so the averaged first moment of the
        # node offsets vanishes for the pair and not for the single grid.
        {
            "setup": """import numpy as np
spacing = 0.02
fraction = np.linspace(0.03, 0.97, 11)
positions = np.stack([(3.0 + fraction) * spacing, (5.0 + fraction[::-1]) * spacing], axis=1)

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def reproduction(fn):
    pair = fn(positions, spacing, (0.0, 0.5), 'compact')
    single = fn(positions, spacing, (0.0,), 'compact')
    unity = np.abs(pair[1].sum(axis=(1, 2)) / 2.0 - 1.0).max()
    paired = np.einsum('pks,pksa->pa', pair[1], pair[3]) / 2.0
    lone = np.einsum('pks,pksa->pa', single[1], single[3])
    flags = float(int(unity < 1.0e-12) + 2 * int(np.abs(paired).max() < 1.0e-12)
                  + 4 * int(np.abs(lone).max() > 1.0e-4))
    return flags + pin(paired) + 3.0 * pin(lone)
""",
            "call": "reproduction(evaluate_kernel_weights)",
            "gold_call": "reproduction(_oracle_evaluate_kernel_weights)",
        },
        # --- Boundary: a particle sitting exactly on a node of the unshifted grid ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.04, 0.06], [0.04, 0.0601]])

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_kernel_weights(positions, spacing, (0.0, 0.5), 'compact'))",
            "gold_call": "pin_all(_oracle_evaluate_kernel_weights(positions, spacing, (0.0, 0.5), 'compact'))",
        },
        # --- Edge: negative coordinates and a three-grid family ---
        {
            "setup": """import numpy as np
spacing = 0.05
positions = np.array([[-0.017, 0.083], [-0.144, -0.201], [0.0, -0.05]])
offsets = (0.0, 1.0 / 3.0, 2.0 / 3.0)

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def pin_all(parts):
    total = 1.0
    for order, part in enumerate(parts):
        total *= (1.0 + pin(part)) ** (1.0 / (order + 1.0))
    return float(total)
""",
            "call": "pin_all(evaluate_kernel_weights(positions, spacing, offsets, 'compact'))",
            "gold_call": "pin_all(_oracle_evaluate_kernel_weights(positions, spacing, offsets, 'compact'))",
        },
        # --- Boundary: exact lattice-node positions keep every weight nonnegative ---
        {
            "setup": """import numpy as np
spacing = 0.02
positions = np.array([[0.0, 0.0], [0.02, 0.04]])

def boundary_nonnegative(fn):
    weights = fn(positions, spacing, (0.0, 0.5), 'compact')[1]
    return float(np.all(weights >= 0.0))
""",
            "call": "boundary_nonnegative(evaluate_kernel_weights)",
            "gold_call": "boundary_nonnegative(_oracle_evaluate_kernel_weights)",
        },
        # --- Invalid: unknown kernel name ---
        {
            "setup": """import numpy as np
positions = np.array([[0.01, 0.02]])
def run_model():
    try:
        evaluate_kernel_weights(positions, 0.02, (0.0, 0.5), 'cubic')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_kernel_weights(positions, 0.02, (0.0, 0.5), 'cubic')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a grid offset of one, which duplicates the unshifted lattice ---
        {
            "setup": """import numpy as np
positions = np.array([[0.01, 0.02]])
def run_model():
    try:
        evaluate_kernel_weights(positions, 0.02, (0.0, 1.0), 'compact')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_kernel_weights(positions, 0.02, (0.0, 1.0), 'compact')
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
