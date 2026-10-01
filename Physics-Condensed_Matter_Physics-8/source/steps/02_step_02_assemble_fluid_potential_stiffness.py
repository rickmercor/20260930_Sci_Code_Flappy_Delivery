"""
The liquid above the plate is incompressible, inviscid and irrotational, so its displacement is the gradient of a scalar potential and incompressibility makes that potential harmonic throughout the layer. The weak form of the harmonic condition is the Dirichlet form, the volume integral of the squared gradient of the potential, and nothing else: an incompressible liquid stores no energy in compression, so the operator this step assembles carries no material constant at all, not even the density. The density enters the coupled problem only where the liquid exchanges force with the plate or with the free surface, which is the next step.

The operator therefore has the dimensions of a length rather than of a stiffness, and the potential has the dimensions of an area, its gradient being a displacement. Reading those dimensions is the quickest check that the density has not been smuggled in here.

The layer is meshed with eight-node hexahedral elements whose in-plane division matches the plate exactly, so that the lowermost face of the liquid column shares its four nodes with the plate element beneath it and the uppermost face shares its four nodes with the free-surface element above it. That conformity is what allows the two couplings to be assembled without any interpolation between meshes. Within the layer the division through the depth is into equal slices.

Nodes are numbered with the first in-plane index fastest, then the second, then the layer index, so that the node at in-plane indices $(i, j)$ in layer $k$ is numbered $i + j (n + 1) + k (n + 1)^2$ for a division into $n$ elements per side. Layer zero is the face wetted by the plate and layer $n_z$ is the free surface. An element takes its four lower nodes anticlockwise from the corner of lowest indices, then the four upper nodes in the same order.

A property of the assembled operator matters later. Because no value of the potential is imposed anywhere on the cell, a potential that is constant throughout the liquid lies in its null space: a constant potential has zero gradient and therefore describes no motion. The operator is consequently singular before any periodic condition is applied, and whether it stays singular afterwards depends on the wave vector.

Returns
-------
dict holding the native int n_dof; the float layer_thickness in metres; and the float64 array stiffness of shape (n_dof, n_dof), the Dirichlet form of the potential over the whole layer with no periodic condition imposed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_fluid_potential_stiffness(
    n_elem: int,
    cell_side: float,
    fluid_depth: float,
    n_layer: int,
) -> dict:
    """Assemble the Dirichlet form of the liquid displacement potential over the layer.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell in plane, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    fluid_depth : float
        Depth of the liquid layer in metres.
    n_layer : int
        Number of equal slices through the depth, one or more.

    Returns
    -------
    dict
        Under the keys n_dof, layer_thickness and stiffness.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when n_layer is not an integer of one or more, or when cell_side or fluid_depth is not finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _counted_layer_division(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _positive_layer_length(value, label):
    """Return a length as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _hex_gradient_form(side_x, side_y, side_z):
    """Dirichlet form of one trilinear hexahedron with edges aligned to the axes."""
    corners = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    form = np.zeros((8, 8))
    jacobian = 0.125 * side_x * side_y * side_z
    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:
                gradient = np.vstack([
                    0.125 * a * (1.0 + b * eta) * (1.0 + c * zeta) * 2.0 / side_x,
                    0.125 * b * (1.0 + a * xi) * (1.0 + c * zeta) * 2.0 / side_y,
                    0.125 * c * (1.0 + a * xi) * (1.0 + b * eta) * 2.0 / side_z])
                form += gradient.T @ gradient * jacobian
    return form


def _oracle_assemble_fluid_potential_stiffness(
    n_elem: int,
    cell_side: float,
    fluid_depth: float,
    n_layer: int,
) -> dict:
    """Reference implementation."""
    n_elem = _counted_layer_division(n_elem, "n_elem", 2)
    n_layer = _counted_layer_division(n_layer, "n_layer", 1)
    cell_side = _positive_layer_length(cell_side, "cell_side")
    fluid_depth = _positive_layer_length(fluid_depth, "fluid_depth")

    side = cell_side / n_elem
    slice_height = fluid_depth / n_layer
    if side <= 0.0 or slice_height <= 0.0:
        raise ValueError("the division leaves an element of no extent")
    n_side = n_elem + 1
    plane = n_side * n_side
    n_dof = plane * (n_layer + 1)
    stiffness = np.zeros((n_dof, n_dof))
    element_form = _hex_gradient_form(side, side, slice_height)
    for layer in range(n_layer):
        for ex in range(n_elem):
            for ey in range(n_elem):
                lower = (ex + ey * n_side + layer * plane,
                         ex + 1 + ey * n_side + layer * plane,
                         ex + 1 + (ey + 1) * n_side + layer * plane,
                         ex + (ey + 1) * n_side + layer * plane)
                upper = tuple(node + plane for node in lower)
                place = np.array(lower + upper)
                stiffness[np.ix_(place, place)] += element_form
    return {
        "n_dof": n_dof,
        "layer_thickness": slice_height,
        "stiffness": stiffness,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# the constant potential describes no motion, so it lies exactly in the null space
def digest(out):
    ones = np.ones(out["n_dof"])
    return (out["n_dof"], round(out["layer_thickness"], 12),
            int(float(np.abs(out["stiffness"] @ ones).max()) < 1.0e-12),
            int(float(np.abs(out["stiffness"] - out["stiffness"].T).max()) < 1.0e-15),
            round(float(np.trace(out["stiffness"])), 9),
            round(float(out["stiffness"][0, 0]), 12))
""",
            "call": "digest(assemble_fluid_potential_stiffness(10, 0.2, 0.01, 4))",
            "gold_call": "digest(_oracle_assemble_fluid_potential_stiffness(10, 0.2, 0.01, 4))",
        },
        {
            "setup": """import numpy as np
# boundary case, a single slice through the depth, and a cubic aspect ratio for comparison
def digest(out):
    ones = np.ones(out["n_dof"])
    eig = np.linalg.eigvalsh(out["stiffness"])
    return (out["n_dof"], round(out["layer_thickness"], 12),
            int(float(np.abs(out["stiffness"] @ ones).max()) < 1.0e-12),
            int(abs(eig[0]) < 1.0e-10), int(eig[1] > 1.0e-8),
            round(float(eig[-1]), 10))
""",
            "call": "digest(assemble_fluid_potential_stiffness(4, 0.2, 0.05, 1))",
            "gold_call": "digest(_oracle_assemble_fluid_potential_stiffness(4, 0.2, 0.05, 1))",
        },
        {
            "setup": """import numpy as np
# the form has the dimensions of a length: scaling every edge by a factor scales it by that factor
def digest(pair):
    coarse, fine = pair
    return (round(float(np.trace(coarse["stiffness"])), 9),
            round(float(np.trace(fine["stiffness"])), 9),
            round(float(np.trace(fine["stiffness"]) / np.trace(coarse["stiffness"])), 9))
""",
            "call": "digest((assemble_fluid_potential_stiffness(3, 0.3, 0.15, 2), assemble_fluid_potential_stiffness(3, 0.6, 0.30, 2)))",
            "gold_call": "digest((_oracle_assemble_fluid_potential_stiffness(3, 0.3, 0.15, 2), _oracle_assemble_fluid_potential_stiffness(3, 0.6, 0.30, 2)))",
        },
        {
            "setup": """import numpy as np
def sentinel(fn):
    codes = []
    for args in [(1, 0.2, 0.01, 4), (10, 0.2, 0.01, 0), (10, 0.0, 0.01, 4),
                 (10, 0.2, -0.01, 4), (10, 0.2, np.nan, 4), (10.0, 0.2, 0.01, 4)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "sentinel(assemble_fluid_potential_stiffness)",
            "gold_call": "sentinel(_oracle_assemble_fluid_potential_stiffness)",
        },
    ]
