"""
Three boundary integrals close the coupled cell, and none of them introduces a degree of freedom of its own. The first lives on the wetted face of the plate and pairs the transverse deflection with the liquid potential of the nodes directly beneath the free surface, on the lowermost face of the liquid column. It carries the no-penetration condition in one direction and the pressure the liquid exerts on the plate in the other, and it is the same operator in both because the two conditions are adjoint. Only the deflection appears in it: the rotations of the transverse normal do no work against a pressure acting along the normal, so every row belonging to a rotation is empty.

The second lives on the free surface and pairs the elevation with the liquid potential on the uppermost face of the column. It states that the normal displacement of the liquid at the surface is the elevation, which is what keeps the liquid incompressible while the surface rises and falls.

The third is the only elastic operator above the plate. Displacing the free surface by an elevation raises liquid above the undisturbed level and leaves a deficit below it, and gravity supplies the restoring pressure, proportional to the liquid density, the gravitational acceleration and the elevation. Its weak form is that product times the surface integral of the elevation against itself, so this operator alone carries the density and the gravitational acceleration.

All three are built from the same object, the surface integral of the bilinear shape functions against one another over a square face, because the plate, the liquid face and the free surface share the same in-plane division and the same interpolation. Evaluating that integral with the two-by-two rule gives the consistent form, in which a node is coupled to its neighbours. Replacing it with its row sums, the lumped form, is a different and cruder operator and changes every coupled frequency.

The signs of the two coupling operators can be taken either way provided each is used consistently, because reversing the sense of the interface normal reverses both the force the liquid applies to the plate and the displacement the plate imposes on the liquid. The eigenvalues of the coupled cell do not see the choice.

Node numbering follows the two preceding steps: the free surface is numbered like the plate, with the first in-plane index fastest, and the liquid nodes carry the layer index slowest, so layer zero is the wetted face and layer $n_z$ is the free surface.

Returns
-------
dict holding the native int n_surface_dof; the float64 array fsi_coupling of shape (3 (n_elem + 1)^2, (n_elem + 1)^2 (n_layer + 1)) pairing the plate degrees of freedom with the liquid potential on the wetted face; the float64 array surface_coupling of shape ((n_elem + 1)^2, (n_elem + 1)^2 (n_layer + 1)) pairing the elevation with the liquid potential on the free surface; and the float64 array surface_stiffness of shape ((n_elem + 1)^2, (n_elem + 1)^2), the gravitational restoring operator of the free surface.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_interface_operators(
    n_elem: int,
    cell_side: float,
    n_layer: int,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Assemble the two interface couplings and the gravitational restoring operator.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell in plane, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    n_layer : int
        Number of slices through the depth of the liquid, one or more.
    fluid_density : float
        Liquid density in kilogram per cubic metre.
    gravity : float
        Gravitational acceleration in metre per second squared.

    Returns
    -------
    dict
        Under the keys n_surface_dof, fsi_coupling, surface_coupling and surface_stiffness.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when n_layer is not an integer of one or more, or when cell_side, fluid_density or gravity is not finite and above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _q4_shape_face(xi, eta):
    """Bilinear shape functions of the reference square and their parametric gradients."""
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    value = 0.25 * (1.0 + corners[:, 0] * xi) * (1.0 + corners[:, 1] * eta)
    d_xi = 0.25 * corners[:, 0] * (1.0 + corners[:, 1] * eta)
    d_eta = 0.25 * corners[:, 1] * (1.0 + corners[:, 0] * xi)
    return value, d_xi, d_eta


def _counted_face_division(value, label, floor):
    """Return a count as a native int once it is known to be an integer at or above a floor."""
    if not isinstance(value, (int, np.integer)) or int(value) < floor:
        raise ValueError("%s wants an integer of %d or more" % (label, floor))
    return int(value)


def _positive_face_length(value, label):
    """Return a quantity as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _face_gram(side):
    """Consistent surface integral of the bilinear shape functions against one another."""
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    gram = np.zeros((4, 4))
    for xi in gauss:
        for eta in gauss:
            value, _, _ = _q4_shape_face(xi, eta)
            gram += np.outer(value, value) * 0.25 * side * side
    return gram


def _oracle_assemble_interface_operators(
    n_elem: int,
    cell_side: float,
    n_layer: int,
    fluid_density: float,
    gravity: float,
) -> dict:
    """Reference implementation."""
    n_elem = _counted_face_division(n_elem, "n_elem", 2)
    n_layer = _counted_face_division(n_layer, "n_layer", 1)
    cell_side = _positive_face_length(cell_side, "cell_side")
    fluid_density = _positive_face_length(fluid_density, "fluid_density")
    gravity = _positive_face_length(gravity, "gravity")

    side = cell_side / n_elem
    if side <= 0.0 or not np.isfinite(fluid_density * gravity):
        raise ValueError("the division or the restoring scale leaves no usable operator")
    n_side = n_elem + 1
    plane = n_side * n_side
    n_fluid = plane * (n_layer + 1)
    gram = _face_gram(side)
    fsi_coupling = np.zeros((3 * plane, n_fluid))
    surface_coupling = np.zeros((plane, n_fluid))
    surface_stiffness = np.zeros((plane, plane))
    top = n_layer * plane
    for ex in range(n_elem):
        for ey in range(n_elem):
            face = np.array([ex + ey * n_side, ex + 1 + ey * n_side,
                             ex + 1 + (ey + 1) * n_side, ex + (ey + 1) * n_side])
            deflection = 3 * face
            fsi_coupling[np.ix_(deflection, face)] += -gram
            surface_coupling[np.ix_(face, face + top)] += gram
            surface_stiffness[np.ix_(face, face)] += fluid_density * gravity * gram
    return {
        "n_surface_dof": plane,
        "fsi_coupling": fsi_coupling,
        "surface_coupling": surface_coupling,
        "surface_stiffness": surface_stiffness,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# every consistent face integral sums to the area it covers, so the three operators
# reproduce the cell area, the cell area and the weight per unit area of a unit elevation
def digest(out):
    n = out["n_surface_dof"]
    area_fsi = float(np.abs(out["fsi_coupling"]).sum())
    area_surface = float(np.abs(out["surface_coupling"]).sum())
    restoring = float(out["surface_stiffness"].sum())
    rotations = float(np.abs(out["fsi_coupling"][1::3]).max() + np.abs(out["fsi_coupling"][2::3]).max())
    # a lumped form has the same row sums and the same support, so the totals above cannot
    # tell the two apart. These do: the consistent face integral couples a node to its
    # neighbours, in the ratio 4 : 2 : 1 across an element, and pairs a non-uniform
    # deflection with a non-uniform potential through those off-diagonal entries.
    deflection = out["fsi_coupling"][0::3]
    row = np.unique(np.round(np.abs(deflection[0])[np.abs(deflection[0]) > 0.0], 15))[::-1]
    probe_w = np.cos(np.arange(n) * 0.7)
    probe_phi = np.sin(np.arange(deflection.shape[1]) * 0.3)
    # magnitude only: the step description licenses either sense of the interface normal,
    # so a digest that carried the sign would reject one of the two permitted choices
    pairing = abs(float(probe_w @ deflection @ probe_phi))
    return (n, round(area_fsi, 12), round(area_surface, 12), round(restoring, 9),
            round(rotations, 15),
            int(float(np.abs(out["surface_stiffness"] - out["surface_stiffness"].T).max()) < 1.0e-12),
            round(float(row[0]), 12), round(float(row[1]), 12), round(float(row[2]), 12),
            round(pairing, 12))
""",
            "call": "digest(assemble_interface_operators(10, 0.2, 4, 1000.0, 9.8))",
            "gold_call": "digest(_oracle_assemble_interface_operators(10, 0.2, 4, 1000.0, 9.8))",
        },
        {
            "setup": """import numpy as np
# the two couplings must reach different faces of the liquid: the plate the lowermost,
# the free surface the uppermost, and neither anything in between
def digest(out):
    n = out["n_surface_dof"]
    fsi = out["fsi_coupling"]
    top = out["surface_coupling"]
    n_layer = fsi.shape[1] // n - 1
    lowest = float(np.abs(fsi[:, :n]).sum())
    elsewhere = float(np.abs(fsi[:, n:]).sum())
    highest = float(np.abs(top[:, n_layer * n:]).sum())
    below = float(np.abs(top[:, :n_layer * n]).sum())
    return (n, n_layer, round(lowest, 12), round(elsewhere, 15),
            round(highest, 12), round(below, 15))
""",
            "call": "digest(assemble_interface_operators(6, 0.3, 3, 1000.0, 9.8))",
            "gold_call": "digest(_oracle_assemble_interface_operators(6, 0.3, 3, 1000.0, 9.8))",
        },
        {
            "setup": """import numpy as np
# boundary case, the coarsest admissible cell with one slice, where the consistent form
# is easiest to read off: the diagonal entry of one interior node is four ninths of the element area
def digest(out):
    g = out["surface_stiffness"] / (1000.0 * 9.8)
    return (out["n_surface_dof"], round(float(g.sum()), 12),
            round(float(g.max()), 12), round(float(g.min()), 12),
            round(float(np.sort(np.diag(g))[-1]), 12),
            round(float(np.abs(out["surface_coupling"]).max()), 12))
""",
            "call": "digest(assemble_interface_operators(2, 0.2, 1, 1000.0, 9.8))",
            "gold_call": "digest(_oracle_assemble_interface_operators(2, 0.2, 1, 1000.0, 9.8))",
        },
        {
            "setup": """import numpy as np
def sentinel(fn):
    codes = []
    for args in [(1, 0.2, 4, 1000.0, 9.8), (10, 0.2, 0, 1000.0, 9.8),
                 (10, 0.0, 4, 1000.0, 9.8), (10, 0.2, 4, -1000.0, 9.8),
                 (10, 0.2, 4, 1000.0, 0.0), (10, 0.2, 4, np.inf, 9.8),
                 (10, 0.2, 4.5, 1000.0, 9.8)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "sentinel(assemble_interface_operators)",
            "gold_call": "sentinel(_oracle_assemble_interface_operators)",
        },
    ]
