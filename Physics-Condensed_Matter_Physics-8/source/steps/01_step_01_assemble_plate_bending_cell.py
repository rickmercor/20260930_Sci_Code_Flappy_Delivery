"""
The lattice is a flat plate of uniform thickness whose material alternates between a compliant matrix and a stiff, dense square inclusion repeated on a square lattice. Because the plate is flat and its section is symmetric about the mid-plane, stretching of the mid-plane and bending of the section do not couple, and only the bending family is driven by a liquid resting on the face. The cell therefore carries three generalised displacements at each node, the transverse deflection $w$ and the two rotations of the transverse normal, and the moderately thick description is used, in which those rotations are independent of the slope of $w$ so that transverse shear is a strain of its own rather than a constraint.

Two element integrals make up the cell. The bending energy pairs the gradients of the two rotations through the section-integrated plane-stress moduli, which carry a factor of the cube of the thickness. The transverse shear energy pairs the difference between the slope of the deflection and the rotation, through the shear modulus times the thickness times the shear correction factor. Integrating both with the same rule makes a four-node element far too stiff in bending, because the bilinear deflection cannot match the bilinear rotations closely enough to keep the shear strain small, so the shear term is evaluated at one point while the bending term keeps the two-by-two rule. That choice is a property of the element, not of the physics, and it moves every frequency the cell predicts.

The inertia of the section is likewise section-integrated: the deflection carries the density times the thickness, and each rotation carries the density times the cube of the thickness over twelve, the rotary inertia. The two phases differ in all three constants, and a node sitting on the phase boundary receives contributions from elements of both phases, which is what makes the assembled operators those of a genuine composite rather than of an averaged plate.

Nodes are numbered along the first in-plane axis fastest, so the node at in-plane indices $(i, j)$ is numbered $i + j (n + 1)$ for a division into $n$ elements per side, and the three degrees of freedom of a node occupy consecutive positions in the order deflection, then the rotation of the transverse normal in the plane spanned by the first in-plane axis and the thickness direction, then the rotation in the plane spanned by the second in-plane axis and the thickness direction. The first rotation is the one that pairs with the slope of the deflection along the first axis in the transverse shear strain, and the second with the slope along the second axis. Reading them instead as rotations about those axes exchanges the pair and transposes the cross-curvature terms. An element takes its four nodes anticlockwise starting from the corner of lowest indices. An element belongs to the inclusion when its centre lies inside the inclusion square, and the configuration is required to place the inclusion boundary on element boundaries so that no element is split.

Returns
-------
dict holding the native ints n_dof and n_inclusion_element; and the float64 arrays stiffness and mass, each of shape (n_dof, n_dof), assembled over the whole cell without any periodic condition imposed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_plate_bending_cell(
    n_elem: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
) -> dict:
    """Assemble the unreduced bending stiffness and inertia of the two-phase plate cell.

    Parameters
    ----------
    n_elem : int
        Elements per side of the square cell, two or more.
    cell_side : float
        Lattice constant of the square cell in metres.
    inclusion_side : float
        Side of the centred square inclusion in metres.
    plate_thickness : float
        Uniform plate thickness in metres.
    shear_factor : float
        Transverse shear correction factor.
    matrix_constants : np.ndarray
        Young modulus, Poisson ratio and density of the matrix phase, shape (3,).
    inclusion_constants : np.ndarray
        Young modulus, Poisson ratio and density of the inclusion phase, shape (3,).

    Returns
    -------
    dict
        Under the keys n_dof, n_inclusion_element, stiffness and mass.

    Raises
    ------
    ValueError
        When n_elem is not an integer of two or more, when any length is not finite and above zero, when inclusion_side does not sit below cell_side, when the inclusion boundary does not fall on element boundaries, when shear_factor is outside the half-open interval from zero to one, when either constants array does not hold three finite entries, or when a modulus or density fails to sit above zero or a Poisson ratio falls outside the open interval from minus one to one half.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _q4_shape_plate(xi, eta):
    """Bilinear shape functions of the reference square and their parametric gradients."""
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    value = 0.25 * (1.0 + corners[:, 0] * xi) * (1.0 + corners[:, 1] * eta)
    d_xi = 0.25 * corners[:, 0] * (1.0 + corners[:, 1] * eta)
    d_eta = 0.25 * corners[:, 1] * (1.0 + corners[:, 0] * xi)
    return value, d_xi, d_eta


def _positive_plate_constant(value, label):
    """Return a length or a modulus as a float once it is known to be finite and above zero."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("%s wants a finite value above zero" % label)
    return number


def _phase_constants(values, label):
    """Return the three constants of one phase once they are known to be admissible."""
    trio = np.asarray(values, dtype=float)
    if trio.shape != (3,):
        raise ValueError("%s wants three constants" % label)
    if not np.isfinite(trio).all():
        raise ValueError("%s holds an entry that is not finite" % label)
    modulus, poisson, density = float(trio[0]), float(trio[1]), float(trio[2])
    if modulus <= 0.0 or density <= 0.0:
        raise ValueError("%s wants a modulus and a density above zero" % label)
    if not -1.0 < poisson < 0.5:
        raise ValueError("%s wants a Poisson ratio between minus one and one half" % label)
    return modulus, poisson, density


def _bending_element(side, modulus, poisson, density, thickness, shear_factor):
    """Stiffness and inertia of one square bending element with one-point shear integration."""
    shear_modulus = modulus / (2.0 * (1.0 + poisson))
    direct = modulus / (1.0 - poisson ** 2)
    cross = modulus * poisson / (1.0 - poisson ** 2)
    curvature_moduli = (thickness ** 3 / 12.0) * np.array(
        [[direct, cross, 0.0], [cross, direct, 0.0], [0.0, 0.0, shear_modulus]])
    section_inertia = density * np.array(
        [thickness, thickness ** 3 / 12.0, thickness ** 3 / 12.0])
    gauss = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    jacobian = 0.25 * side * side
    stiffness = np.zeros((12, 12))
    inertia = np.zeros((12, 12))
    for xi in gauss:
        for eta in gauss:
            value, d_xi, d_eta = _q4_shape_plate(xi, eta)
            d_x, d_y = d_xi * 2.0 / side, d_eta * 2.0 / side
            curvature = np.zeros((3, 12))
            interpolation = np.zeros((3, 12))
            for node in range(4):
                base = 3 * node
                curvature[0, base + 1] = d_x[node]
                curvature[1, base + 2] = d_y[node]
                curvature[2, base + 1] = d_y[node]
                curvature[2, base + 2] = d_x[node]
                interpolation[:, base:base + 3] = value[node] * np.eye(3)
            stiffness += curvature.T @ curvature_moduli @ curvature * jacobian
            inertia += interpolation.T @ np.diag(section_inertia) @ interpolation * jacobian
    value, d_xi, d_eta = _q4_shape_plate(0.0, 0.0)
    d_x, d_y = d_xi * 2.0 / side, d_eta * 2.0 / side
    shear = np.zeros((2, 12))
    for node in range(4):
        base = 3 * node
        shear[0, base] = d_x[node]
        shear[0, base + 1] = -value[node]
        shear[1, base] = d_y[node]
        shear[1, base + 2] = -value[node]
    shear_moduli = shear_factor * shear_modulus * thickness * np.eye(2)
    stiffness += shear.T @ shear_moduli @ shear * (side * side)
    return stiffness, inertia


def _oracle_assemble_plate_bending_cell(
    n_elem: int,
    cell_side: float,
    inclusion_side: float,
    plate_thickness: float,
    shear_factor: float,
    matrix_constants: np.ndarray,
    inclusion_constants: np.ndarray,
) -> dict:
    """Reference implementation."""
    if not isinstance(n_elem, (int, np.integer)) or int(n_elem) < 2:
        raise ValueError("n_elem wants an integer of two or more")
    n_elem = int(n_elem)
    cell_side = _positive_plate_constant(cell_side, "cell_side")
    inclusion_side = _positive_plate_constant(inclusion_side, "inclusion_side")
    plate_thickness = _positive_plate_constant(plate_thickness, "plate_thickness")
    if inclusion_side >= cell_side:
        raise ValueError("the inclusion must sit inside the cell")
    factor = float(shear_factor)
    if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise ValueError("shear_factor wants a value above zero and at most one")
    matrix = _phase_constants(matrix_constants, "matrix_constants")
    inclusion = _phase_constants(inclusion_constants, "inclusion_constants")

    side = cell_side / n_elem
    margin = 0.5 * (cell_side - inclusion_side) / side
    if abs(margin - round(margin)) > 1.0e-9 * max(1.0, abs(margin)):
        raise ValueError("the inclusion boundary must fall on element boundaries")
    low = int(round(margin))
    high = n_elem - low
    if high <= low:
        raise ValueError("the inclusion must span at least one element")

    n_side = n_elem + 1
    n_dof = 3 * n_side * n_side
    stiffness = np.zeros((n_dof, n_dof))
    mass = np.zeros((n_dof, n_dof))
    matrix_pair = _bending_element(side, *matrix, plate_thickness, factor)
    inclusion_pair = _bending_element(side, *inclusion, plate_thickness, factor)
    n_inclusion_element = 0
    for ex in range(n_elem):
        for ey in range(n_elem):
            inside = low <= ex < high and low <= ey < high
            n_inclusion_element += int(inside)
            element_stiffness, element_mass = inclusion_pair if inside else matrix_pair
            nodes = (ex + ey * n_side, ex + 1 + ey * n_side,
                     ex + 1 + (ey + 1) * n_side, ex + (ey + 1) * n_side)
            place = np.array([3 * node + c for node in nodes for c in range(3)])
            stiffness[np.ix_(place, place)] += element_stiffness
            mass[np.ix_(place, place)] += element_mass
    return {
        "n_dof": n_dof,
        "n_inclusion_element": n_inclusion_element,
        "stiffness": stiffness,
        "mass": mass,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [{'setup': 'import numpy as np\n'
           '# a uniform cell: the total translational inertia and the rigid-translation null vector are '
           'exact\n'
           'UNIFORM = np.array([6.0e6, 0.47, 1290.0])\n'
           'def digest(out):\n'
           '    n = out["n_dof"]\n'
           '    rigid = np.zeros(n)\n'
           '    rigid[0::3] = 1.0\n'
           '    total = float(rigid @ out["mass"] @ rigid)\n'
           '    scale = float(np.abs(out["stiffness"]).max())\n'
           '    residual = float(np.abs(out["stiffness"] @ rigid).max()) / scale\n'
           '    symmetry = float(np.abs(out["stiffness"] - out["stiffness"].T).max()) / scale\n'
           '    return (n, out["n_inclusion_element"], round(total, 9), int(residual < 1.0e-12),\n'
           '            int(symmetry < 1.0e-12), round(float(np.trace(out["mass"])), 9))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(assemble_plate_bending_cell)(4, 0.2, 0.1, 0.02, 5.0 / 6.0, UNIFORM, '
          'UNIFORM))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_assemble_plate_bending_cell)(4, 0.2, 0.1, 0.02, 5.0 / '
               '6.0, UNIFORM, UNIFORM))'},
 {'setup': 'import numpy as np\n'
           '# the graded two-phase cell: 36 of the 100 elements carry the inclusion\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           'def digest(out):\n'
           '    rigid = np.zeros(out["n_dof"])\n'
           '    rigid[0::3] = 1.0\n'
           '    return (out["n_dof"], out["n_inclusion_element"],\n'
           '            round(float(rigid @ out["mass"] @ rigid), 9),\n'
           '            # five decimals, not six: the maximum is exactly 5.3515625e9, which sits on the\n'
           '            # six-decimal rounding boundary, so an equivalent arithmetic ordering differing in\n'
           '            # the last bit flips the digit and fails an implementation that is correct\n'
           '            round(float(np.trace(out["stiffness"]) / 1.0e9), 5),\n'
           '            round(float(np.abs(out["stiffness"]).max() / 1.0e9), 5),\n'
           '            round(float(out["mass"][0, 0]), 12))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(assemble_plate_bending_cell)(10, 0.2, 0.12, 0.02, 5.0 / 6.0, MATRIX, '
          'TUNGSTEN))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_assemble_plate_bending_cell)(10, 0.2, 0.12, 0.02, 5.0 / '
               '6.0, MATRIX, TUNGSTEN))'},
 {'setup': 'import numpy as np\n'
           '# one-point shear integration leaves the element rank deficient in shear, which a boundary case '
           'exposes:\n'
           '# the coarsest cell that can host a centred inclusion still annihilates rigid translation\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           'def digest(out):\n'
           '    rigid = np.zeros(out["n_dof"])\n'
           '    rigid[0::3] = 1.0\n'
           '    eig = np.linalg.eigvalsh(out["stiffness"])\n'
           '    scale = float(np.abs(out["stiffness"]).max())\n'
           '    return (out["n_dof"], out["n_inclusion_element"],\n'
           '            int(float(np.abs(out["stiffness"] @ rigid).max()) / scale < 1.0e-12),\n'
           '            int(eig.min() > -1.0e-9 * eig.max()), round(float(eig.max() / 1.0e9), 6),\n'
           '            round(float(rigid @ out["mass"] @ rigid), 9))\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'digest(_scicode_fresh_inputs(assemble_plate_bending_cell)(3, 0.3, 0.1, 0.02, 5.0 / 6.0, MATRIX, '
          'TUNGSTEN))',
  'gold_call': 'digest(_scicode_fresh_inputs(_oracle_assemble_plate_bending_cell)(3, 0.3, 0.1, 0.02, 5.0 / '
               '6.0, MATRIX, TUNGSTEN))'},
 {'setup': 'import numpy as np\n'
           'MATRIX = np.array([6.0e6, 0.47, 1290.0])\n'
           'TUNGSTEN = np.array([411.0e9, 0.28, 19300.0])\n'
           'BAD_NU = np.array([6.0e6, 0.7, 1290.0])\n'
           'def sentinel(fn):\n'
           '    codes = []\n'
           '    for args in [\n'
           '        (1, 0.2, 0.12, 0.02, 5.0/6.0, MATRIX, TUNGSTEN),\n'
           '        (10, 0.2, 0.13, 0.02, 5.0/6.0, MATRIX, TUNGSTEN),\n'
           '        (10, 0.2, 0.25, 0.02, 5.0/6.0, MATRIX, TUNGSTEN),\n'
           '        (10, 0.2, 0.12, 0.0, 5.0/6.0, MATRIX, TUNGSTEN),\n'
           '        (10, 0.2, 0.12, 0.02, 1.5, MATRIX, TUNGSTEN),\n'
           '        (10, 0.2, 0.12, 0.02, 5.0/6.0, BAD_NU, TUNGSTEN),\n'
           '        (10, np.inf, 0.12, 0.02, 5.0/6.0, MATRIX, TUNGSTEN),\n'
           '    ]:\n'
           '        try:\n'
           '            fn(*args)\n'
           '            codes.append(0)\n'
           '        except ValueError:\n'
           '            codes.append(1)\n'
           '        except Exception:\n'
           '            codes.append(2)\n'
           '    return tuple(codes)\n'
           '\n'
           '\n'
           '# Each candidate/oracle invocation receives equivalent independent inputs.\n'
           '# Copy the whole arguments object to retain aliases within one invocation.\n'
           'def _scicode_fresh_inputs(function):\n'
           '    def invoke(*args, **kwargs):\n'
           '        from copy import deepcopy\n'
           '        call_args, call_kwargs = deepcopy((args, kwargs))\n'
           '        return function(*call_args, **call_kwargs)\n'
           '    return invoke\n',
  'call': 'sentinel(_scicode_fresh_inputs(assemble_plate_bending_cell))',
  'gold_call': 'sentinel(_scicode_fresh_inputs(_oracle_assemble_plate_bending_cell))'}]
