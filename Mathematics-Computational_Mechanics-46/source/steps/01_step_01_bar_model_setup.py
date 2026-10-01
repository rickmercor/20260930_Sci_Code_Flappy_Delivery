"""
Builds the duplicated-node degree-of-freedom map, the connectivity, and the lumped nodal masses for a uniform two-node bar discretization with cohesive interfaces at every other interior node, plus the two derived elastic-wave scalars used to nondimensionalize the whole simulation.

Geometry, duplicated-node degree-of-freedom map, and lumped masses.




A one-dimensional bar of uniform cross-section is discretized into equal-length two-node elements. A cohesive interface can only carry a displacement jump if the two faces it separates are independent unknowns, so every node that sits on an interface is duplicated into a minus-face and a plus-face degree of freedom while every other node keeps a single degree of freedom; the resulting map from node index to one or two degree-of-freedom indices is the object every later step indexes into. Consistent (lumped) mass assigns each node the mass of the bar material tributary to it: a regular interior node owns one full element on each side, an end node or an interface face owns only the one element attached to that face, so end nodes and interface faces each carry half the mass of a regular interior node, and the discrete masses must sum exactly to the bar's total mass with no leakage. The two derived scalars record how information crosses the whole bar: the elastic wave speed of the undamaged material and the time a wave takes to traverse the bar twice, which nondimensionalizes every later time reported for the simulation.

Returns
-------
setup : dict -- n_dof, h_e, wave_speed, bar_period, interface_nodes, dof_minus, dof_plus, element_dofs, mass, mass_regular, mass_face (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bar_model_setup(n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4) -> dict:
    """Build the duplicated-node DOF map, connectivity, and lumped masses.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2;
        interfaces sit at every other interior node, which requires an even
        element count so the last node is never itself an interface node).
    length : float
        Total bar length L, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus E, in pascals (must be > 0).
    density : float
        Bulk mass density rho, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area A, in m^2 (must be > 0).

    Returns
    -------
    setup : dict
        n_dof : int, total degree-of-freedom count (regular nodes + 2 per
            interface node).
        h_e : float, element length L / n_e.
        wave_speed : float, elastic bar wave speed sqrt(E / rho).
        bar_period : float, 2 * length / wave_speed.
        interface_nodes : (n_if,) int array, node indices carrying an
            interface (the odd interior nodes).
        dof_minus, dof_plus : (n_if,) int arrays, the minus-face and
            plus-face degree-of-freedom index of each interface node.
        element_dofs : (n_e, 2) int array, the (left, right) degree-of-freedom
            index of every bulk element, in element order.
        mass : (n_dof,) float array, lumped mass at every degree of freedom.
        mass_regular : float, lumped mass of a regular (non-end, non-interface)
            node, rho * A * h_e.
        mass_face : float, lumped mass of an end node or an interface face,
            rho * A * h_e / 2.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if length, youngs_modulus,
        density, or area is not a positive real number.
    """
    return setup

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bar_model_setup(n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4) -> dict:
    """Reference implementation of bar_model_setup."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (length > 0.0 and youngs_modulus > 0.0 and density > 0.0 and area > 0.0):
        raise ValueError("length, youngs_modulus, density, and area must be positive")
    n_e = int(n_e)
    n_nodes = n_e + 1
    h_e = length / n_e
    wave_speed = float(np.sqrt(youngs_modulus / density))
    bar_period = 2.0 * length / wave_speed

    interface_nodes = np.arange(1, n_e, 2)
    n_if = interface_nodes.size

    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof

    dof_minus = np.array([node_dofs[k][0] for k in interface_nodes], dtype=int)
    dof_plus = np.array([node_dofs[k][1] for k in interface_nodes], dtype=int)

    element_dofs = np.empty((n_e, 2), dtype=int)
    for e in range(n_e):
        left_node, right_node = e, e + 1
        left_dof = node_dofs[left_node][1] if len(node_dofs[left_node]) == 2 else node_dofs[left_node][0]
        right_dof = node_dofs[right_node][0]
        element_dofs[e] = (left_dof, right_dof)

    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    mass = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular

    return {
        "n_dof": n_dof,
        "h_e": h_e,
        "wave_speed": wave_speed,
        "bar_period": bar_period,
        "interface_nodes": interface_nodes,
        "dof_minus": dof_minus,
        "dof_plus": dof_plus,
        "element_dofs": element_dofs,
        "mass": mass,
        "mass_regular": mass_regular,
        "mass_face": mass_face,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: normal scenario (benchmark n_e=500), boundary case (minimal
    n_e=2), edge case (invalid n_e raises ValueError).
    """
    return [
        {
            # benchmark config: wave speed, bar period, and total mass
            # conservation (sum of the lumped masses must equal rho*A*L
            # exactly, up to floating-point round-off).
            "setup": "import numpy as np",
            "call": (
                "float(bar_model_setup(500)['wave_speed']) "
                "+ float(bar_model_setup(500)['bar_period']) "
                "+ float(np.sum(bar_model_setup(500)['mass']))"
            ),
            "gold_call": (
                "float(_oracle_bar_model_setup(500)['wave_speed']) "
                "+ float(_oracle_bar_model_setup(500)['bar_period']) "
                "+ float(np.sum(_oracle_bar_model_setup(500)['mass']))"
            ),
        },
        {
            # regular vs. face lumped masses, and the interface/element
            # counts, at the benchmark config
            "setup": "import numpy as np",
            "call": (
                "float(bar_model_setup(500)['mass_regular']) "
                "+ float(bar_model_setup(500)['mass_face']) "
                "+ float(bar_model_setup(500)['interface_nodes'].size) "
                "+ float(bar_model_setup(500)['n_dof'])"
            ),
            "gold_call": (
                "float(_oracle_bar_model_setup(500)['mass_regular']) "
                "+ float(_oracle_bar_model_setup(500)['mass_face']) "
                "+ float(_oracle_bar_model_setup(500)['interface_nodes'].size) "
                "+ float(_oracle_bar_model_setup(500)['n_dof'])"
            ),
        },
        {
            # boundary: minimal two-element bar (one interface at node 1)
            "setup": "import numpy as np",
            "call": (
                "float(np.sum(bar_model_setup(2)['mass'])) "
                "+ float(bar_model_setup(2)['n_dof']) "
                "+ float(np.max(bar_model_setup(2)['element_dofs']))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_bar_model_setup(2)['mass'])) "
                "+ float(_oracle_bar_model_setup(2)['n_dof']) "
                "+ float(np.max(_oracle_bar_model_setup(2)['element_dofs']))"
            ),
        },
        {
            # edge: odd n_e is invalid (interfaces would land on the last node)
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: bar_model_setup(3))",
            "gold_call": "_guard(lambda: _oracle_bar_model_setup(3))",
        },
        {
            # edge: n_e below the minimum of 2 is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: bar_model_setup(1))",
            "gold_call": "_guard(lambda: _oracle_bar_model_setup(1))",
        },
    ]
