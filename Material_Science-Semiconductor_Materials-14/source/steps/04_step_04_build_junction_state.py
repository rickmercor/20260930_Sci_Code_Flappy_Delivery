"""
Evaluate the scaled net doping at every unknown and the Ohmic contact values of the potential and the two carrier densities.

The doping profile is the only place where the device geometry enters the physics, and for the abrupt junction used here it is piecewise constant: donor dominated below the metallurgical junction at half the device height, acceptor dominated above it, and the arithmetic mean of the two exactly on it. The third branch is not decoration. On a structured mesh whose row count is even, the vertices of one whole mesh line sit exactly on the junction, and any implementation that tests only for less than or greater than will silently assign those nodes to one side and change the assembled charge density. Because the mesh distortion leaves the vertices on the two vertical boundaries where they are, this is exactly what happens on the benchmark configuration, so the three-branch profile is load bearing rather than pedantic.




An Ohmic contact is a boundary at which the semiconductor is assumed to remain in thermal equilibrium and electrically neutral no matter what current flows. Those two statements fix all three unknowns there. Neutrality requires the net charge to vanish, so the difference of the hole and electron densities equals the net doping with the sign reversed, while equilibrium requires the mass action law, so their product equals the square of the effective intrinsic density. Solving the resulting quadratic and keeping the positive root gives densities equal to half of the net doping plus or minus half the square root of the net doping squared plus four times the intrinsic density squared, with the plus root for electrons and the minus root for holes. Written this way the expressions are stable for both signs of the doping and degrade gracefully to the intrinsic density when the doping vanishes, which is what happens on the junction line itself.




The contact potential then follows from the equilibrium relation between the electron density and the quasi-Fermi level. Measuring potentials in units of the thermal voltage, the built-in part of the contact potential is the natural logarithm of the ratio of the equilibrium electron density to the intrinsic density, and the externally applied bias is added on top of it. The current that eventually flows is exponentially sensitive to how much of the built-in voltage the applied bias removes, so an error of one thermal voltage in the contact potential changes the answer by a factor of e.




All quantities returned here are already nondimensionalised: densities by the reference doping, potentials by the thermal voltage, so that the intrinsic density enters as a small number of order 1e-5 and the doping of the benchmark device is exactly plus or minus one.

Returns
-------
np.ndarray of shape (n_nodes, 4), float: the scaled net doping and the Ohmic electron density, hole density and potential at every node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_junction_state(nodes: np.ndarray, anode_voltage: float) -> np.ndarray:
    """Evaluate the scaled doping and the Ohmic contact values at every node.

    Parameters
    ----------
    nodes : np.ndarray
        Node table of shape (n_nodes, 5) as returned by
        build_ddfv_node_table: coordinate, control volume measure, equation
        class code and contact tag.
    anode_voltage : float
        Voltage applied at the anode contact, in volts. The cathode contact
        is grounded.

    Returns
    -------
    state : np.ndarray
        Array of shape (n_nodes, 4). Column 0 is the scaled net doping,
        column 1 the Ohmic electron density, column 2 the Ohmic hole density
        and column 3 the Ohmic potential in units of the thermal voltage.
        Columns 1 to 3 are meaningful only at Dirichlet nodes but are
        evaluated everywhere, because they also serve as the charge-neutral
        initial guess of the Newton iteration.

    Raises
    ------
    ValueError
        If nodes is not a finite array of shape (n_nodes, 5), or
        anode_voltage is not finite.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_build_junction_state(nodes: np.ndarray, anode_voltage: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 5 or nodes.shape[0] < 1:
        raise ValueError("nodes must be a 2D array with shape (n_nodes, 5)")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes must contain only finite values")
    if not (isinstance(anode_voltage, (int, float))
            and np.isfinite(anode_voltage)):
        raise ValueError("anode_voltage must be a finite number")

    # Silicon at 300 K, scaled by the reference doping and the thermal voltage.
    thermal_voltage = 0.025852                       # V
    scaled_intrinsic = 1.087386e10 / 1.0e15          # n_ie / N_0

    height = nodes[:, 1]
    contact = nodes[:, 4]

    # Abrupt junction: donor dominated below the junction line, acceptor
    # dominated above it, and the mean of the two exactly on it.
    doping = np.where(height < 0.5, 1.0, np.where(height > 0.5, -1.0, 0.0))

    # Charge neutrality together with the mass action law at an Ohmic contact.
    root = np.sqrt(doping * doping + 4.0 * scaled_intrinsic * scaled_intrinsic)
    electrons = 0.5 * (doping + root)
    holes = 0.5 * (-doping + root)

    applied = np.where(contact == 2.0, float(anode_voltage) / thermal_voltage, 0.0)
    potential = applied + np.log(electrons / scaled_intrinsic)

    return np.column_stack([doping, electrons, holes, potential])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark configuration at its working bias ---
        {
            "setup": """import numpy as np
ys = np.linspace(0.0, 1.0, 21)
nodes = np.zeros((21, 5))
nodes[:, 1] = ys
nodes[:, 2] = 0.01
nodes[0, 4] = 1.0
nodes[-1, 4] = 2.0
anode_voltage = 0.4
""",
            "call": "float(np.sum((_a := np.ravel(build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: nodes sitting exactly on the metallurgical junction ---
        {
            "setup": """import numpy as np
nodes = np.zeros((5, 5))
nodes[:, 1] = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
nodes[:, 2] = 0.02
nodes[0, 4] = 1.0
nodes[4, 4] = 2.0
anode_voltage = 0.4
""",
            "call": "float(np.sum((_a := np.ravel(build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
        },
        # --- Boundary: zero applied bias, so both contacts sit at equilibrium ---
        {
            "setup": """import numpy as np
nodes = np.zeros((7, 5))
nodes[:, 1] = np.linspace(0.0, 1.0, 7)
nodes[:, 2] = 0.05
nodes[0, 4] = 1.0
nodes[6, 4] = 2.0
anode_voltage = 0.0
""",
            "call": "float(np.sum((_a := np.ravel(build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
        },
        # --- Edge: reverse bias beyond the built-in voltage ---
        {
            "setup": """import numpy as np
nodes = np.zeros((9, 5))
nodes[:, 1] = np.linspace(0.0, 1.0, 9)
nodes[:, 2] = 0.03
nodes[0, 4] = 1.0
nodes[8, 4] = 2.0
anode_voltage = -5.0
""",
            "call": "float(np.sum((_a := np.ravel(build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
            "gold_call": "float(np.sum((_a := np.ravel(_oracle_build_junction_state(nodes, anode_voltage))) * np.arange(1, _a.size + 1)))",
        },
        # --- Invalid: node table with the wrong column count ---
        {
            "setup": """import numpy as np
nodes = np.zeros((4, 3))
def run_model():
    try:
        build_junction_state(nodes, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_junction_state(nodes, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite applied bias ---
        {
            "setup": """import numpy as np
nodes = np.zeros((4, 5))
nodes[:, 1] = np.linspace(0.0, 1.0, 4)
def run_model():
    try:
        build_junction_state(nodes, float('nan'))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_junction_state(nodes, float('nan'))
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
