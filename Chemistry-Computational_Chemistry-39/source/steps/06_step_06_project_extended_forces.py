"""
Project every extended-system force observation onto the noised coarse-grained beads using the fitted extended force map.

The fitted extended force map acts on the combined atomistic and noise-site axis. Replicates, frames, and Cartesian components must remain separate during the projection so that every generated force observation remains available to the final conservative regression.

Returns
-------
Return the projected target forces with shape (replicates, frames, beads, components).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def project_extended_forces(
    joint_forces: np.ndarray,
    extended_force_map: np.ndarray,
) -> np.ndarray:
    """Project extended-system forces to the noised CG beads.

    Parameters
    ----------
    joint_forces
        Array with shape
        (replicates, frames, extended_sites, components).
    extended_force_map
        Matrix with shape (beads, extended_sites).

    Returns
    -------
    np.ndarray
        Target CG forces with shape
        (replicates, frames, beads, components).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_project_extended_forces(joint_forces, extended_force_map):
    import numpy as np

    joint = np.asarray(joint_forces, dtype=float)
    force_map = np.asarray(extended_force_map, dtype=float)
    if joint.ndim != 4 or min(joint.shape) == 0:
        raise ValueError("joint_forces must be a nonempty 4D array")
    if (
        force_map.ndim != 2
        or force_map.shape[0] == 0
        or force_map.shape[1] != joint.shape[2]
    ):
        raise ValueError("extended_force_map has incompatible shape")
    if np.any(~np.isfinite(joint)) or np.any(~np.isfinite(force_map)):
        raise ValueError("joint forces and map must be finite")
    return np.einsum("qj,sfjc->sfqc", force_map, joint)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nG=np.arange(96,dtype=float).reshape(2,4,4,3)/11; "
            "S=np.array([[1.,0.,.2,-.1],[0.,1.,-.3,.4]])",
            "call": "project_extended_forces(G,S)",
            "gold_call": "_oracle_project_extended_forces(G,S)",
        },
        {
            "setup": "import numpy as np\nG=np.array([[[[1.,-1.],[2.,3.]]]]); S=np.eye(2)",
            "call": "project_extended_forces(G,S)",
            "gold_call": "_oracle_project_extended_forces(G,S)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(15); G=rng.normal(size=(3,5,7,1)); "
            "S=rng.normal(size=(2,7))",
            "call": "project_extended_forces(G,S)",
            "gold_call": "_oracle_project_extended_forces(G,S)",
        },
        {
            "setup": "import numpy as np\nG=np.arange(12,dtype=float).reshape(1,2,3,2)-3.; "
            "S=np.array([[0.,0.,0.],[1.,-1.,.5],[-.25,.75,1.]])",
            "call": "project_extended_forces(G,S)",
            "gold_call": "_oracle_project_extended_forces(G,S)",
        },
    ]
