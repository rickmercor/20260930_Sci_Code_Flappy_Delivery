"""
Fit the bead-space stage of the force-and-noise aggregation while keeping the noise-variable coefficient fixed to identity.

The compatible base force map first projects the atomistic force block into bead space. A second bead-space matrix W is fitted so that the retained atomistic contribution and the noise force combine optimally. The complete extended map has block form [W T, I], and the ridge penalty applies only to W.

Returns
-------
Return the extended force map with shape (beads, atoms + beads).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_staged_force_map(
    joint_forces: np.ndarray,
    base_force_map: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Fit the second stage of a force-and-noise aggregation map.

    The atomistic block is first projected by the compatible base map. A
    bead-space matrix W is then fitted while the noise block keeps an identity
    coefficient. The returned extended map is [W @ base_force_map, I].

    Parameters
    ----------
    joint_forces
        Extended forces with shape
        (replicates, frames, atoms + beads, components).
    base_force_map
        Compatible map with shape (beads, atoms).
    ridge
        Nonnegative regularization on W.

    Returns
    -------
    np.ndarray
        Extended force map with shape (beads, atoms + beads).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_staged_force_map(joint_forces, base_force_map, ridge):
    import numpy as np

    joint = np.asarray(joint_forces, dtype=float)
    base_map = np.asarray(base_force_map, dtype=float)
    if joint.ndim != 4 or min(joint.shape) == 0:
        raise ValueError("joint_forces must be a nonempty 4D array")
    if base_map.ndim != 2 or min(base_map.shape) == 0:
        raise ValueError("base_force_map must be a nonempty matrix")
    n_beads, n_atoms = base_map.shape
    if joint.shape[2] != n_atoms + n_beads:
        raise ValueError("extended force axis has incompatible length")
    if np.any(~np.isfinite(joint)) or np.any(~np.isfinite(base_map)):
        raise ValueError("joint forces and base map must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    atom_block = joint[:, :, :n_atoms, :]
    noise_block = joint[:, :, n_atoms:, :]
    mapped_atom = np.einsum("qa,sfac->sfqc", base_map, atom_block)
    design = np.transpose(mapped_atom, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    response = np.transpose(noise_block, (0, 1, 3, 2)).reshape(
        -1, n_beads
    )
    count = len(design)
    gram = design.T @ design / count + ridge * np.eye(n_beads)
    cross = response.T @ design / count
    try:
        stage = -np.linalg.solve(gram.T, cross.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("staged force-map system is singular") from exc
    return np.concatenate([stage @ base_map, np.eye(n_beads)], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(5); G=rng.normal(size=(4,6,6,3)); "
            "T=np.array([[1.1,.2,-.1,.0],[-.2,.3,.5,.8]])",
            "call": "fit_staged_force_map(G,T,.03)",
            "gold_call": "_oracle_fit_staged_force_map(G,T,.03)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(8); G=rng.normal(size=(2,3,4,1)); "
            "T=np.eye(2)",
            "call": "fit_staged_force_map(G,T,.1)",
            "gold_call": "_oracle_fit_staged_force_map(G,T,.1)",
        },
        {
            "setup": "import numpy as np\nG=np.array([[[[1.],[2.],[-1.]],[[2.],[-1.],[.5]]],"
            "[[[.5],[1.5],[-.2]],[[1.2],[.4],[.8]]]]); "
            "T=np.array([[.4,.6]])",
            "call": "fit_staged_force_map(G,T,.02)",
            "gold_call": "_oracle_fit_staged_force_map(G,T,.02)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(29); "
            "G=rng.normal(size=(3,4,5,2)); "
            "T=np.array([[.7,.3,0.],[0.,.2,.8]])",
            "call": "fit_staged_force_map(G,T,0.0)",
            "gold_call": "_oracle_fit_staged_force_map(G,T,0.0)",
        },
    ]
