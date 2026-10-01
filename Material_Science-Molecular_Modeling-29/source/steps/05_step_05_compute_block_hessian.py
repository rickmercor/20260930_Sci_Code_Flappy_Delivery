"""
Build the mass-scaled second-derivative matrix of the all-atom potential truncated to one diagonal block per monomer.

The curvature of the all-atom surface enters the coarse-graining twice: it defines the subspace along which the fast coordinates are driven to equilibrium, and it supplies the normal-mode frequencies that price the harmonic free energy of those coordinates. Using the full second-derivative matrix for either purpose costs a diagonalization that scales as the cube of the total number of atomic coordinates, which is what made the original formulation of this coarse-graining more expensive than the all-atom simulation it was meant to replace.

The truncation that removes the bottleneck keeps only the diagonal blocks, one per group of atoms treated as a rigid unit, and discards every block coupling different groups. The diagonalization cost then becomes the sum of the cubes of the individual block sizes, which for a cluster of small monomers is a negligible fraction of the original. The truncation is not a change of physics for the purpose it serves: the modes being priced are the stiff intramolecular ones, which are localized on a single monomer, while the discarded off-diagonal blocks carry the soft intermolecular couplings that the coarse-grained degrees of freedom represent explicitly.

Each block must be mass-scaled on both sides by the inverse square roots of the atomic masses of its own group. Working in mass-scaled coordinates is what makes the eigenvalues of a block the squared angular frequencies of its normal modes rather than bare force constants, and it is also what makes the relaxation step a Newton step in the metric that the vibrational problem actually uses. Scaling on one side only, or omitting the scaling and treating the raw force-constant matrix as if it were the dynamical matrix, changes both the mode ordering and the relaxation path.

The blocks are built by central differences of the analytic gradient rather than of the energy, which retains far more significant figures, and only the coordinates of the block's own atoms are displaced. The gradient itself is not re-derived here: it is supplied as a callable, so this step owns only the differentiation, and a curvature built from some other gradient than the one the relaxation follows would put the two halves of the algorithm on different surfaces. Because differencing does not produce an exactly symmetric result, each block is explicitly symmetrized. The blocks are genuinely those of the full cluster Hessian, not of an isolated monomer: displacing an atom changes the intermolecular terms as well, so the intermolecular environment is imprinted on the curvature, and that is precisely how the surrounding hydrogen-bond network comes to shift the monomer frequencies.

Returns
-------
np.ndarray of shape (n_monomers, 9, 9), float: the symmetric mass-scaled second-derivative block of each monomer, in kcal/(mol angstrom**2 u).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_block_hessian(coords: np.ndarray, atom_masses: np.ndarray,
                          gradient_fn, step: float = 1.0e-5) -> np.ndarray:
    """Build the mass-scaled block-diagonal Hessian of the cluster potential.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the atomic
        positions, ordered O, H, H within each monomer.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    gradient_fn : callable
        Analytic gradient of the potential being differentiated: it maps an
        array of positions of the same shape as ``coords`` to the derivative of
        the energy with respect to every atomic coordinate, an array of that
        same shape. For the cluster of this task it is the function built in
        sub-problem 03, called with its default q-TIP4P/F parameters.
    step : float
        Central-difference displacement in angstrom used to differentiate the
        analytic gradient (step > 0).

    Returns
    -------
    blocks : np.ndarray
        Array of shape (n_monomers, 9, 9) holding one symmetric mass-scaled
        second-derivative block per monomer, in kcal/(mol angstrom**2 u). The
        row and column order within a block is the atom-major flattening
        O(x, y, z), H(x, y, z), H(x, y, z).

    Raises
    ------
    ValueError
        If ``coords`` has the wrong shape, atom count, or non-finite entries;
        if ``atom_masses`` has a mismatched shape or any non-finite or
        non-positive entry; if ``gradient_fn`` is not callable; or if ``step``
        is non-finite or non-positive.
    """
    return blocks  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_block_hessian(coords: np.ndarray, atom_masses: np.ndarray,
                                  gradient_fn, step: float = 1.0e-5) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not callable(gradient_fn):
        raise ValueError("gradient_fn must be callable")
    if not (isinstance(step, (int, float, np.floating, np.integer))
            and not isinstance(step, bool) and np.isfinite(step) and float(step) > 0.0):
        raise ValueError("step must be a finite number > 0")

    step = float(step)

    n_monomers = coords.shape[0] // 3
    blocks = np.zeros((n_monomers, 9, 9), dtype=float)

    for mono in range(n_monomers):
        rows = np.arange(3 * mono, 3 * mono + 3)
        block = np.zeros((9, 9), dtype=float)
        # Displace only this monomer's coordinates and read back only this
        # monomer's gradient components: that is exactly the diagonal block.
        for local_atom in range(3):
            for axis in range(3):
                column = 3 * local_atom + axis
                forward = coords.copy()
                forward[rows[local_atom], axis] += step
                backward = coords.copy()
                backward[rows[local_atom], axis] -= step
                grad_forward = gradient_fn(forward)
                grad_backward = gradient_fn(backward)
                difference = (grad_forward[rows] - grad_backward[rows]) / (2.0 * step)
                block[:, column] = difference.reshape(9)

        block = 0.5 * (block + block.T)
        inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
        blocks[mono] = block * np.outer(inverse_root, inverse_root)

    return blocks

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    # Shared setup: the gradient to be differentiated is supplied explicitly, so
    # each test stands alone and none of them depends on a built-in potential.
    preamble = """import numpy as np

def _grad(positions, bond_force=550.0, bond_length=0.96):
    # Gradient of a compact test surface standing in for the all-atom one: two
    # harmonic O-H bonds and a harmonic H-H strut inside each monomer, plus an
    # inverse-cube repulsion between every pair of atoms on different monomers,
    # so that a block still carries the imprint of its neighbours.
    grad = np.zeros_like(positions)
    for mono in range(positions.shape[0] // 3):
        o, h1, h2 = 3 * mono, 3 * mono + 1, 3 * mono + 2
        for i, j, force, length in ((o, h1, bond_force, bond_length),
                                    (o, h2, bond_force, bond_length),
                                    (h1, h2, 70.0, 1.54)):
            offset = positions[j] - positions[i]
            distance = float(np.linalg.norm(offset))
            scalar = 2.0 * force * (distance - length) / distance
            grad[j] += scalar * offset
            grad[i] -= scalar * offset
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            if i // 3 == j // 3:
                continue
            offset = positions[j] - positions[i]
            distance = float(np.linalg.norm(offset))
            scalar = -36.0 / distance ** 5
            grad[j] += scalar * offset
            grad[i] -= scalar * offset
    return grad
"""
    return [
        # --- Valid: the benchmark cyclic tetramer (normal scenario) ---
        {
            "setup": preamble + """
coords = np.array([
    [1.965757, 0.000000, 0.000000], [1.276254, 0.689503, 0.097837],
    [2.425271, 0.000868, 0.837181], [0.000000, 1.965757, 0.000000],
    [-0.680176, 1.285581, -0.077118], [0.004437, 2.439238, -0.852304],
    [-1.965757, 0.000000, 0.000000], [-1.270755, -0.695002, 0.118515],
    [-2.428412, 0.000015, 0.841160], [0.000000, -1.965757, 0.000000],
    [0.684659, -1.281098, -0.058165], [0.015394, -2.427969, -0.869683]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 4)
""",
            "call": "compute_block_hessian(coords, masses, _grad)",
            "gold_call": "_oracle_compute_block_hessian(coords, masses, _grad)",
        },
        # --- Valid: a single isolated monomer, where the block is the whole
        #     Hessian and six of its eigenvalues must be the free translations
        #     and rotations ---
        {
            "setup": preamble + """
half = np.deg2rad(107.4) / 2.0
coords = np.array([[0.0, 0.0, 0.0],
                   [0.9419 * np.sin(half), 0.0, 0.9419 * np.cos(half)],
                   [-0.9419 * np.sin(half), 0.0, 0.9419 * np.cos(half)]])
masses = np.array([15.9994, 1.00794, 1.00794])
""",
            "call": "compute_block_hessian(coords, masses, _grad)",
            "gold_call": "_oracle_compute_block_hessian(coords, masses, _grad)",
        },
        # --- Boundary: a coarser difference step, which must still return a
        #     symmetric block ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 2)
step = 1.0e-3
""",
            "call": "compute_block_hessian(coords, masses, _grad, step)",
            "gold_call": "_oracle_compute_block_hessian(coords, masses, _grad, step)",
        },
        # --- Edge: deuterated masses, which rescale every block entry ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 2.0141, 2.0141]), 2)
""",
            "call": "compute_block_hessian(coords, masses, _grad)",
            "gold_call": "_oracle_compute_block_hessian(coords, masses, _grad)",
        },
        # --- Edge: a stiffer supplied gradient, which must change every block.
        #     A step function that quietly differentiates its own built-in
        #     potential instead of the callable it was handed returns the
        #     previous case's answer here ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 2.0141, 2.0141]), 2)
def stiff_grad(positions):
    return _grad(positions, 900.0, 1.02)
""",
            "call": "compute_block_hessian(coords, masses, stiff_grad)",
            "gold_call": "_oracle_compute_block_hessian(coords, masses, stiff_grad)",
        },
        # --- Invalid: non-positive difference step ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
def run_model():
    try:
        compute_block_hessian(coords, masses, _grad, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_hessian(coords, masses, _grad, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the gradient is not callable ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
def run_model():
    try:
        compute_block_hessian(coords, masses, None)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_hessian(coords, masses, None)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: atom count is not a whole number of monomers ---
        {
            "setup": preamble + """
coords = np.zeros((5, 3))
masses = np.ones(5)
def run_model():
    try:
        compute_block_hessian(coords, masses, _grad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_block_hessian(coords, masses, _grad)
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
