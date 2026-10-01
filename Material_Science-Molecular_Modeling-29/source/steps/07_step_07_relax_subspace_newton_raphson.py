"""
Drive the monomers to mechanical equilibrium along their stiff degrees of freedom with a fixed number of Newton-Raphson steps preconditioned by the subspace pseudoinverse.

The manifold that the coarse-grained model integrates over is the set of configurations at which the potential gradient vanishes when projected onto the stiff subspace: equilibrium is imposed along the fast coordinates only, while the slow coordinates are free and are exactly the coarse-grained degrees of freedom. Sampling that manifold directly is impractical, so it is approached indirectly, by starting from the zeroth-order manifold and taking a small, fixed number of Newton steps towards it.

The step is a Newton step in mass-scaled coordinates. The gradient is scaled into the mass-weighted metric, hit with the subspace pseudoinverse, and scaled back into Cartesian displacements; the composition of the three operations is applied to the Cartesian gradient and subtracted from the current positions. The double mass-scaling is not decorative: the pseudoinverse was diagonalized in the mass-weighted metric, so a gradient entering it in any other metric is being projected onto the wrong subspace, and the displacement leaks into the rigid-body directions that must remain untouched.

Two economies define the algorithm and both affect the answer. First, the iteration count is fixed in advance instead of being run to a convergence tolerance. One step is exact for a harmonic surface; two steps hold the coarse-grained free energy of a water cluster to about a wavenumber, which is why two is the working choice. Continuing to convergence would cost more gradient evaluations per configuration for no useful accuracy, and would make the cost of a coarse-grained energy evaluation configuration-dependent. Second, the pseudoinverse is built once, at the starting configuration, and reused unchanged at every step rather than being rebuilt from the curvature at the current point. This is what reduces the work to one block-Hessian construction for the whole relaxation, and it is justified by the observation that for systems with a clean fast/slow separation the stiff subspace barely rotates over the short distance the relaxation travels.

The displacement produced here is small but is not a refinement that could be skipped. Freezing the monomers instead, which is what stopping at zero steps means, misplaces the intramolecular geometry in the field of the neighbours, and both the potential energy and the stiff-mode frequencies inherit that error.

Returns
-------
np.ndarray of shape (n_atoms, 3), float: the subspace-relaxed cluster configuration, in angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def relax_subspace_newton_raphson(coords: np.ndarray, pseudoinverse: np.ndarray,
                                  atom_masses: np.ndarray, gradient_fn,
                                  n_iterations: int = 2) -> np.ndarray:
    """Relax a cluster along its stiff subspace by fixed-count Newton steps.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3 * n_monomers, 3) in angstrom holding the starting
        configuration, ordered O, H, H within each monomer.
    pseudoinverse : np.ndarray
        Array of shape (n_monomers, 9, 9) holding the stiff-subspace
        pseudoinverse of each mass-scaled Hessian block, built once at the
        starting configuration and held fixed throughout.
    atom_masses : np.ndarray
        Array of shape (3 * n_monomers,) in unified atomic mass units holding
        the mass of every atom, in the same order as ``coords``. All entries
        must be strictly positive.
    gradient_fn : callable
        Analytic gradient of the potential being relaxed on: it maps an array
        of positions of the same shape as ``coords`` to the derivative of the
        energy with respect to every atomic coordinate, an array of that same
        shape. It must be the gradient of the same potential whose curvature
        produced ``pseudoinverse``. For the cluster of this task it is the
        function built in sub-problem 03, called with its default q-TIP4P/F
        parameters.
    n_iterations : int
        Number of Newton-Raphson iterations to perform (n_iterations >= 0);
        zero returns the starting configuration unchanged.

    Returns
    -------
    relaxed_coords : np.ndarray
        Array with the same shape as ``coords`` in angstrom, holding the
        configuration after the prescribed number of subspace Newton steps.

    Raises
    ------
    ValueError
        If the coordinate, pseudoinverse, or mass arrays have incompatible
        shapes or non-finite entries; if any mass is non-positive; if
        ``gradient_fn`` is not callable; or if ``n_iterations`` is not a
        non-negative integer.
    """
    return relaxed_coords  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_relax_subspace_newton_raphson(coords: np.ndarray, pseudoinverse: np.ndarray,
                                          atom_masses: np.ndarray, gradient_fn,
                                          n_iterations: int = 2) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    coords = np.asarray(coords, dtype=float)
    pseudoinverse = np.asarray(pseudoinverse, dtype=float)
    atom_masses = np.asarray(atom_masses, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (n_atoms, 3)")
    if coords.shape[0] < 3 or coords.shape[0] % 3 != 0:
        raise ValueError("coords must hold a whole number of three-atom monomers")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite entries")
    n_monomers = coords.shape[0] // 3
    if pseudoinverse.shape != (n_monomers, 9, 9):
        raise ValueError("pseudoinverse must have shape (n_monomers, 9, 9)")
    if not np.all(np.isfinite(pseudoinverse)):
        raise ValueError("pseudoinverse must contain only finite entries")
    if atom_masses.ndim != 1 or atom_masses.size != coords.shape[0]:
        raise ValueError("atom_masses must have one entry per atom")
    if not np.all(np.isfinite(atom_masses)) or np.any(atom_masses <= 0.0):
        raise ValueError("atom_masses must be finite and strictly positive")
    if not callable(gradient_fn):
        raise ValueError("gradient_fn must be callable")
    if not (isinstance(n_iterations, (int, np.integer)) and not isinstance(n_iterations, bool)
            and int(n_iterations) >= 0):
        raise ValueError("n_iterations must be an integer >= 0")

    n_iterations = int(n_iterations)

    relaxed_coords = coords.copy()

    for _ in range(n_iterations):
        gradient = gradient_fn(relaxed_coords)
        displacement = np.zeros_like(relaxed_coords)
        for mono in range(n_monomers):
            rows = np.arange(3 * mono, 3 * mono + 3)
            inverse_root = 1.0 / np.sqrt(np.repeat(atom_masses[rows], 3))
            scaled_gradient = inverse_root * gradient[rows].reshape(9)
            displacement[rows] = (inverse_root
                                  * (pseudoinverse[mono] @ scaled_gradient)).reshape(3, 3)
        relaxed_coords = relaxed_coords - displacement

    return relaxed_coords

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    # Shared setup: a valid stiff-subspace pseudoinverse is built from a
    # prescribed spectrum using numpy alone, and the gradient the relaxation
    # follows is supplied explicitly, so each test stands alone.
    preamble = """import numpy as np

def _pseudoinverse(n_mono, spectrum, seed):
    rng = np.random.default_rng(seed)
    pinv = np.zeros((n_mono, 9, 9))
    for mono in range(n_mono):
        basis = np.linalg.qr(rng.normal(size=(9, 9)))[0]
        values = np.array(spectrum, dtype=float)
        columns = basis[:, :values.size]
        pinv[mono] = (columns / values) @ columns.T
    return pinv

def _grad(positions, bond_force=550.0, bond_length=0.96):
    # Gradient of a compact test surface standing in for the all-atom one: two
    # harmonic O-H bonds and a harmonic H-H strut inside each monomer, plus an
    # inverse-cube repulsion between every pair of atoms on different monomers.
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
        # --- Valid: the benchmark tetramer, two iterations (normal scenario) ---
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
pinv = _pseudoinverse(coords.shape[0] // 3, [1295.7, 1238.7, 213.8], 17)
""",
            "call": "relax_subspace_newton_raphson(coords, pinv, masses, _grad, 2)",
            "gold_call": "_oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, 2)",
        },
        # --- Boundary: zero iterations must return the input untouched ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 2)
pinv = _pseudoinverse(coords.shape[0] // 3, [1295.7, 1238.7, 213.8], 17)
""",
            "call": "relax_subspace_newton_raphson(coords, pinv, masses, _grad, 0)",
            "gold_call": "_oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, 0)",
        },
        # --- Valid: a longer relaxation on a dimer, where the fixed
        #     pseudoinverse must not be refreshed between steps ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 2)
pinv = _pseudoinverse(coords.shape[0] // 3, [1295.7, 1238.7, 213.8], 17)
""",
            "call": "relax_subspace_newton_raphson(coords, pinv, masses, _grad, 6)",
            "gold_call": "_oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, 6)",
        },
        # --- Edge: an isolated monomer far from its intramolecular minimum,
        #     where the Newton step is large ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [1.18, 0.0, 0.0], [-0.42, 0.82, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
pinv = _pseudoinverse(coords.shape[0] // 3, [1295.7, 1238.7, 213.8], 17)
""",
            "call": "relax_subspace_newton_raphson(coords, pinv, masses, _grad, 3)",
            "gold_call": "_oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, 3)",
        },
        # --- Valid: a stiffer supplied gradient, which must change the step.
        #     A step function that quietly differentiates its own built-in
        #     potential instead of the callable it was handed returns the
        #     previous case's answer here ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [1.18, 0.0, 0.0], [-0.42, 0.82, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
pinv = _pseudoinverse(coords.shape[0] // 3, [1295.7, 1238.7, 213.8], 17)
def stiff_grad(positions):
    return _grad(positions, 900.0, 1.02)
""",
            "call": "relax_subspace_newton_raphson(coords, pinv, masses, stiff_grad, 3)",
            "gold_call": "_oracle_relax_subspace_newton_raphson(coords, pinv, masses, stiff_grad, 3)",
        },
        # --- Invalid: the gradient is not callable ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
pinv = np.zeros((1, 9, 9))
def run_model():
    try:
        relax_subspace_newton_raphson(coords, pinv, masses, None, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_relax_subspace_newton_raphson(coords, pinv, masses, None, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative iteration count ---
        {
            "setup": preamble + """
coords = np.array([[0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0]])
masses = np.array([15.9994, 1.00794, 1.00794])
pinv = np.zeros((1, 9, 9))
def run_model():
    try:
        relax_subspace_newton_raphson(coords, pinv, masses, _grad, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: pseudoinverse block count inconsistent with the cluster ---
        {
            "setup": preamble + """
coords = np.array([
    [0.0, 0.0, 0.0], [0.9419, 0.0, 0.0], [-0.2836, 0.8983, 0.0],
    [2.85, 0.15, 0.10], [3.55, 0.62, 0.55], [2.70, -0.55, 0.60]])
masses = np.tile(np.array([15.9994, 1.00794, 1.00794]), 2)
pinv = np.zeros((1, 9, 9))
def run_model():
    try:
        relax_subspace_newton_raphson(coords, pinv, masses, _grad, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_relax_subspace_newton_raphson(coords, pinv, masses, _grad, 2)
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
