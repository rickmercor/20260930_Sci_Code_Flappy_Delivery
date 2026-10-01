"""
Intermolecular Ohno repulsion matrix between two identical chains stacked face to face in antiparallel orientation.

The dimer is built from two copies of the same linear chain. Molecule A keeps the geometry of the monomer: its atoms on the x axis, the first at the origin and each next one displaced by the bond lengths. Molecule B is molecule A turned by 180 degrees about the axis that is parallel to y and passes through the midpoint of A, and then lifted along z by the stacking separation. Atom k of B therefore sits at x = 2 x_c - x_k, y = 0, z = separation, where x_k is the x position of atom k of A and x_c is the mean x position of the atoms of A. The two molecules lie in parallel planes and their chain axes point in opposite directions, the card-stack arrangement in which a dipolar chromophore has no net static dipole.

This arrangement is chosen because it makes A and B equivalent by a two-fold rotation. A symmetric dimer of this kind has no first-order energy difference between the state with the excitation on A and the state with the excitation on B, so the splitting of the pair of locally excited states is controlled by the excitonic coupling alone.

Between atom mu of A and atom nu of B the pi electrons repel through the same Ohno interpolation that acts inside each molecule, gamma = 14.397 / sqrt(r^2 + a^2) with a = 2 * 14.397 / (U_mu + U_nu), r in angstrom and gamma in eV. With zero differential overlap there is no resonance integral and no exchange integral between the molecules.

Returns
-------
numpy.ndarray of shape (n_sites, n_sites): gamma_AB[mu, nu], the Ohno repulsion between atom mu of molecule A and atom nu of molecule B, in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def antiparallel_stack_interactions(bonds: "np.ndarray", hubbard_u: "np.ndarray",
                                    separation: float) -> "np.ndarray":
    '''Ohno repulsion between the atoms of A and of its antiparallel stacked copy B.

    Parameters
    ----------
    bonds : numpy.ndarray
        One-dimensional array of the n_sites - 1 bond lengths of the chain, in
        angstrom; all must be positive.
    hubbard_u : numpy.ndarray
        One-centre repulsion of each atom, in eV, length n_sites; all must be positive.
    separation : float
        Distance between the planes of the two molecules along z, in angstrom; must
        be positive.

    Returns
    -------
    gamma_ab : numpy.ndarray
        Real array of shape (n_sites, n_sites) whose element [mu, nu] is the Ohno
        repulsion between atom mu of molecule A and atom nu of molecule B.

    Raises
    ------
    ValueError
        If bonds is empty or has a non-positive entry, if hubbard_u does not have
        length n_sites or has a non-positive entry, or if separation is not a
        positive finite number.
    '''
    return gamma_ab

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _antiparallel_partner(bonds, separation):
    import numpy as np
    XA = _chain_coordinates(bonds)
    xc = XA[:, 0].mean()
    XB = np.column_stack([2.0 * xc - XA[:, 0], XA[:, 1], XA[:, 2] + float(separation)])
    return XA, XB


def _oracle_antiparallel_stack_interactions(bonds: "np.ndarray", hubbard_u: "np.ndarray",
                                            separation: float) -> "np.ndarray":
    import numpy as np
    b = np.atleast_1d(np.asarray(bonds, dtype=float))
    u = np.atleast_1d(np.asarray(hubbard_u, dtype=float))
    if b.ndim != 1 or b.size == 0 or not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError("bonds must be a non-empty array of positive lengths")
    if u.shape != (b.size + 1,) or not np.all(np.isfinite(u)) or np.any(u <= 0.0):
        raise ValueError("hubbard_u must hold one positive value per site")
    if not np.isfinite(separation) or separation <= 0.0:
        raise ValueError("separation must be a positive finite number")
    XA, XB = _antiparallel_partner(b, separation)
    return _ohno_kernel(XA, u, XB, u)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the benchmark C=C-N stack at 7 angstrom ---
        {
            "setup": """import numpy as np
""",
            "call": "antiparallel_stack_interactions(np.array([1.34, 1.40]), np.array([11.13, 11.13, 16.76]), 7.0)",
            "gold_call": "_oracle_antiparallel_stack_interactions(np.array([1.34, 1.40]), np.array([11.13, 11.13, 16.76]), 7.0)",
            "tol": 1e-12,
        },
        # --- Normal: a butadiene stack at van der Waals contact ---
        {
            "setup": """import numpy as np
""",
            "call": "antiparallel_stack_interactions(np.array([1.35, 1.46, 1.35]), np.full(4, 11.13), 3.5)",
            "gold_call": "_oracle_antiparallel_stack_interactions(np.array([1.35, 1.46, 1.35]), np.full(4, 11.13), 3.5)",
            "tol": 1e-12,
        },
        # --- Boundary: a two-atom polar unit, whose rotated copy swaps the two ends ---
        {
            "setup": """import numpy as np
""",
            "call": "antiparallel_stack_interactions(np.array([1.34]), np.array([11.13, 12.34]), 5.0)",
            "gold_call": "_oracle_antiparallel_stack_interactions(np.array([1.34]), np.array([11.13, 12.34]), 5.0)",
            "tol": 1e-12,
        },
        # --- Edge: large separation, approaching the bare Coulomb tail ---
        {
            "setup": """import numpy as np
""",
            "call": "antiparallel_stack_interactions(np.array([2.5, 0.9]), np.array([6.0, 11.13, 19.0]), 60.0)",
            "gold_call": "_oracle_antiparallel_stack_interactions(np.array([2.5, 0.9]), np.array([6.0, 11.13, 19.0]), 60.0)",
            "tol": 1e-12,
        },
        # --- Invalid: a zero stacking separation ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        antiparallel_stack_interactions(np.array([1.34, 1.40]), np.array([11.13, 11.13, 16.76]), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_antiparallel_stack_interactions(np.array([1.34, 1.40]), np.array([11.13, 11.13, 16.76]), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
