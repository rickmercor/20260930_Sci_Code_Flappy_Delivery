"""
Reduce the displaced defect cluster to the single mass-weighted coordinate offset that couples the electronic transition to the lattice.

A colour centre couples to its lattice through the way the atoms move when the
electronic state changes. Collapsing that multidimensional relaxation onto a
single effective vibrational mode requires one number: the length of the path
between the two relaxed geometries measured in the mass-weighted coordinate

delta_Q = sqrt( sum_i m_i |r_i(excited) - r_i(ground)|^2 ),

where m_i is the mass of atom i and the displacement runs over every atom that
moves. Weighting by mass is what makes delta_Q a genuine normal-mode amplitude
rather than a bare geometric distance: a heavy antimony impurity contributes far
more per Angstrom than a light carbon neighbour, so the same visual distortion
gives a very different vibrational quantum depending on which sublattice moves.
With masses in atomic mass units and displacements in Angstrom, delta_Q carries
units of sqrt(amu) Angstrom, the convention used throughout configuration
coordinate analyses of point defects.

Returns
-------
float, the mass-weighted configuration coordinate offset in sqrt(amu) Angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mass_weighted_displacement(masses: np.ndarray, displacements: np.ndarray) -> float:
    '''Mass-weighted configuration coordinate offset between two geometries.

    Parameters
    ----------
    masses : array_like, shape (N,)
        Atomic mass of each moving atom in atomic mass units. Must be positive.
    displacements : array_like, shape (N, 3)
        Cartesian displacement of each atom in Angstrom, taken as the relaxed
        excited-state position minus the relaxed ground-state position.

    Returns
    -------
    delta_q : float
        The mass-weighted configuration coordinate offset in
        sqrt(amu) Angstrom. Raises ValueError on non-finite input, on a
        non-positive mass, or when the two arrays disagree in length or the
        displacements are not three-dimensional.
    '''
    return delta_q

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_geometry(masses, displacements):
    m = np.asarray(masses, dtype=float)
    d = np.asarray(displacements, dtype=float)
    if m.ndim != 1 or m.size == 0:
        raise ValueError("masses must be a non-empty one-dimensional array")
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError("displacements must have shape (N, 3)")
    if d.shape[0] != m.size:
        raise ValueError("masses and displacements must describe the same atoms")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(d))):
        raise ValueError("masses and displacements must be finite")
    if np.any(m <= 0.0):
        raise ValueError("atomic masses must be positive")
    return m, d


def _oracle_mass_weighted_displacement(masses: np.ndarray, displacements: np.ndarray) -> float:
    m, d = _validated_geometry(masses, displacements)
    return float(np.sqrt(np.sum(m * np.sum(d * d, axis=1))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    from textwrap import dedent
    _CLUSTER = dedent("""import numpy as np
    MASS = {"Sb": 121.760, "Si": 28.0855, "C": 12.011}
    SPECIES = ["Sb", "Si", "Si", "Si", "C", "C", "C", "C", "C", "C", "C", "C", "C"]
    DISP = np.array([
        [ 0.0258, -0.0179,  0.0409],
        [-0.0394,  0.0280, -0.0320],
        [ 0.0318, -0.0421, -0.0289],
        [ 0.0093,  0.0370,  0.0438],
        [ 0.0546, -0.0258,  0.0211],
        [-0.0471,  0.0393, -0.0179],
        [ 0.0248,  0.0511, -0.0344],
        [-0.0180, -0.0458,  0.0401],
        [ 0.0382,  0.0222,  0.0499],
        [-0.0526, -0.0169, -0.0256],
        [ 0.0154, -0.0554,  0.0196],
        [-0.0324,  0.0234,  0.0453],
        [ 0.0430, -0.0346, -0.0185]])
    M = np.array([MASS[s] for s in SPECIES])
    """).replace("\n    ", "\n")

    return [
        {   # normal: the full defect cluster of the task
            "setup": _CLUSTER,
            "call": "round(mass_weighted_displacement(M.copy(), DISP.copy()), 12)",
            "gold_call": "round(_oracle_mass_weighted_displacement(M.copy(), DISP.copy()), 12)",
        },
        {   # normal: a single heavy atom moved along one axis
            "setup": "import numpy as np\n",
            "call": "round(mass_weighted_displacement([121.760], [[0.0, 0.0, 0.13]]), 12)",
            "gold_call": "round(_oracle_mass_weighted_displacement([121.760], [[0.0, 0.0, 0.13]]), 12)",
        },
        {   # boundary: an entirely static cluster gives exactly zero
            "setup": "import numpy as np\nm = np.array([28.0855, 12.011])\nd = np.zeros((2, 3))\n",
            "call": "round(mass_weighted_displacement(m.copy(), d.copy()), 12)",
            "gold_call": "round(_oracle_mass_weighted_displacement(m.copy(), d.copy()), 12)",
        },
        {   # boundary: mass weighting must dominate, a small heavy shift against a large light one
            "setup": "import numpy as np\nm = [121.760, 12.011]\nd = [[0.02, 0.0, 0.0], [0.0, 0.09, 0.0]]\n",
            "call": "round(mass_weighted_displacement(m.copy(), d.copy()), 12)",
            "gold_call": "round(_oracle_mass_weighted_displacement(m.copy(), d.copy()), 12)",
        },
        {   # edge: a negative atomic mass must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn([28.0855, -1.0], [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(mass_weighted_displacement)",
            "gold_call": "probe(_oracle_mass_weighted_displacement)",
        },
        {   # edge: a ragged geometry must raise ValueError
            "setup": """import numpy as np
def probe(fn):
    try:
        fn([28.0855, 12.011], [[0.01, 0.0, 0.0]])
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "probe(mass_weighted_displacement)",
            "gold_call": "probe(_oracle_mass_weighted_displacement)",
        },
    ]
