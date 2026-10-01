"""
Step 05 - Time-integrated two-photon correlations after biexciton preparation.

Time-integrated two-photon correlations of the photon pairs emitted after biexciton
preparation.

The dot starts in XX with both cavity modes empty and the lattice relaxed around XX,
which in the polaron frame of steps 02-04 is the density matrix |XX,0,0><XX,0,0| on the
13-state space of step 04 (index 5 of its basis order). It then relaxes by cavity
emission towards |G,0,0>, and the photon pairs are characterised by the equal-time
two-photon correlations

    G_ij(t) = Tr[ rho(t) a_i^dag a_i^dag a_j a_j ],      i, j in {H, V},

where a_i are the cavity annihilation operators and rho(t) evolves under the equation
of motion of step 04.

This step returns the integrals over the whole emission, integral_0^infinity G_ij(t) dt,
for G_HH, G_VV and the complex coherence G_HV (in ps). They must be exact to at least
eight significant figures, with no truncation of the time axis at that level.

Parameters are dot = [E_B (meV), delta (meV), g (meV), gamma (1/ps)] as in step 04 (here
both g and gamma must be positive, so that the biexciton empties and every correlation
decays) and the lattice
array of step 01.

Returns
-------
numpy.ndarray of shape (4,): [int G_HH dt, int G_VV dt, Re int G_HV dt, Im int G_HV dt] in ps
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrated_pair_correlations(T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Integrals over all times of G_HH, G_VV and G_HV after biexciton preparation.

    Parameters
    ----------
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    integrals : np.ndarray
        Shape (4,): [int G_HH dt, int G_VV dt, Re int G_HV dt, Im int G_HV dt], in ps.

    Raises
    ------
    ValueError
        If T is not finite and positive, if dot is not a numpy array of four finite real
        numbers with g > 0 and gamma > 0, or if lattice is invalid (as in step 01).
    '''
    return integrals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_integrated_pair_correlations(T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    T = _check_temperature(T)
    _check_dot(dot, need_loss=True)
    basis = np.zeros((169, 13, 13), dtype=complex)
    for k in range(169):
        basis[k].flat[k] = 1.0
    cols = _oracle_polaron_master_equation_rhs(basis, T, dot, lattice)
    gen = (cols[:, 0] + 1j * cols[:, 1]).reshape(169, 169).T
    # |G,0,0><G,0,0| (vector index 0) is the stationary state; it carries no two-photon
    # correlation, so drop it and solve for the time integral of everything else
    keep = np.arange(1, 169)
    start = np.zeros(169, dtype=complex)
    start[5 * 13 + 5] = 1.0
    tint = np.zeros(169, dtype=complex)
    tint[keep] = np.linalg.solve(gen[np.ix_(keep, keep)], -start[keep])
    tint = tint.reshape(13, 13)
    ghh = 2.0 * tint[8, 8].real
    gvv = 2.0 * tint[9, 9].real
    ghv = 2.0 * tint[9, 8]
    return np.array([ghh, gvv, ghv.real, ghv.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark dot and cavity at 10 K ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.2, 0.15, 0.2, 0.3])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.5 / 1.15])\n",
            "call": "integrated_pair_correlations(10.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_pair_correlations(10.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Boundary: cold lattice (2 K), weaker coupling and slower cavity ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([1.5, 0.1, 0.1, 0.25])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.0, 3.0 / 1.15])\n",
            "call": "integrated_pair_correlations(2.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_pair_correlations(2.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Edge: no fine-structure splitting, strong coupling, hot lattice ---
        {
            "setup": "import numpy as np\n"
                     "dot = np.array([0.9, 0.0, 0.3, 0.45])\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 4.0, 4.0 / 1.15])\n",
            "call": "integrated_pair_correlations(35.0, dot.copy(), lat.copy())",
            "gold_call": "_oracle_integrated_pair_correlations(35.0, dot.copy(), lat.copy())",
            "tol": 1e-7,
        },
        # --- Invalid: lossless cavity, the correlations never decay ---
        {
            "setup": "import numpy as np\n"
                     "lat = np.array([7.0, -3.5, 5370.0, 5110.0, 3.5, 3.0])\n"
                     "def run_model():\n"
                     "    try:\n"
                     "        integrated_pair_correlations(10.0, np.array([1.2, 0.15, 0.2, 0.0]), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "def run_gold():\n"
                     "    try:\n"
                     "        _oracle_integrated_pair_correlations(10.0, np.array([1.2, 0.15, 0.2, 0.0]), lat.copy())\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
