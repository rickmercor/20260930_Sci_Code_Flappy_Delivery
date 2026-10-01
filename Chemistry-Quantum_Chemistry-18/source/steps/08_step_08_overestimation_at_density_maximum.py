"""
Step 08: Overestimation at the radial density maximum and its large-N coefficient (orchestrator). Orchestrator: overestimation of the reduced on-top density by a thresholded natural-orbital expansion at the maximum of the radial electron density, at a finite threshold and in the large-N limit.

The electrons of the trap are found most often on the shell where the radial density 4 pi r^2 rho1(r) is largest, at a
radius fixed by the exact one-electron density. There the reduced on-top density is much smaller than at the centre,
because correlation pushes the second electron to the far side of the trap, and a natural-orbital expansion has to build
that depletion out of orbitals of many angular momenta.

At a finite threshold the ratio of the truncated estimate to the exact value measures the error of an expansion of a
given size. Its approach to one as the retained set grows is algebraic and slow, and the coefficient of that approach is
not accessible by fitting the finite-threshold sequence at the sizes reachable here; it has to come from the asymptotic
behaviour of the weakly occupied orbitals, which is fixed by the on-top density of the same state.

Returns
-------
numpy.ndarray [r*, F, G]: radius of maximal radial density, overestimation ratio at the threshold and large-N coefficient G
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def overestimation_at_density_maximum(omega: float, c: float, occ_min: float) -> "np.ndarray":
    '''Radius r* of maximal radial density, the ratio F at threshold occ_min, and the large-N coefficient G.

    Parameters
    ----------
    omega : float
        Gaussian exponent parameter, 0.05 <= omega <= 5 (atomic units).
    c : float
        Coefficient of r12^2 in p(s) = 1 + s/2 + c s^2, 0 <= c <= 0.5.
    occ_min : float
        Occupation threshold, 1e-9 <= occ_min <= 0.1; no occupation of the pair function lies within 10 percent of it.

    Returns
    -------
    result : np.ndarray
        Float array [r*, F, G]: r* in bohr accurate to 1e-6, F = phi_N(r*) / phi(r*) at the given threshold, accurate
        to 1e-6 relative, and G the coefficient of the large-N law phi_N(r*) / phi(r*) = 1 + G N^(-1/3) + o(N^(-1/3)),
        accurate to 1e-6 relative. G is the value implied by the asymptotics of the weakly occupied natural orbitals
        evaluated with the exact on-top density at r*, not a fit to the finite-threshold sequence.

    Raises
    ------
    ValueError
        If occ_min lies outside its range.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize_scalar


def _oracle_overestimation_at_density_maximum(omega: float, c: float, occ_min: float) -> "np.ndarray":
    """Reference implementation."""
    if not 1e-9 <= float(occ_min) <= 0.1:
        raise ValueError("occ_min out of range")
    omega = float(omega)
    upper = 5.0 / np.sqrt(omega)
    radial = lambda x: -x * x * _oracle_radial_densities(omega, c, x)[1]
    grid = np.linspace(0.05 * upper, 0.95 * upper, 37)
    best = grid[int(np.argmin([radial(x) for x in grid]))]
    step = grid[1] - grid[0]
    found = minimize_scalar(radial, bounds=(best - step, best + step), method="bounded", options={"xatol": 1e-9})
    r_star = float(found.x)
    estimate = _oracle_truncated_reduced_ontop(omega, c, r_star, occ_min)[0]
    densities = _oracle_radial_densities(omega, c, r_star)
    ratio = estimate / densities[2]
    constants = _oracle_weak_orbital_asymptotics(omega, c)
    volume, amplitude = constants[0], constants[2]
    coefficient = 6.0 * amplitude * densities[0] ** -0.125 / volume
    return np.array([r_star, ratio, coefficient])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: exact omega = 1/10 ground state, moderate threshold ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([v[0], np.log(v[1]), np.log(v[2])])\n",
            "call": "parts(overestimation_at_density_maximum(0.1, 0.05, 3e-6))",
            "gold_call": "parts(_oracle_overestimation_at_density_maximum(0.1, 0.05, 3e-6))",
            "tol": 1e-6,
        },
        # --- Normal: exact omega = 1/2 ground state ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([v[0], np.log(v[1]), np.log(v[2])])\n",
            "call": "parts(overestimation_at_density_maximum(0.5, 0.0, 3e-8))",
            "gold_call": "parts(_oracle_overestimation_at_density_maximum(0.5, 0.0, 3e-8))",
            "tol": 1e-6,
        },
        # --- Boundary: tight trap with a quadratic term and a coarse threshold ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([v[0], np.log(v[1]), np.log(v[2])])\n",
            "call": "parts(overestimation_at_density_maximum(3.0, 0.1, 1e-4))",
            "gold_call": "parts(_oracle_overestimation_at_density_maximum(3.0, 0.1, 1e-4))",
            "tol": 1e-6,
        },
        # --- Edge: the loosest trap and the deepest threshold the contract allows, where the internal
        #     angular-momentum limit comes closest to binding ---
        {
            "setup": "import numpy as np\ndef parts(values):\n    v = np.asarray(values, dtype=float).reshape(3)\n    return np.array([v[0], np.log(v[1]), np.log(v[2])])\n",
            "call": "parts(overestimation_at_density_maximum(0.05, 0.3, 1e-9))",
            "gold_call": "parts(_oracle_overestimation_at_density_maximum(0.05, 0.3, 1e-9))",
            "tol": 1e-6,
        },
        # --- Error: a threshold below 1e-9 must raise ValueError ---
        {
            "setup": "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, 0.0, 1e-12)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    return 0\n",
            "call": "_probe(overestimation_at_density_maximum)",
            "gold_call": "_probe(_oracle_overestimation_at_density_maximum)",
        },
    ]
