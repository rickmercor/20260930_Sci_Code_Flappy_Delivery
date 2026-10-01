"""
Build the discrete-ordinates set for one-dimensional cross-plane transport, returning the direction cosines and the solid-angle weights that integrate a distribution over the full sphere.

The Boltzmann equation is posed on the full sphere of propagation directions, and a deterministic solver replaces that sphere by a finite set of ordinates with quadrature weights. When the geometry depends on a single spatial coordinate the distribution can only depend on the angle between the propagation direction and that coordinate, so the azimuthal integral is exact and contributes a factor of two pi, and what remains is a one-dimensional integral over the direction cosine on the interval from minus one to one. Gauss-Legendre nodes are the canonical choice there: with a given number of ordinates they integrate the highest possible polynomial degree exactly, they are symmetric about zero so that forward and backward hemispheres are treated identically, and no node ever lands on the cosine zero, which would correspond to a direction transporting nothing along the coordinate and would make the transport operator singular.




The weights matter as much as the nodes because two different integrals are taken against them downstream, and both must be consistent with the way the equilibrium distribution is normalised. The angular integral of the energy density gives the energy per unit volume, and the angular integral of the energy density times the velocity component along the coordinate gives the heat flux; the equilibrium distribution is reconstructed by dividing the product of the band heat capacity and the lattice temperature by the total angular weight, so that its angular integral returns exactly the heat capacity times the temperature. Carrying the azimuthal factor of two pi inside the weights is what makes the total weight equal the four pi steradians of the full sphere and keeps all three statements consistent at once. Dropping it, or normalising the weights to unity, changes the ratio between the streaming and the scattering terms and silently rescales the mean free paths of the problem.




An odd number of ordinates would place a node at zero cosine, and a single ordinate cannot represent both hemispheres, so the ordinate count is required to be even and at least two.

Returns
-------
np.ndarray of shape (n_dirs, 2), float: direction cosines in column 0 and solid-angle weights in steradians in column 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_angular_quadrature(n_dirs: int) -> np.ndarray:
    """Build the one-dimensional discrete-ordinates set.

    Parameters
    ----------
    n_dirs : int
        Number of ordinates; must be even and at least 2.

    Returns
    -------
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) whose first column holds the direction
        cosines and whose second column holds the solid-angle weights, the
        weights summing to the full solid angle.

    Raises
    ------
    ValueError
        If ``n_dirs`` is not an even integer at least 2.
    """
    return quadrature  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_angular_quadrature(n_dirs: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    if not (isinstance(n_dirs, (int, np.integer)) and not isinstance(n_dirs, bool)
            and int(n_dirs) >= 2 and int(n_dirs) % 2 == 0):
        raise ValueError("n_dirs must be an even integer >= 2")

    cosines, weights = np.polynomial.legendre.leggauss(int(n_dirs))

    # The azimuthal integral is exact and contributes 2*pi, so the weights
    # sum to the 4*pi steradians of the full sphere.
    return np.column_stack([cosines, 2.0 * np.pi * weights])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark ordinate count (normal scenario) ---
        {
            "setup": """import numpy as np
n_dirs = 16
""",
            "call": "build_angular_quadrature(n_dirs)",
            "gold_call": "_oracle_build_angular_quadrature(n_dirs)",
        },
        # --- Valid: refined ordinate set ---
        {
            "setup": """import numpy as np
n_dirs = 48
""",
            "call": "build_angular_quadrature(n_dirs)",
            "gold_call": "_oracle_build_angular_quadrature(n_dirs)",
        },
        # --- Boundary: the smallest admissible set, one ordinate per hemisphere ---
        {
            "setup": """import numpy as np
n_dirs = 2
""",
            "call": "build_angular_quadrature(n_dirs)",
            "gold_call": "_oracle_build_angular_quadrature(n_dirs)",
        },
        # --- Edge: large ordinate set where the outermost cosines approach unity ---
        {
            "setup": """import numpy as np
n_dirs = 128
""",
            "call": "build_angular_quadrature(n_dirs)",
            "gold_call": "_oracle_build_angular_quadrature(n_dirs)",
        },
        # --- Invalid: odd ordinate count would place a node at zero cosine ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_angular_quadrature(15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_angular_quadrature(15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a single ordinate cannot represent both hemispheres ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_angular_quadrature(1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_angular_quadrature(1)
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
