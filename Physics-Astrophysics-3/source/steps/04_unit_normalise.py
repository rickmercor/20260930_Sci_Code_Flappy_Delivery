"""
Step 04: pointwise unit-magnitude normalisation of a vector field.

Contract

--------

The input G is a (3, N, N, N) float array on the periodic unit cube,

component-first, with the spatial axes ordered (x, y, z). The return has the

same shape, dtype float64, and unit magnitude at every grid site whose input

magnitude was nonzero. The magnitude is formed pointwise as the square root of

the sum of squares over the component axis.



The one numerical convention is the treatment of grid sites where the vector is

exactly zero. Such a site has no direction to normalise to, so its magnitude is

guarded to 1.0 before the division and the zero vector is returned unchanged

rather than producing NaN.



Inputs

------

G : numpy.ndarray of shape (3, N, N, N), float64, component-first, defined on the

    periodic unit cube with grid x_i = i/N for i = 0..N-1 (likewise y_j, z_k).



Returns

-------

numpy.ndarray of shape (3, N, N, N), float64: the pointwise normalised field,

with unit magnitude at every site whose input magnitude was nonzero and the

original (zero) vector retained at every site whose input magnitude was zero.

Returns
-------
numpy.ndarray of shape (3, N, N, N), float64, the pointwise unit-magnitude field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def unit_normalise(G):
    """Divide a vector field pointwise by its magnitude.

    Parameters
    ----------
    G : numpy.ndarray
        Vector field of shape (3, N, N, N), component-first, on the periodic unit
        cube.  Sites whose magnitude is exactly zero are left unchanged.

    Returns
    -------
    numpy.ndarray
        Normalised field of shape (3, N, N, N), float64.

    Raises
    ------
    ValueError
        If the input is not a (3, N, N, N) array with equal trailing
        dimensions, or if N < 1.  The operation is pointwise and couples no
        neighbouring samples, so N = 1 is admissible here even though the
        spectral steps require N >= 2.
    """
    return normalised  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_unit_normalise(G):
    """Reference implementation of the pointwise unit-magnitude normalisation.

    Parameters
    ----------
    G : numpy.ndarray
        Vector field of shape (3, N, N, N), component-first, on the periodic unit
        cube.  Sites whose magnitude is exactly zero are left unchanged.

    Returns
    -------
    numpy.ndarray
        Normalised field of shape (3, N, N, N), float64.

    Raises
    ------
    ValueError
        If the input is not a (3, N, N, N) array with equal trailing dimensions.
    """
    arr = np.asarray(G)
    if arr.ndim != 4:
        raise ValueError("field must have 4 dimensions (3, N, N, N)")
    if arr.shape[0] != 3:
        raise ValueError("field must be component-first with shape (3, N, N, N)")
    if not (arr.shape[1] == arr.shape[2] == arr.shape[3]):
        raise ValueError("field must be defined on a cubic grid (3, N, N, N)")
    if arr.shape[1] < 1:
        raise ValueError("grid size N must be at least 1")

    field = np.asarray(arr, dtype=np.float64)
    mag = np.sqrt(np.sum(field ** 2, axis=0))
    safe = np.where(mag == 0.0, 1.0, mag)
    return field / safe

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for unit_normalise."""
    seed_setup = (
        "import numpy as np\n"
        "N = 8\n"
        "A = 20.0\n"
        "sigma = 1.0 / 30.0\n"
        "kx = 4\n"
        "coord = np.arange(N) / N\n"
        "X, Y, Z = np.meshgrid(coord, coord, coord, indexing=\"ij\")\n"
        "dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2\n"
        "phi = 2.0 * np.pi * kx * X\n"
        "env = A * np.exp(-dr2 / (2.0 * sigma ** 2))\n"
        "F0 = np.zeros((3, N, N, N), dtype=np.float64)\n"
        "F0[0] = 1.0\n"
        "F0[1] = np.cos(phi) * env\n"
        "F0[2] = np.sin(phi) * env\n"
    )

    unit_setup = (
        "import numpy as np\n"
        "N = 6\n"
        "coord = np.arange(N) / N\n"
        "X, Y, Z = np.meshgrid(coord, coord, coord, indexing=\"ij\")\n"
        "alpha = 2.0 * np.pi * X\n"
        "beta = 2.0 * np.pi * Y\n"
        "U = np.zeros((3, N, N, N), dtype=np.float64)\n"
        "U[0] = np.sin(beta) * np.cos(alpha)\n"
        "U[1] = np.sin(beta) * np.sin(alpha)\n"
        "U[2] = np.cos(beta)\n"
    )

    zero_site_setup = (
        "import numpy as np\n"
        "N = 4\n"
        "Z0 = np.zeros((3, N, N, N), dtype=np.float64)\n"
        "Z0[0] = 2.0\n"
        "Z0[1] = -3.0\n"
        "Z0[2] = 6.0\n"
        "Z0[:, 1, 2, 3] = 0.0\n"
    )

    bad_setup = (
        "import numpy as np\n"
        "N = 5\n"
        "B = np.ones((2, N, N, N), dtype=np.float64)\n"
        "def run_model():\n"
        "    try:\n"
        "        unit_normalise(B)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_unit_normalise(B)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )

    bad_rect_setup = (
        "import numpy as np\n"
        "C = np.ones((3, 4, 5, 4), dtype=np.float64)\n"
        "def run_model():\n"
        "    try:\n"
        "        unit_normalise(C)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold():\n"
        "    try:\n"
        "        _oracle_unit_normalise(C)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )

    return [
        {
            "setup": seed_setup,
            "call": "unit_normalise(F0)",
            "gold_call": "_oracle_unit_normalise(F0)",
        },
        {
            "setup": unit_setup,
            "call": "unit_normalise(U)",
            "gold_call": "_oracle_unit_normalise(U)",
        },
        {
            "setup": zero_site_setup,
            "call": "unit_normalise(Z0)",
            "gold_call": "_oracle_unit_normalise(Z0)",
        },
        {
            "setup": bad_setup,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": bad_rect_setup,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
