"""
Step 01: build the seed vector field F0 that the rest of the pipeline starts

from.

Grid, layout and dtype

----------------------

The domain is the periodic unit cube sampled at x_i = i/N for i = 0..N-1, and

likewise for y_j and z_k. Coordinate cubes are built with

np.meshgrid(coord, coord, coord, indexing="ij"), so the returned array is

indexed as F0[component, i, j, k] with i along x, j along y and k along z. The

return is float64 with shape (3, N, N, N), component-first.



Definition

----------

The seed is fixed by the problem statement: a uniform background of unit

strength along x plus a circularly polarised transverse perturbation of peak

amplitude A under a Gaussian envelope of width sigma centred on

(0.5, 0.5, 0.5), rotating with axial wavenumber kx.



    F0[0] = 1.0

    F0[1] = cos(phi) * env

    F0[2] = sin(phi) * env



    phi  = 2 * pi * kx * X

    env  = A * exp(-dr2 / (2 * sigma**2))

    dr2  = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2



Inputs

------

N : int

    Number of grid points per axis (N >= 2). The grid is periodic with

    spacing 1/N.

A : float

    Peak amplitude of the transverse perturbation, in units of the background

    field strength. Must be finite. A = 0 leaves the uniform background alone.

sigma : float

    Gaussian envelope width, in units of the box length. Must be strictly

    positive.

kx : int

    Axial wavenumber of the perturbation, in cycles per box length. Must be a

    non-negative integer. kx = 0 gives phi = 0 everywhere, so the perturbation

    lies along y and the z component is identically zero.



Returns

-------

numpy.ndarray

    The seed field F0 with shape (3, N, N, N) and dtype float64, laid out

    component-first.

Returns
-------
F0 : numpy.ndarray of shape (3, N, N, N), dtype float64, component-first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def seed_field(N, A, sigma, kx):
    """Build the (3, N, N, N) seed field F0 on the periodic unit cube.

    The field is a uniform background of unit strength along x plus a
    circularly polarised transverse perturbation of peak amplitude A, confined
    by a Gaussian envelope of width sigma centred at (0.5, 0.5, 0.5) and
    rotating with axial wavenumber kx:

        F0[0] = 1.0
        F0[1] = cos(phi) * env
        F0[2] = sin(phi) * env
        phi   = 2 * pi * kx * X
        env   = A * exp(-dr2 / (2 * sigma**2))
        dr2   = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2

    Coordinates are x_i = i/N for i = 0..N-1 on every axis, combined with
    np.meshgrid(coord, coord, coord, indexing="ij").

    Parameters
    ----------
    N : int
        Grid points per axis. Must be an integer with N >= 2.
    A : float
        Peak amplitude of the transverse perturbation. Must be finite.
    sigma : float
        Gaussian envelope width in box-length units. Must be > 0.
    kx : int
        Axial wavenumber in cycles per box length. Must be a non-negative
        integer.

    Returns
    -------
    F0 : numpy.ndarray
        Array of shape (3, N, N, N) and dtype float64 holding the seed field,
        component-first.

    Raises
    ------
    ValueError
        If N is not an integer >= 2, if sigma is not strictly positive and
        finite, if A is not finite, or if kx is not a non-negative integer.
    """
    return F0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_seed_field(N, A, sigma, kx):
    """Reference implementation of seed_field (deterministic)."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)):
        raise ValueError("N must be an integer")
    N = int(N)
    if N < 2:
        raise ValueError("N must be >= 2")

    if isinstance(kx, bool) or not isinstance(kx, (int, np.integer)):
        raise ValueError("kx must be an integer")
    kx = int(kx)
    if kx < 0:
        raise ValueError("kx must be non-negative")

    try:
        A = float(A)
    except (TypeError, ValueError):
        raise ValueError("A must be a real number")
    if not np.isfinite(A):
        raise ValueError("A must be finite")

    try:
        sigma = float(sigma)
    except (TypeError, ValueError):
        raise ValueError("sigma must be a real number")
    if not np.isfinite(sigma):
        raise ValueError("sigma must be finite")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")

    coord = np.arange(N, dtype=np.float64) / float(N)
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")

    dr2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    env = A * np.exp(-dr2 / (2.0 * sigma ** 2))
    phi = 2.0 * np.pi * float(kx) * X

    F0 = np.empty((3, N, N, N), dtype=np.float64)
    F0[0] = 1.0
    F0[1] = np.cos(phi) * env
    F0[2] = np.sin(phi) * env
    return F0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for seed_field."""
    cases = []

    # Normal case: the reference configuration on a small grid.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
        ),
        "call": "seed_field(N, A, sigma, kx)",
        "gold_call": "_oracle_seed_field(N, A, sigma, kx)",
    })

    # Boundary case: zero amplitude leaves the pure uniform background field.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 6\n"
            "A = 0.0\n"
            "sigma = 0.1\n"
            "kx = 3\n"
        ),
        "call": "seed_field(N, A, sigma, kx)",
        "gold_call": "_oracle_seed_field(N, A, sigma, kx)",
    })

    # Edge case: kx = 0 makes phi vanish, so the perturbation is purely
    # along y and the z component is identically zero.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 10\n"
            "A = 5.0\n"
            "sigma = 0.2\n"
            "kx = 0\n"
        ),
        "call": "seed_field(N, A, sigma, kx)",
        "gold_call": "_oracle_seed_field(N, A, sigma, kx)",
    })

    # Invalid input: sigma = 0 is not a legal envelope width.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 0.0\n"
            "kx = 4\n"
            "def run_model():\n"
            "    try:\n"
            "        seed_field(N, A, sigma, kx)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_seed_field(N, A, sigma, kx)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: N = 1 is below the minimum grid size.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 1\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "def run_model():\n"
            "    try:\n"
            "        seed_field(N, A, sigma, kx)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_seed_field(N, A, sigma, kx)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    return cases
