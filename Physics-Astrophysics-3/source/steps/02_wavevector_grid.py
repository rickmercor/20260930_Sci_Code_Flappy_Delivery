"""
Discrete wavevector grid for the periodic unit cube.

Convention

----------

The cube is sampled at x_i = i / N for i = 0 .. N - 1, and likewise along y and

z, so the box has period 1 in each direction and a mode with signed integer

index n has physical wavenumber 2 * pi * n / L = 2 * pi * n. The component

arrays are therefore



    k1 = 2 * pi * np.fft.fftfreq(N, d=1.0/N)



on each axis, which lays the signed integer mode numbers

0, 1, ..., N/2 - 1, -N/2, ..., -1 out in the storage order that ``np.fft.fftn``

uses. The three cubes are broadcast with ``indexing="ij"`` so that axis 0 of the

returned array runs over x modes, axis 1 over y modes and axis 2 over z modes.



Every spectral operation downstream attaches these wavevectors to the Fourier

coefficients of ``np.fft.fftn``, so both the storage order and the factor of

2 * pi are part of the contract.



Inputs

------

N : int

    Number of grid points per axis, N >= 2.



Returns

-------

numpy.ndarray

    Array of shape (3, N, N, N), float64, holding KX, KY, KZ in that order.

Returns
-------
K : numpy.ndarray of shape (3, N, N, N), float64, physical wavenumbers equal to 2 pi times the integer FFT frequency, in numpy fftn storage order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def wavevector_grid(N):
    """Build the discrete wavevector components for the periodic unit cube.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the cube. Must be an integer
        greater than or equal to 2.

    Returns
    -------
    numpy.ndarray
        Array ``K`` of shape (3, N, N, N) and dtype float64 with
        ``K[0] = KX``, ``K[1] = KY`` and ``K[2] = KZ``. Each component is the
        physical wavenumber, i.e. 2 * pi times the integer FFT frequency, laid
        out in numpy ``fftn`` storage order and broadcast over the three axes
        with ``indexing="ij"``.

    Raises
    ------
    ValueError
        If ``N`` is not an integer or if ``N`` is smaller than 2.
    """
    return K  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_wavevector_grid(N):
    """Reference implementation of :func:`wavevector_grid`.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the cube. Must be an integer
        greater than or equal to 2.

    Returns
    -------
    numpy.ndarray
        Array ``K`` of shape (3, N, N, N) and dtype float64 with
        ``K[0] = KX``, ``K[1] = KY`` and ``K[2] = KZ``.

    Raises
    ------
    ValueError
        If ``N`` is not an integer or if ``N`` is smaller than 2.
    """
    if isinstance(N, bool):
        raise ValueError("N must be an integer, got a bool")
    if isinstance(N, (int, np.integer)):
        n_int = int(N)
    elif isinstance(N, (float, np.floating)):
        if not float(N).is_integer():
            raise ValueError("N must be an integer, got %r" % (N,))
        n_int = int(N)
    else:
        raise ValueError("N must be an integer, got type %s" % type(N).__name__)
    if n_int < 2:
        raise ValueError("N must be >= 2, got %d" % n_int)

    k1 = 2.0 * np.pi * np.fft.fftfreq(n_int, d=1.0 / n_int)
    KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
    K = np.empty((3, n_int, n_int, n_int), dtype=np.float64)
    K[0] = KX
    K[1] = KY
    K[2] = KZ
    return K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`wavevector_grid`."""
    cases = []

    # Normal case: generic even grid.
    cases.append({
        "setup": "import numpy as np\nN = 8",
        "call": "wavevector_grid(N)",
        "gold_call": "_oracle_wavevector_grid(N)",
    })

    # Boundary case: smallest legal grid.
    cases.append({
        "setup": "import numpy as np\nN = 2",
        "call": "wavevector_grid(N)",
        "gold_call": "_oracle_wavevector_grid(N)",
    })

    # Edge case: odd grid, where no Nyquist mode exists.
    cases.append({
        "setup": "import numpy as np\nN = 5",
        "call": "wavevector_grid(N)",
        "gold_call": "_oracle_wavevector_grid(N)",
    })

    # Invalid input: N below the minimum legal grid size.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 1\n"
            "def run_model():\n"
            "    try:\n"
            "        wavevector_grid(N)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_wavevector_grid(N)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: non-integer N.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8.5\n"
            "def run_model():\n"
            "    try:\n"
            "        wavevector_grid(N)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_wavevector_grid(N)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    return cases
