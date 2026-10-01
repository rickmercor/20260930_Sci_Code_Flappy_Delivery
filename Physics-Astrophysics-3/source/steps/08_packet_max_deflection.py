"""
Step 08: the end-to-end driver for the construction and its maximum deflection

angle.

Contract

--------

Starting from the step-01 seed built with (N, A, sigma, kx), run exactly n_iter

cycles of the construction and return the maximum deflection angle of the

delivered field from the background direction xhat, in degrees, as a Python

float.



Each cycle applies one solenoidal projection (step 03) and one pointwise

normalisation (step 04). The angle field is the one step 05 computes, and the

maximum is taken over every grid site of the (N, N, N) spatial grid.



The driver composes the public functions of steps 01-07; it does not

re-implement them.



Inputs

------

N : int

    Number of grid points per axis of the periodic unit cube, N >= 2.

A : float

    Amplitude of the Gaussian envelope of the seed field.

sigma : float

    Width of the Gaussian envelope, sigma > 0.

kx : int

    Integer number of transverse rotations of the seed along x.

n_iter : int

    Number of cycles to run, n_iter >= 1.



Returns

-------

float

    The maximum deflection angle in degrees, taken over every grid site of the

    delivered field.

Returns
-------
float : maximum deflection angle in degrees over all grid sites of the delivered field.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def packet_max_deflection(N, A, sigma, kx, n_iter):
    """Run the construction end to end and report its peak deflection.

    The seed of step 01 is built from ``N``, ``A``, ``sigma`` and ``kx`` and
    relaxed by ``n_iter`` cycles, each applying one solenoidal projection
    (step 03) and one pointwise normalisation (step 04). The returned value is
    the maximum over the spatial grid of the step-05 deflection angle of the
    delivered field, in degrees.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the periodic unit cube. Must be
        an integer greater than or equal to 2.
    A : float
        Amplitude of the Gaussian envelope of the seed field.
    sigma : float
        Width of the Gaussian envelope of the seed field. Must be positive.
    kx : int
        Integer number of transverse rotations of the seed along x.
    n_iter : int
        Number of projection/normalisation cycles. Must be an integer greater
        than or equal to 1.

    Returns
    -------
    float
        Maximum deflection angle in degrees over all grid sites of the
        delivered field.

    Raises
    ------
    ValueError
        If ``n_iter`` is not an integer or is smaller than 1, or if any of
        ``N``, ``A``, ``sigma`` or ``kx`` fails the seed-field validation.
    """
    return theta_max  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_packet_max_deflection(N, A, sigma, kx, n_iter):
    """Reference implementation of :func:`packet_max_deflection`.

    Parameters
    ----------
    N : int
        Number of grid points along each axis of the periodic unit cube. Must be
        an integer greater than or equal to 2.
    A : float
        Amplitude of the Gaussian envelope of the seed field.
    sigma : float
        Width of the Gaussian envelope of the seed field. Must be positive.
    kx : int
        Integer number of transverse rotations of the seed along x.
    n_iter : int
        Number of projection/normalisation cycles. Must be an integer greater
        than or equal to 1.

    Returns
    -------
    float
        Maximum deflection angle in degrees over all grid sites of the final
        projected field ``G``.

    Raises
    ------
    ValueError
        If ``n_iter`` is not an integer or is smaller than 1, or if any of
        ``N``, ``A``, ``sigma`` or ``kx`` fails the seed-field validation.
    """
    if isinstance(n_iter, bool):
        raise ValueError("n_iter must be an integer, got a bool")
    if isinstance(n_iter, (int, np.integer)):
        n_steps = int(n_iter)
    elif isinstance(n_iter, (float, np.floating)):
        if not float(n_iter).is_integer():
            raise ValueError("n_iter must be an integer, got %r" % (n_iter,))
        n_steps = int(n_iter)
    else:
        raise ValueError(
            "n_iter must be an integer, got type %s" % type(n_iter).__name__
        )
    if n_steps < 1:
        raise ValueError("n_iter must be >= 1, got %d" % n_steps)

    # The seed builder performs the step-01 validation of N, A, sigma and kx and
    # raises ValueError on bad input.  The golden composes the pipeline from the
    # oracle implementations of steps 01-07 directly, so the reference answer
    # never depends on a submitted public function.
    F = _oracle_seed_field(N, A, sigma, kx)

    # n_steps >= 1, so the loop always runs and G is the final projected field.
    G = np.asarray(F, dtype=np.float64)
    for _ in range(n_steps):
        G = _oracle_solenoidal_projection(F)
        F = _oracle_unit_normalise(G)

    theta = _oracle_deflection_angles(G)

    # Certify the delivered iterate against the two constraints before
    # reporting: the spectral grid must match the field, the hard constraint
    # must be measurable, and the soft-constraint residual must be finite.  A
    # non-finite diagnostic means the pipeline was corrupted upstream.
    K = _oracle_wavevector_grid(G.shape[1])
    div_max = _oracle_max_divergence(G)
    defect = _oracle_magnitude_defect(G)
    if K.shape[1:] != G.shape[1:]:
        raise ValueError("wavevector grid does not match the field grid")
    if not (np.isfinite(div_max) and np.isfinite(defect)
            and np.all(np.isfinite(theta))):
        raise ValueError("non-finite diagnostic in the delivered field")

    # The hard constraint is imposed exactly, so the delivered field is
    # solenoidal to round-off at every cycle count: the largest value over
    # every shipped configuration is 1.4e-13.  Refuse to report an angle
    # for a field that fails it.  The soft constraint is not gated here -
    # the magnitude defect is still 0.44 after a single cycle and is only
    # approached asymptotically.
    if div_max > 1.0e-6:
        raise ValueError(
            "delivered field is not solenoidal: max|div| = %.3e" % div_max
        )

    return float(np.max(theta))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for :func:`packet_max_deflection`."""
    cases = []

    # Normal case: the reference configuration on a small grid.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 3"
        ),
        "call": "packet_max_deflection(N, A, sigma, kx, n_iter)",
        "gold_call": "_oracle_packet_max_deflection(N, A, sigma, kx, n_iter)",
    })

    # Normal case: a broader, weaker packet on a different grid.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 12\n"
            "A = 5.0\n"
            "sigma = 0.1\n"
            "kx = 2\n"
            "n_iter = 4"
        ),
        "call": "packet_max_deflection(N, A, sigma, kx, n_iter)",
        "gold_call": "_oracle_packet_max_deflection(N, A, sigma, kx, n_iter)",
    })

    # Boundary case: the smallest legal number of cycles, a single projection
    # followed by a single normalisation.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 1"
        ),
        "call": "packet_max_deflection(N, A, sigma, kx, n_iter)",
        "gold_call": "_oracle_packet_max_deflection(N, A, sigma, kx, n_iter)",
    })

    # Edge case: zero envelope amplitude gives the uniform field B = x_hat, which
    # is already solenoidal and already of unit magnitude, so it is a fixed point
    # of the iteration and its maximum deflection is exactly 0.0 degrees.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 0.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 2"
        ),
        "call": "packet_max_deflection(N, A, sigma, kx, n_iter)",
        "gold_call": "_oracle_packet_max_deflection(N, A, sigma, kx, n_iter)",
    })

    # Normal case: a deep run, far enough into the iteration that the
    # transient of the first few cycles has passed and the sequence is in
    # the slowly descending regime the benchmark configuration sits in.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 60"
        ),
        "call": "packet_max_deflection(N, A, sigma, kx, n_iter)",
        "gold_call": "_oracle_packet_max_deflection(N, A, sigma, kx, n_iter)",
    })

    # Invalid input: zero cycles, below the minimum of one.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 0\n"
            "def run_model():\n"
            "    try:\n"
            "        packet_max_deflection(N, A, sigma, kx, n_iter)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_packet_max_deflection(N, A, sigma, kx, n_iter)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })

    # Invalid input: a non-integer number of cycles.
    cases.append({
        "setup": (
            "import numpy as np\n"
            "N = 8\n"
            "A = 20.0\n"
            "sigma = 1.0 / 30.0\n"
            "kx = 4\n"
            "n_iter = 2.5\n"
            "def run_model():\n"
            "    try:\n"
            "        packet_max_deflection(N, A, sigma, kx, n_iter)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_packet_max_deflection(N, A, sigma, kx, n_iter)\n"
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
