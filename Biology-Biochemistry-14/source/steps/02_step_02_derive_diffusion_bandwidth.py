"""
Fix the width of the Gaussian diffusion kernel from the neighbourhood radius and the fraction of released ligand allowed to escape it.

A secreted ligand released at a point is modelled as spreading into an isotropic Gaussian in the tissue plane, and the kernel is made as wide as possible while keeping all but a negligible fraction of the release inside the signalling neighbourhood.

Returns
-------
float: the Gaussian bandwidth sigma.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def derive_diffusion_bandwidth(radius: float, tail_mass: float) -> float:
    """Return the largest Gaussian bandwidth that keeps a release inside a radius.

    A unit amount of ligand released at a point of the tissue plane is spread
    as an isotropic two-dimensional Gaussian with standard deviation
    ``sigma`` along each axis. Return the largest ``sigma`` for which the
    fraction of that amount lying at distance ``radius`` or more from the
    release point does not exceed ``tail_mass``.

    Parameters
    ----------
    radius : float
        Neighbourhood radius, in the same length unit as ``sigma``.
    tail_mass : float
        Largest admissible fraction of the release beyond ``radius``.

    Returns
    -------
    float
        The bandwidth ``sigma``.

    Raises
    ------
    ValueError
        If ``radius`` is not a finite positive number, or if ``tail_mass``
        is not a finite number strictly between 0 and 1 (booleans are
        rejected for both).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_derive_diffusion_bandwidth(radius: float, tail_mass: float) -> float:
    """Reference implementation (closed-form tail mass of a planar Gaussian)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(radius) and radius > 0.0):
        raise ValueError("radius must be a finite positive number")
    if not (_is_number(tail_mass) and 0.0 < tail_mass < 1.0):
        raise ValueError("tail_mass must be a finite number in (0, 1)")
    # In polar coordinates the mass of a planar Gaussian beyond distance t is
    # exp(-t^2 / (2 sigma^2)), which increases with sigma; setting it equal
    # to the admissible tail gives the widest kernel.
    sigma = float(radius) / np.sqrt(-2.0 * np.log(float(tail_mass)))
    return float(sigma)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "derive_diffusion_bandwidth(150.0, 1e-6) / 10.0",
            "gold_call": "_oracle_derive_diffusion_bandwidth(150.0, 1e-6) / 10.0",
        },
        {
            "setup": "import numpy as np\n",
            "call": "derive_diffusion_bandwidth(1.0, 0.5)",
            "gold_call": "_oracle_derive_diffusion_bandwidth(1.0, 0.5)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "derive_diffusion_bandwidth(2.5, 0.01)",
            "gold_call": "_oracle_derive_diffusion_bandwidth(2.5, 0.01)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "derive_diffusion_bandwidth(40.0, 1e-12) / 10.0",
            "gold_call": "_oracle_derive_diffusion_bandwidth(40.0, 1e-12) / 10.0",
        },
        {
            "setup": status,
            "call": "_status(lambda: derive_diffusion_bandwidth(0.0, 0.1))",
            "gold_call": "_status(lambda: _oracle_derive_diffusion_bandwidth(0.0, 0.1))",
        },
        {
            "setup": status,
            "call": "_status(lambda: derive_diffusion_bandwidth(10.0, 1.0))",
            "gold_call": "_status(lambda: _oracle_derive_diffusion_bandwidth(10.0, 1.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: derive_diffusion_bandwidth(float('nan'), 0.1))",
            "gold_call": "_status(lambda: _oracle_derive_diffusion_bandwidth(float('nan'), 0.1))",
        },
    ]
