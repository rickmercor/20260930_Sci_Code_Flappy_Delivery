"""
Evaluate the two solid-angle integrals of the swept-error factor that both iteration operators are assembled from.

Once the between-iteration error is expanded in a single spatial Fourier mode, the kinetic sweep stops differentiating: advection together with the loss term of the scattering operator collapses to multiplication by a factor that depends on direction only through the projection of the direction vector onto the wave vector, scaled by the product of that species' relaxation Knudsen number with its group speed. Two integrals of that factor over the whole unit sphere are needed downstream. The plain one closes the energy balances, since the new energy density is a solid-angle integral of the swept error. The one weighted by the squared projection carries the higher-order closure term of the flux equations, because contracting the stress-like velocity moment twice with the wave vector leaves exactly that weight. Both integrals are real, the odd part of the integrand cancelling between opposite directions. The analysis requires them exactly rather than by numerical quadrature, and both must stay accurate to full double precision across the whole sweep, from arguments so small that the two carriers are effectively diffusive to arguments large enough that they are ballistic.

Returns
-------
np.ndarray of shape (2,), float: the plain and the squared-projection weighted solid-angle integrals of the swept-error factor.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_angular_moments(u: float) -> np.ndarray:
    """Evaluate the two solid-angle integrals of the swept-error factor.

    Parameters
    ----------
    u : float
        Product of the relevant relaxation Knudsen number and group speed,
        finite and non-negative. Zero is admissible.

    Returns
    -------
    moments : np.ndarray
        Shape (2,): the plain solid-angle integral of the swept-error factor
        over the full unit sphere, then the integral of the same factor
        weighted by the squared projection of the direction vector onto the
        unit wave vector. Both are real.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(2, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_angular_moments(u: float) -> np.ndarray:
    import numpy as np

    u = float(u)
    if not (np.isfinite(u) and u >= 0.0):
        raise ValueError("u must be finite and >= 0")
    four_pi = 4.0 * np.pi

    if u <= 1.0e-1:
        u2 = u * u
        s0 = (1.0 - u2 / 3.0 + u2**2 / 5.0 - u2**3 / 7.0
              + u2**4 / 9.0 - u2**5 / 11.0 + u2**6 / 13.0 - u2**7 / 15.0)
        s2 = (1.0 / 3.0 - u2 / 5.0 + u2**2 / 7.0 - u2**3 / 9.0
              + u2**4 / 11.0 - u2**5 / 13.0 + u2**6 / 15.0 - u2**7 / 17.0)
        return np.array([four_pi * s0, four_pi * s2], dtype=float)

    ratio = np.arctan(u) / u
    return np.array([four_pi * ratio, four_pi * (1.0 - ratio) / (u * u)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: both arguments at the benchmark's interior worst-case point.
        {"setup": "import numpy as np\nkn = 10.0 ** 0.5\n",
         "call": "np.concatenate([compute_angular_moments(kn), compute_angular_moments(kn / 2.0)])",
         "gold_call": "np.concatenate([_oracle_compute_angular_moments(kn), _oracle_compute_angular_moments(kn / 2.0)])"},
        # Normal: a transition-regime electron argument.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(2.5)",
         "gold_call": "_oracle_compute_angular_moments(2.5)"},
        # Normal: the smaller phonon argument that the reduced scale produces.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(1.25)",
         "gold_call": "_oracle_compute_angular_moments(1.25)"},
        # Edge: strongly ballistic, both integrals far from their isotropic values.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(1000.0)",
         "gold_call": "_oracle_compute_angular_moments(1000.0)"},
        # Edge: diffusive end of the sweep, where full precision is hardest to keep.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(1e-8)",
         "gold_call": "_oracle_compute_angular_moments(1e-8)"},
        # Boundary: the former branch point, where the direct J2 formula loses precision.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(1e-4)",
         "gold_call": "_oracle_compute_angular_moments(1e-4)"},
        # Boundary: vanishing argument.
        {"setup": "import numpy as np\n",
         "call": "compute_angular_moments(0.0)",
         "gold_call": "_oracle_compute_angular_moments(0.0)"},
    ]
