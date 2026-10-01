"""
Return the length density of age-1 female recruits at the given mesh midpoints: the normal probability density in metres with the given mean and standard deviation, evaluated as is, without renormalisation over the mesh.

New recruits enter the census with a narrow length distribution around the mean length at the end of the first growing season; this density, times the number of recruits, is the recruitment term of the projection.

Returns
-------
numpy.ndarray of float64, same shape as midpoints: recruit length density per metre.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recruit_length_density(midpoints: "numpy.ndarray", mean: float = 0.319,
                           sd: float = 0.040) -> "numpy.ndarray":
    """Return the length density of age-1 female recruits at the given mesh midpoints: the normal probability density in metres with the given mean and standard deviation, evaluated as is, without renormalisation over the mesh.

    Parameters
    ----------
    midpoints : numpy.ndarray
        One-dimensional array of finite bin midpoints in metres (row 0 of step 04).
    mean : float
        Mean recruit length in metres; default 0.319.
    sd : float
        Positive standard deviation of the recruit length in metres; default 0.040.

    Returns
    -------
    recruit_density : numpy.ndarray
        Array of the same shape as midpoints holding the density values per metre (float64).

    Raises
    ------
    ValueError
        If midpoints is not a non-empty one-dimensional finite array, mean is not finite, or sd is not positive and finite.
    """
    return recruit_density

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_recruit_length_density(midpoints: "numpy.ndarray", mean: float = 0.319,
                                   sd: float = 0.040) -> "numpy.ndarray":
    """Age-1 recruit length density C1(z) = N(z; mean, sd) evaluated at the midpoints."""
    z = np.asarray(midpoints, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("midpoints must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)):
        raise ValueError("midpoints must be finite")
    if not (np.isfinite(mean) and np.isfinite(sd)) or sd <= 0.0:
        raise ValueError("mean must be finite, sd positive and finite")
    return np.exp(-0.5 * ((z - float(mean)) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\n",
            "call": "recruit_length_density(z)",
            "gold_call": "_oracle_recruit_length_density(z)",
        },
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.01, 1.5, 40)\nz, w = mesh[0], mesh[1]\n",
            "call": "recruit_length_density(z)",
            "gold_call": "_oracle_recruit_length_density(z)",
        },
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nmean, sd = 0.25, 0.03\n",
            "call": "recruit_length_density(z, mean, sd)",
            "gold_call": "_oracle_recruit_length_density(z, mean, sd)",
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nmean, sd = 0.319, 0.0\ndef run_model():\n    try:\n        recruit_length_density(z, mean, sd)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_recruit_length_density(z, mean, sd)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
