"""
Calculate the two principal thermal conductivities required by the omnidirectional counterpart of each local unidirectional design. Keep the conductivity along the design-state temperature-gradient direction equal to the physical-equivalence conductivity and calculate the orthogonal principal conductivity from the field-derived construction.

For each cell, the conductivity along the local design-state temperature-gradient direction remains

$$

\kappa_{\parallel,i}=\kappa_{P,i}.

$$

The conductivity in the orthogonal principal direction is

$$

\kappa_{\perp,i}=\frac{\kappa_0^2}{\kappa_{P,i}}.

$$

Therefore, the two principal conductivities of cell $i$ are

$$

\left(\kappa_{\parallel,i},\kappa_{\perp,i}\right)=\left(\kappa_{P,i},\frac{\kappa_0^2}{\kappa_{P,i}}\right).

$$

Their product satisfies

$$

\kappa_{\parallel,i}\kappa_{\perp,i}=\kappa_0^2,

$$

which provides a direct consistency check on the principal values.

Returns
-------
numerical NumPy array with shape (N, 2), containing the parallel and orthogonal principal thermal conductivities for each cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_principal_conductivities(
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "np.ndarray":
    """
    Calculate the principal conductivities of the omnidirectional cells.

    Parameters
    ----------
    kappa_0 : float
        Positive background thermal conductivity.
    kappa_p : np.ndarray
        Positive local physical-equivalence conductivities with shape (N,).

    Returns
    -------
    principal_conductivities : np.ndarray
        Principal conductivity pairs with shape (N, 2), where column 0
        contains the conductivity parallel to the local design-state
        gradient and column 1 contains the orthogonal conductivity.

    Raises
    ------
    ValueError
        If `kappa_0` is non-finite or non-positive, or if `kappa_p` contains
        any non-finite or non-positive value.
    """
    return principal_conductivities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_principal_conductivities(
    kappa_0: float,
    kappa_p: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the principal-conductivity calculation."""
    kappa_p = np.asarray(kappa_p, dtype=float)

    if not np.isscalar(kappa_0) or not np.isfinite(float(kappa_0)):
        raise ValueError("kappa_0 must be a finite scalar.")

    kappa_0 = float(kappa_0)

    if kappa_0 <= 0.0:
        raise ValueError("kappa_0 must be positive.")

    if kappa_p.ndim != 1 or kappa_p.size < 1:
        raise ValueError("kappa_p must have shape (N,) with N >= 1.")

    if not np.all(np.isfinite(kappa_p)):
        raise ValueError("kappa_p must contain only finite values.")

    if np.any(kappa_p <= 0.0):
        raise ValueError("kappa_p must be positive.")

    principal_conductivities = np.empty((kappa_p.size, 2), dtype=float)

    principal_conductivities[:, 0] = kappa_p
    principal_conductivities[:, 1] = (kappa_0**2) / kappa_p

    return principal_conductivities

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                "import numpy as np\n"
                "kappa_0 = 1.0\n"
                "kappa_p = np.array([0.22, 1.85, 4.60], dtype=float)\n"
            ),
            "call": (
                "compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
            "gold_call": (
                "_oracle_compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "kappa_0 = 2.0\n"
                "kappa_p = np.array([2.0, 2.0], dtype=float)\n"
            ),
            "call": (
                "compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
            "gold_call": (
                "_oracle_compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "kappa_0 = 1.0\n"
                "kappa_p = np.array([0.05, 20.0], dtype=float)\n"
            ),
            "call": (
                "compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
            "gold_call": (
                "_oracle_compute_principal_conductivities(kappa_0, kappa_p.copy())"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "kappa_0 = 1.0\n"
                "kappa_p = np.array([1.0, 0.0], dtype=float)\n"
                "\n"
                "def run_model():\n"
                "    try:\n"
                "        compute_principal_conductivities(kappa_0, kappa_p.copy())\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_compute_principal_conductivities(kappa_0, kappa_p.copy())\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
