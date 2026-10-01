"""
Map microscopic rate constants to the macroscopic kinetic observables used in the benchmark: kcat, KM, catalytic efficiency kcat/KM, and KD. The mapping must depend on whether the supplied rates describe the simple or extended catalytic cycle.

Macroscopic enzyme kinetic parameters are emergent quantities obtained from the microscopic transitions of the catalytic cycle. In the simple mechanism, kcat is directly determined by the chemical transition, whereas KM and catalytic efficiency combine binding and catalytic rates. In the extended mechanism, product-state formation and reversible conversion introduce additional pathways, making kcat and KM nonlinear functions of multiple microscopic rates. In contrast, KD retains the same microscopic binding-rate relationship when the catalytic mechanism is extended. These distinctions are central to the paper because non-specific epistasis can appear when additive microscopic perturbations are passed through nonlinear kinetic mappings.

Returns
-------
np.ndarray, a two-dimensional floating-point array of shape (n, 4), containing [kcat, KM, kcat_over_KM, KD] for each supplied microscopic-rate configuration.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_kinetic_observables(
    rate_states: "np.ndarray",
    complex_model: bool,
) -> "np.ndarray":
    """
    Compute the source-defined kinetic observables from microscopic
    rate constants for the selected representation.

    Parameters
    ----------
    rate_states : np.ndarray
        Microscopic rate constants in the state ordering defined by the task.

    complex_model : bool
        Selects the reduced or complete kinetic representation.

    Returns
    -------
    np.ndarray
        A two-dimensional floating-point array with four observable
        columns in the ordering required by the downstream calculations.

    Raises
    ------
    ValueError
        If the rate array has an invalid shape, contains non-finite or
        non-positive values, or the representation selector is invalid.
    """
    return observables

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_kinetic_observables(
    rate_states: "np.ndarray",
    complex_model: bool,
) -> "np.ndarray":
    r = np.asarray(rate_states, dtype=float)
    nr = 5 if complex_model else 3

    if (
        r.ndim != 2
        or r.shape[1] != nr
        or r.shape[0] < 1
        or not np.all(np.isfinite(r))
        or np.any(r <= 0)
    ):
        raise ValueError("invalid rate states")

    if complex_model:
        k1, km1, k2, km2, k3 = r.T
        D = k2 + km2 + k3
        kcat = k2 * k3 / D
        km = (
            k2 * k3
            + km1 * km2
            + km1 * k3
        ) / (k1 * D)
    else:
        k1, km1, k2 = r.T
        kcat = k2
        km = (km1 + k2) / k1

    eta = kcat / km
    kd = km1 / k1

    out = np.column_stack((kcat, km, eta, kd))

    if not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("invalid observables")

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nr=np.array([[2.,1.,3.],[4.,2.,5.]],float)\n",
            "call": "compute_kinetic_observables(r,False)",
            "gold_call": "_oracle_compute_kinetic_observables(r,False)",
        },
        {
            "setup": "import numpy as np\nr=np.array([[2.,1.,3.,4.,5.]],float)\n",
            "call": "compute_kinetic_observables(r,True)",
            "gold_call": "_oracle_compute_kinetic_observables(r,True)",
        },
        {
            "setup": "import numpy as np\nr=np.ones((1,3),float)\n",
            "call": "compute_kinetic_observables(r,False)",
            "gold_call": "_oracle_compute_kinetic_observables(r,False)",
        },
        {
            "setup": "import numpy as np\nr=np.ones((1,4),float)\ndef run_model():\n    try: compute_kinetic_observables(r,False); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_compute_kinetic_observables(r,False); return 0\n    except ValueError: return 1\n    except Exception: return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
