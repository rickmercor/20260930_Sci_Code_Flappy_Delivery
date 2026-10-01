"""
Standardise the phenotype and return the two variance components the moment

system assigns to the region and to the residual.

For y = g + e with g ~ N(0, sigma_g^2 R) and e ~ N(0, sigma_e^2 I_n), a

moment-based estimator equates the observed quadratic forms y^T R y and y^T y to

their expectations under that model, giving two equations in the two components.

The phenotype is standardised to zero mean and y^T y = n, Tr R is known exactly

from the construction of R, and only Tr R^2 comes from the stochastic estimator.

Both components are carried forward: together they define the phenotypic

covariance that the prediction step has to solve against.

Returns
-------
tuple (float, float), the genetic and residual variance components, both as native Python floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rhe_variance_component(
    grm: "np.ndarray",
    phenotype: "np.ndarray",
    trace_r_squared: float,
) -> tuple:
    """Fit the moment system for one region and phenotype.

    Args:
        grm: array of shape (n, n), the relatedness matrix.
        phenotype: array of shape (n,) holding the raw phenotype values, which
            are standardised here to zero mean and y^T y = n.
        trace_r_squared: the estimate of Tr R^2.

    Returns:
        tuple (sigma_g_squared, sigma_e_squared) of native floats, the genetic
        and residual variance components of the fitted model.

    Raises:
        ValueError: if phenotype length does not match the relatedness matrix,
            if the phenotype has zero variance, or if the two moment conditions
            are not independent, which leaves the estimate unidentified.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rhe_variance_component(
    grm: "np.ndarray",
    phenotype: "np.ndarray",
    trace_r_squared: float,
) -> tuple:
    """Reference implementation."""
    grm = np.asarray(grm, dtype=float)
    trait = np.asarray(phenotype, dtype=float).ravel()
    n_individuals = grm.shape[0]
    if trait.size != n_individuals:
        raise ValueError("phenotype length does not match the relatedness matrix")
    centred = trait - trait.mean()
    scale = float(centred @ centred)
    if scale <= 0.0:
        raise ValueError("phenotype has zero variance")
    trace_r = float(n_individuals)
    if abs(float(trace_r_squared) - trace_r) <= 0.0:
        raise ValueError("the moment conditions are not independent")
    standardised = centred * np.sqrt(n_individuals / scale)
    quadratic = float(standardised @ (grm @ standardised))
    sigma_g_squared = (quadratic - trace_r) / (float(trace_r_squared) - trace_r)
    sigma_e_squared = (float(standardised @ standardised)
                       - sigma_g_squared * trace_r) / n_individuals
    return float(sigma_g_squared), float(sigma_e_squared)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
grm = np.array([[1.4, 0.2, -0.3, 0.1, 0.0], [0.2, 1.1, 0.0, -0.2, 0.1], [-0.3, 0.0, 0.9, 0.2, -0.1], [0.1, -0.2, 0.2, 1.0, 0.3], [0.0, 0.1, -0.1, 0.3, 0.6]])
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def pack(r):
    return [round(float(r[0]), 10), round(float(r[1]), 10)]
""",
            "call": "pack(rhe_variance_component(grm, trait, 6.0))",
            "gold_call": "pack(_oracle_rhe_variance_component(grm, trait, 6.0))",
        },
        {
            "setup": """import numpy as np
grm = np.array([[1.4, 0.2, -0.3, 0.1, 0.0], [0.2, 1.1, 0.0, -0.2, 0.1], [-0.3, 0.0, 0.9, 0.2, -0.1], [0.1, -0.2, 0.2, 1.0, 0.3], [0.0, 0.1, -0.1, 0.3, 0.6]])
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def pack(r):
    return [round(float(r[0]), 10), round(float(r[1]), 10)]
""",
            "call": "pack(rhe_variance_component(grm, trait, 4.25))",
            "gold_call": "pack(_oracle_rhe_variance_component(grm, trait, 4.25))",
        },
        {
            "setup": """import numpy as np
grm = np.array([[1.4, 0.2, -0.3, 0.1, 0.0], [0.2, 1.1, 0.0, -0.2, 0.1], [-0.3, 0.0, 0.9, 0.2, -0.1], [0.1, -0.2, 0.2, 1.0, 0.3], [0.0, 0.1, -0.1, 0.3, 0.6]])
trait = np.array([10.4, 12.1, 9.7, 11.5, 13.2])

def run_model():
    try:
        rhe_variance_component(grm, np.array([7.0, 7.0, 7.0, 7.0, 7.0]), 6.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_rhe_variance_component(grm, np.array([7.0, 7.0, 7.0, 7.0, 7.0]), 6.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
