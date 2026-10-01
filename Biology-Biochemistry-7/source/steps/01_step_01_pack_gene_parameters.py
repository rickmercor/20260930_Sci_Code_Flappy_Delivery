"""
Validate and pack the delayed auto-inhibition parameters.

The model combines five kinetic rates, a delay shape, two resource terms, and two auto-inhibition Hill parameters in one fixed numerical convention.

Returns
-------
np.ndarray: length-10 parameter vector [beta_m, beta_mstar, beta_p, gamma_m, gamma_p, alpha, M_a, v, h, P_a].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pack_gene_parameters(beta_m: float, beta_mstar: float, beta_p: float,
                         gamma_m: float, gamma_p: float, alpha: float,
                         m_a: float, resource_v: float, hill_h: float,
                         p_a: float) -> "np.ndarray":
    """Return parameters in the task's canonical order.

    Returns
    -------
    np.ndarray
        ``[beta_m,beta_mstar,beta_p,gamma_m,gamma_p,alpha,M_a,v,h,P_a]``.

    Raises
    ------
    ValueError
        If a parameter is non-finite, a rate/scale/exponent is non-positive,
        or ``resource_v`` is negative.
    """
    return parameters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_pack_gene_parameters(beta_m: float, beta_mstar: float, beta_p: float,
                                  gamma_m: float, gamma_p: float, alpha: float,
                                  m_a: float, resource_v: float, hill_h: float,
                                  p_a: float) -> "np.ndarray":
    values = np.asarray([beta_m, beta_mstar, beta_p, gamma_m, gamma_p,
                         alpha, m_a, resource_v, hill_h, p_a], dtype=float)
    if values.shape != (10,) or not np.all(np.isfinite(values)):
        raise ValueError("all parameters must be finite scalars")
    positive = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])
    if np.any(values[positive] <= 0.0):
        raise ValueError("rates, scales, and exponents must be positive")
    if values[7] < 0.0:
        raise ValueError("resource_v must be non-negative")
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\na=(10,.175,1,.08,.05,2.5,10,.5,1.5,5)",
            "call": "float(np.sum(pack_gene_parameters(*a)))",
            "gold_call": "float(np.sum(_oracle_pack_gene_parameters(*a)))",
        },
        {
            "setup": "import numpy as np\na=(1,2,3,4,5,1,2,.1,1,3)",
            "call": "float(np.prod(pack_gene_parameters(*a)))",
            "gold_call": "float(np.prod(_oracle_pack_gene_parameters(*a)))",
        },
        {
            "setup": "import numpy as np\na=(.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0)",
            "call": "float(np.dot(pack_gene_parameters(*a),np.arange(1,11)))",
            "gold_call": "float(np.dot(_oracle_pack_gene_parameters(*a),np.arange(1,11)))",
        },
        {"setup": "import numpy as np\ndef bad():\n    try: pack_gene_parameters(-1,.175,1,.08,.05,2.5,10,.5,1.5,5.); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_pack_gene_parameters(-1,.175,1,.08,.05,2.5,10,.5,1.5,5.); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
