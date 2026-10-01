"""
Evaluate initiation, translation, and degradation rates.

Protein represses transcription initiation, whereas translation and both loss channels follow mass action. Delayed completions are handled separately.

Returns
-------
np.ndarray: length-4 propensity vector, one entry per non-delayed channel.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gene_channel_propensities(state: "np.ndarray", parameters: "np.ndarray") -> "np.ndarray":
    """Return propensities in initiation, translation, mRNA-loss, protein-loss order.

    Parameters
    ----------
    state : array_like
        Non-negative integer-like ``[M, P, Mstar]``.
    parameters : array_like
        Length-10 vector from step 01.

    Raises
    ------
    ValueError
        If either vector has the wrong shape or contains inadmissible values.
    """
    return propensities  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_gene_channel_propensities(state: "np.ndarray",
                                       parameters: "np.ndarray") -> "np.ndarray":
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if x.shape != (3,) or p.shape != (10,):
        raise ValueError("state and parameters must have lengths 3 and 10")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or np.any(x != np.floor(x)):
        raise ValueError("state must contain non-negative counts")
    if not np.all(np.isfinite(p)) or np.any(p[[0, 1, 2, 3, 4, 5, 6, 8, 9]] <= 0.0) or p[7] < 0.0:
        raise ValueError("parameters are inadmissible")
    m, protein = x[:2]
    initiation = p[0] * p[9] ** p[8] / (p[9] ** p[8] + protein ** p[8])
    return np.array([initiation, p[2] * m, p[3] * m, p[4] * protein], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nx=np.array([8,20,3]); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.dot(gene_channel_propensities(x,p),np.arange(1,5)))",
            "gold_call": "float(np.dot(_oracle_gene_channel_propensities(x,p),np.arange(1,5)))",
        },
        {
            "setup": "import numpy as np\nx=np.zeros(3); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.sum(gene_channel_propensities(x,p)))",
            "gold_call": "float(np.sum(_oracle_gene_channel_propensities(x,p)))",
        },
        {
            "setup": "import numpy as np\nx=np.array([1,100000,0]); p=np.array([2,.2,3,.1,.2,1,4,0,2,5.])",
            "call": "float(gene_channel_propensities(x,p)[0])",
            "gold_call": "float(_oracle_gene_channel_propensities(x,p)[0])",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\ndef bad():\n    try: gene_channel_propensities(np.zeros(2),p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_gene_channel_propensities(np.zeros(2),p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
