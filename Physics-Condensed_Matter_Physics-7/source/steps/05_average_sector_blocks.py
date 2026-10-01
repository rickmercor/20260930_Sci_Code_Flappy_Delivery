"""
Average equal-weight signed projected-matrix samples within every independent block and take the symmetric part of each block mean while preserving both symmetry sectors and both matrix channels.

The two sectors can occupy different leading subspaces of a shared padded matrix dimension. Zero padding is retained, while the density and shifted-energy channels are averaged independently.

Returns
-------
np.ndarray, a float array of shape (2, 2, blocks, d, d)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def average_sector_blocks(sample_terms: np.ndarray) -> np.ndarray:
    """Average and symmetrize two-sector matrix samples by block.

    Parameters
    ----------
    sample_terms : np.ndarray
        Finite float array of shape ``(2, 2, blocks, samples, d, d)``. The
        first axis labels the two sectors and the second labels density then
        shifted energy. At least three blocks and one sample are required.

    Returns
    -------
    block_matrices : np.ndarray
        Float array of shape ``(2, 2, blocks, d, d)`` containing symmetric
        equal-weight block means.

    Raises
    ------
    ValueError
        If the shape is invalid or any entry is non-finite.
    """
    return block_matrices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_average_sector_blocks(sample_terms: np.ndarray) -> np.ndarray:
    import numpy as np

    terms = np.asarray(sample_terms, dtype=float)
    if (
        terms.ndim != 6
        or terms.shape[0] != 2
        or terms.shape[1] != 2
        or terms.shape[2] < 3
        or terms.shape[3] < 1
        or terms.shape[4] < 1
        or terms.shape[4] != terms.shape[5]
        or not np.all(np.isfinite(terms))
    ):
        raise ValueError(
            "sample_terms must have finite shape (2,2,blocks>=3,samples>=1,d,d)"
        )
    means = np.mean(terms, axis=3)
    return 0.5 * (means + np.swapaxes(means, -1, -2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx=np.arange(2*2*3*2*2*2,dtype=float).reshape(2,2,3,2,2,2)",
            "call": "average_sector_blocks(x.copy())",
            "gold_call": "_oracle_average_sector_blocks(x.copy())",
        },
        {
            "setup": "import numpy as np\nx=np.ones((2,2,4,1,1,1),dtype=float)",
            "call": "average_sector_blocks(x.copy())",
            "gold_call": "_oracle_average_sector_blocks(x.copy())",
        },
        {
            "setup": "import numpy as np\nx=np.zeros((2,2,3,5,3,3),dtype=float)\nx[1,0,0,0,0,2]=4.0",
            "call": "average_sector_blocks(x.copy())",
            "gold_call": "_oracle_average_sector_blocks(x.copy())",
        },
        {
            "setup": """import numpy as np
x=np.zeros((2,2,2,3,2,2),dtype=float)
def run_model():
 try:
  average_sector_blocks(x.copy())
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_average_sector_blocks(x.copy())
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
x=np.zeros((2,2,3,1,2,2),dtype=float); x[0,0,0,0,0,0]=np.nan
def run_model():
 try:
  average_sector_blocks(x.copy())
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_average_sector_blocks(x.copy())
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
