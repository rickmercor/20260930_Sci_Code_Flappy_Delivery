"""
Build the direct electronic RPA blocks.

The Gram-factor interaction specifies the direct Coulomb channel in the finite transition basis. No additional exchange or spin multiplier is included.

Returns
-------
return electronic_a, electronic_b
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transition_blocks(gaps, factors):
    """Build the direct electronic RPA blocks.

    Parameters
    ----------
    gaps : real array, shape (n,)
        Positive electronic transition gaps.
    factors : real array, shape (n,r)
        Direct Coulomb factors; n and r are positive integers.
    
    Returns
    -------
    electronic_a, electronic_b : real arrays, shape (n,n)
        electronic_b=factors@factors.T; electronic_a=diag(gaps)+electronic_b.
        No extra spin factor is applied. Do not mutate either input.
        NumPy and SciPy are available. Import dependencies inside the function.
        Tests use absolute and relative tolerances of 2e-8.
    """
    return electronic_a, electronic_b

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transition_blocks(gaps, factors):
    import numpy as np
    gaps = np.asarray(gaps, dtype=float)
    factors = np.asarray(factors, dtype=float)
    interaction = factors @ factors.T
    return np.diag(gaps) + interaction, interaction

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': "transition_blocks(model['gaps'],model['factors'])",
      'gold_call': "_oracle_transition_blocks(model['gaps'],model['factors'])",
      'name': 'four_transitions',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'transition_blocks([1,2],[[1,0],[1,1]])',
      'gold_call': '_oracle_transition_blocks([1,2],[[1,0],[1,1]])',
      'name': 'integer_factors',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'transition_blocks([.3,.3,.8],np.zeros((3,2)))',
      'gold_call': '_oracle_transition_blocks([.3,.3,.8],np.zeros((3,2)))',
      'name': 'zero_interaction',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08},
     {'call': 'transition_blocks([.12],[[.4,-.7]])',
      'gold_call': '_oracle_transition_blocks([.12],[[.4,-.7]])',
      'name': 'one_transition',
      'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'gaps': [0.42, 0.71, 1.03, 1.28], 'factors': [[0.19, -0.06, 0.04], [0.08, 0.17, "
               "-0.05], [-0.11, 0.09, 0.16], [0.05, -0.12, 0.14]], 'dipoles': [[0.82, -0.37], [-0.46, "
               "0.71], [0.35, 0.58], [0.62, -0.29]], 'frequencies': [0.53, 0.88], 'couplings': [0.37, "
               "0.46], 'orders': [3, 3]}\n",
      'tol': 2e-08}]
