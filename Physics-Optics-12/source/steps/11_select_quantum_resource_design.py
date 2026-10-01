"""
Filter column 11 for feasible rows, then select the lexicographic minimum by (CX column 9, max error column 7, negative contrast column 8, candidate index, N_position, group_position, stable row index). Return [row_index,candidate_index,N,group,slices,nq,worst_error_index,min_fidelity,max_error,min_contrast,CX]. Raise ValueError if no row is feasible.

The selector makes the paper's exponential diagonal-synthesis cost load-bearing after physical convergence is established.

Returns
-------
np.ndarray, (11,) real selected-design certificate [row_index,candidate,N,g,slices,nq,error_index,Fmin,Eref,Cmin,CX].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_quantum_resource_design(design_table: 'np.ndarray') -> 'np.ndarray':
    """Select one feasible row using the exact lexicographic resource key.

Parameters
----------
design_table : array_like
    Finite nonempty float array-like (R,14) with the documented design-table columns.

Returns
-------
result : np.ndarray
    (11,) real selected-design certificate [row_index,candidate,N,g,slices,nq,error_index,Fmin,Eref,Cmin,CX].

Notes
-----
Select one feasible row using the exact lexicographic resource key.

design_table has shape (R,14). Returns a float ndarray of shape (11,)
ordered as [row_index,candidate,N,g,slices,nq,error_index,Fmin,Eref,
Cmin,CX]; stable input order resolves an exact tie.
Raises ValueError if the table is empty, nonfinite, has a shape other than
(R,14), or contains no feasible row."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_select_quantum_resource_design(design_table: 'np.ndarray') -> 'np.ndarray':
    """Return the selected row's decisive numeric certificate."""
    table = np.asarray(design_table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != 14:
        raise ValueError('design_table must have shape (R,14)')
    if not np.all(np.isfinite(table)):
        raise ValueError('design_table must be finite')
    feasible = np.where(table[:, 11] == 1.0)[0]
    if feasible.size == 0:
        raise ValueError('no quantum design satisfies every constraint')
    selected = min(feasible.tolist(), key=lambda i: (table[i, 9], table[i, 7], -table[i, 8], table[i, 0], table[i, 12], table[i, 13], i))
    row = table[selected]
    return np.asarray([float(selected), row[0], row[1], row[2], row[3], row[4], row[10], row[6], row[7], row[8], row[9]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np; '
               't=np.array([[0,16,1,6,8,1,1,.01,.13,3000,2,1,0,0],[1,32,3,2,10,1,1,.019,.125,4200,3,1,1,2]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[2,32,3,2,10,1,1,.018,.121,4200,4,1,1,2],[1,32,3,2,10,1,1,.019,.130,4200,3,1,1,2]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[2,32,3,2,10,1,1,.018,.121,4200,4,1,1,2],[1,32,3,2,10,1,1,.018,.130,4200,3,1,1,2]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[1,32,2,3,10,1,1,.01,.13,5000,9,1,1,1],[1,32,2,3,10,1,1,.01,.13,5000,8,1,1,1]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[3,32,2,3,10,1,1,.005,.13,9000,7,1,0,0],[2,32,2,3,10,1,1,.004,.12,9000,6,1,0,0]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[1,16,1,6,8,1,1,.01,.13,1000,5,0,0,0],[4,32,1,6,10,1,1,.01,.13,2000,8,1,1,0]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'},
     {'setup': 'import numpy as np; '
               't=np.array([[2,32,1,6,10,1,1,.01,.13,5000,1,1,0,0],[1,32,1,6,10,1,1,.01,.13,5000,1,1,0,0]],float)\n'
               'import copy\n'
               't_gold = copy.deepcopy(t)',
      'call': 'select_quantum_resource_design(t)',
      'gold_call': '_oracle_select_quantum_resource_design(t_gold)'}]
