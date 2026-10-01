"""
Construct the selected-bond ladder Hamiltonian and its interaction tangent.

Appendix D, Eq. (D1), pairs identical on-site disorder and repulsion on selected nearest-neighbor bonds, while hopping covers every nearest-neighbor lattice edge. The model is H=sum_pq h_pq c_p^dagger c_q+sum_(p<q) U_pq n_p n_q. The tangent differentiates the selected-bond interaction at fixed disorder and hopping.

Returns
-------
return model
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bond_ladder_model(potentials: 'np.ndarray | list | tuple', length: int, width: int, bonds: 'np.ndarray | list | tuple', hopping: float, interaction: float) -> 'np.ndarray':
    """Parameters
    ----------
    potentials : array_like, shape (P/2,)
        Finite real bond potentials, assigned in the supplied bond-row order.
    length, width : int
        Positive open rectangular dimensions, P=length*width even and at most 18.
    bonds : array_like, shape (P/2,2)
        Integer nearest-neighbor pairs partitioning sites p=width*x+a. Row and endpoint orders are arbitrary.
    hopping : float
        Finite positive hopping magnitude.
    interaction : float
        Finite selected-bond repulsion, including negative values.
    
    Returns
    -------
    model : ndarray, shape (2,2,P,P)
        First axis is value and interaction derivative; second is h and U. At value order, every nearest-neighbor off-diagonal h entry is -hopping and each bond shares its potential on h's diagonal. U equals interaction on selected bonds symmetrically, with zero diagonal. Derivative h is zero and derivative U is one on selected bonds. Other entries are zero. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bond_ladder_model(potentials: 'np.ndarray | list | tuple', length: int, width: int, bonds: 'np.ndarray | list | tuple', hopping: float, interaction: float) -> 'np.ndarray':
    e=np.asarray(potentials,dtype=float); links=np.asarray(bonds)
    if not isinstance(length,(int,np.integer)) or not isinstance(width,(int,np.integer)) or length<1 or width<1 or length*width>18:
        raise ValueError('positive integer dimensions with at most 18 sites required')
    P=length*width
    if P%2 or links.shape!=(P//2,2) or e.shape!=(P//2,) or not np.isfinite(e).all() or not np.isfinite(links).all() or np.any(links!=np.floor(links)):
        raise ValueError('one finite potential per matched bond required')
    links=links.astype(int)
    if sorted(links.ravel().tolist())!=list(range(P)):
        raise ValueError('bonds must partition the sites')
    for i,j in links:
        if abs(i//width-j//width)+abs(i%width-j%width)!=1:
            raise ValueError('selected bonds must be nearest neighbors')
    if not np.isfinite(hopping) or hopping<=0 or not np.isfinite(interaction):
        raise ValueError('positive hopping and finite interaction required')
    model=np.zeros((2,P,P))
    for i in range(P):
        for j in range(i+1,P):
            if abs(i//width-j//width)+abs(i%width-j%width)==1:
                model[0,i,j]=model[0,j,i]=-hopping
    for energy,(i,j) in zip(e,links):
        model[0,i,i]=model[0,j,j]=energy
        model[1,i,j]=model[1,j,i]=interaction
    derivative=np.zeros_like(model)
    for i,j in links:derivative[1,i,j]=derivative[1,j,i]=1.
    return np.stack((model,derivative))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# transverse matched dimer\nargs=([0.3], 1, 2, [[0, 1]], 1.0, 2.0)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# longitudinal matched dimer\nargs=([-0.7], 2, 1, [[0, 1]], 0.8, -0.4)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# two transverse rungs\nargs=([-1.9, 1.9], 2, 2, [[0, 1], [2, 3]], 1.0, 2.5)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# two longitudinal bonds\nargs=([-1.9, 1.9], 2, 2, [[0, 2], [1, 3]], 1.0, 2.5)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# reordered selected bonds\nargs=([0.8, -0.3], 2, 2, [[3, 1], [2, 0]], 0.7, 1.4)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# three-site transverse cross-section\nargs=([1.0, 2.0, 3.0], 2, 3, [[0, 3], [1, 4], [2, 5]], 1.0, 1.0)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# all transverse matching\nargs=([-2.0, 0.0, 2.0], 3, 2, [[0, 1], [2, 3], [4, 5]], 1.0, 0.5)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# mixed matching with shared binary disorder\nargs=([-1.0, 1.0, 1.0, -1.0], 4, 2, [[0, 2], [1, 3], [4, 5], [6, 7]], 1.0, 2.5)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# clean selected-bond limit\nargs=([0.0, 0.0, 0.0, 0.0], 4, 2, [[0, 2], [1, 3], [4, 5], [6, 7]], 1.0, 0.0)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# electrostatic offset of paired dots\nargs=([10.0, 12.0, 9.0, 11.0], 4, 2, [[0, 2], [1, 3], [4, 5], [6, 7]], 2.0, 3.0)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# attractive selected bonds\nargs=([0.2, -0.1, 0.7, 0.5], 2, 4, [[0, 1], [2, 3], [4, 5], [6, 7]], 0.8, -1.0)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# square cross-section nearest-neighbor exclusion\nargs=([0.1, 0.4, 0.7, 0.2], 2, 4, [[0, 4], [1, 5], [2, 6], [3, 7]], 1.0, 0.9)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}, {'setup': 'import numpy as np\n# unpaired-site disorder is absent\nargs=([1.0, -1.0, -1.0, 1.0, 1.0, -1.0], 6, 2, [[0, 2], [1, 3], [4, 5], [6, 7], [8, 10], [9, 11]], 1.0, 2.5)\nargs=tuple(np.asarray(value) if i in [0, 3] else value for i,value in enumerate(args))\nfrom copy import deepcopy', 'call': 'bond_ladder_model(*deepcopy(args))', 'gold_call': '_oracle_bond_ladder_model(*deepcopy(args))'}]
