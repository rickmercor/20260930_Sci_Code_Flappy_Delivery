"""
Obtain the full particle-addition density and its response.

The interacting particle-addition SDM and its tangent describe the change between adjacent sectors. The site basis and quenched Hamiltonian agree between sectors; all complex site coherences are retained.

Returns
-------
return delta
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def addition_density(pair: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Parameters
    ----------
    pair : array_like, shape (2,2,P,P)
        Finite real or complex matrices, P>=1; first axis labels N and N-1, second labels density and derivative. The algebraic operation is defined for all finite pairs of this shape.
    
    Returns
    -------
    delta : ndarray, shape (2,P,P)
        Delta and its interaction tangent Delta' under the interacting particle-addition SDM prescription, in the full complex site basis. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_addition_density(pair: 'np.ndarray | list | tuple') -> 'np.ndarray':
    a=np.asarray(pair)
    if a.ndim!=4 or a.shape[:2]!=(2,2) or a.shape[-1]!=a.shape[-2] or a.shape[-1]<1 or not np.isfinite(a).all():
        raise ValueError('finite (2,2,P,P) sector jets required')
    return a[0]-a[1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# single-site number addition\npair=np.array([[[1.]],[[0.]]])\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# rank-one coherent Slater addition\nu=np.array([1.,-2.,3.]);u/=np.linalg.norm(u);pair=np.stack((np.outer(u,u),np.zeros((3,3))))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# diagonal interaction-induced depletion\npair=np.stack((np.diag([.6,.6,.8]),np.diag([.1,.1,.8])))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# indefinite addition matrix\npair=np.stack((np.diag([.1,.9,1.]),np.diag([.8,.1,.1])))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# off-diagonal rearrangement sign\npair=np.array([[[.7,-.2,0.],[-.2,.6,.1],[0.,.1,.7]],[[.4,.15,0.],[.15,.3,-.1],[0.,-.1,.3]]])\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# complex Hermitian phase coherence\nu=np.array([1.,1.j,-1.])/np.sqrt(3);pair=np.stack((np.eye(3)-np.outer(u,u.conj()),np.diag([0.,0.,1.])))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# long-distance intersite coherence\nu=np.array([1.,0.,0.,-1.])/np.sqrt(2);v=np.array([0.,1.,1.,0.])/np.sqrt(2);pair=np.stack((np.outer(u,u)+np.outer(v,v),np.outer(v,v)))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}, {'setup': 'import numpy as np\n# filled-sector hole addition\nu=np.array([1.,2.,-1.,-2.])/np.sqrt(10);pair=np.stack((np.eye(4),np.eye(4)-np.outer(u,u)))\nargs=(pair,)\nargs=tuple(np.asarray(value) if i in [0] else value for i,value in enumerate(args))\nfrom copy import deepcopy\np=args[0];q=np.arange(p.size).reshape(p.shape)/max(1,p.size);args=(np.stack((p,q),axis=1),)', 'call': 'addition_density(*deepcopy(args))', 'gold_call': '_oracle_addition_density(*deepcopy(args))'}]
