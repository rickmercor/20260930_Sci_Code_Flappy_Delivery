"""
Pull back potential and acceleration sensitivities through the frozen FMM.

At fixed geometry the evaluated locals depend on the particle positions
and masses alone. This step returns the vector-Jacobian product of that
dependence for a supplied cotangent on the locals.

Returns
-------
np.ndarray, shape (N, 4)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pullback_fixed_fmm(x: "np.ndarray", masses: "np.ndarray", local_bar: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    """Compute the VJP of monopole-particle locals with fixed hierarchy geometry.

    Parameters
    ----------
    x : np.ndarray
        Finite particle positions, shape (N,3), N >= 1.
    masses : np.ndarray
        Finite source masses, shape (N,), signed values supported.
    local_bar : np.ndarray
        Cotangents of normalized locals, shape (N,M). Only entries of total
        degree 0 or 1 may be nonzero.
    tree : dict
        Fixed geometry: leaf_ids is an integer (N,) array; parent_ids is an
        integer (B,) array assigning each of B leaves to a parent; leaf_centers
        and parent_centers have shapes (B,3) and (A,3). IDs are in range.
        All arrays are finite; empty leaves are allowed. A >= 1, B >= 1.
        Different parents interact at parent level; particles in the same
        parent interact directly, excluding identical particle indices.
    epsilon : float
        Positive softening.
    p : int
        Total interaction order, 2 <= p <= 6.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Shape (N,4); columns 0:3 are position cotangents and column 3 is
        mass cotangent for the contraction sum(local_bar * locals).
        Both source and destination position dependence is included.
        Centers, topology, softening and G are held fixed.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pullback_fixed_fmm(x: "np.ndarray", masses: "np.ndarray", local_bar: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    indices = _indices(p)
    lookup = {n:i for i,n in enumerate(indices)}
    q = np.zeros((len(x),len(indices)))
    q[:,0] = masses
    forward = _oracle_evaluate_fixed_fmm(x,q,tree,epsilon,p,G)
    reverse = _oracle_evaluate_fixed_fmm(x,local_bar,tree,epsilon,p,G)
    result = np.zeros((len(x),4))
    result[:,3] = reverse[:,0]
    for axis in range(3):
        e = tuple(int(a==axis) for a in range(3))
        result[:,axis] = np.asarray(masses)*reverse[:,lookup[e]]
        for j,n in enumerate(indices):
            raised = tuple(n[a]+e[a] for a in range(3))
            if sum(raised)<=p:
                result[:,axis] += (n[axis]+1)*local_bar[:,j]*forward[:,lookup[raised]]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[1.05,-.08,.04],[1.2,.16,-.06]])\ntree=dict(leaf_ids=np.arange(4),parent_ids=np.array([0,0,1,1]),\n          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),\n          parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((4,20)); q[:,0]=[.8,1.2,.9,1.1]\nb=np.zeros_like(q); b[:,:4]=np.arange(16).reshape(4,4)/19-.3\n', 'call': 'pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .18, 3, .7)', 'gold_call': '_oracle_pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .18, 3, .7)', 'tol': 1e-10}, {'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[1.05,-.08,.04],[1.2,.16,-.06]])\ntree=dict(leaf_ids=np.arange(4),parent_ids=np.array([0,0,1,1]),\n          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),\n          parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((4,20)); q[:,0]=[.8,1.2,.9,1.1]\nb=np.zeros_like(q)\n', 'call': 'pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .18, 3, .7)', 'gold_call': '_oracle_pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .18, 3, .7)', 'tol': 1e-10}, {'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[1.05,-.08,.04],[1.2,.16,-.06]])\ntree=dict(leaf_ids=np.arange(4),parent_ids=np.array([0,0,1,1]),\n          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),\n          parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((4,20)); q[:,0]=[.8,1.2,.9,1.1]\nb=np.zeros_like(q); b[:,0]=[.2,-.1,.4,.3]\nx[1]=x[0]\n', 'call': 'pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .2, 3, .7)', 'gold_call': '_oracle_pullback_fixed_fmm(x.copy(), q[:,0].copy(), b.copy(), copy.deepcopy(tree), .2, 3, .7)', 'tol': 1e-10}, {'setup': 'import numpy as np\nimport copy\n'
          'x=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[-1.02,.05,.08],[-1.21,-.03,-.07]])\n'
          'tree=dict(leaf_ids=np.array([0,0,1,1]),parent_ids=np.array([0,0]),\n'
          '          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0]]),\n'
          '          parent_centers=np.array([[-1.1,.05,0]]))\n'
          'm=np.array([.8,-1.2,.9,1.1])\n'
          'b=np.zeros((4,35)); b[:,0]=[.2,-.1,.4,.3]; b[:,1:4]=np.arange(12).reshape(4,3)/23-.25\n',
  'call': 'pullback_fixed_fmm(x.copy(), m.copy(), b.copy(), copy.deepcopy(tree), .18, 4, .7)',
  'gold_call': '_oracle_pullback_fixed_fmm(x.copy(), m.copy(), b.copy(), copy.deepcopy(tree), .18, 4, .7)',
  'tol': 1e-10},
 {'setup': 'import numpy as np\nimport copy\n'
          'x=np.array([[-1.15,-.1,.05],[1.05,-.08,.04],[.1,.9,-.3],[-.2,-1.1,.25]])\n'
          'tree=dict(leaf_ids=np.arange(4),parent_ids=np.arange(4),\n'
          '          leaf_centers=np.array([[-1.1,-.1,0],[1.1,-.1,0],[.1,.9,0],[-.2,-1.1,0]]),\n'
          '          parent_centers=np.array([[-1.1,-.1,0],[1.1,-.1,0],[.1,.9,0],[-.2,-1.1,0]]))\n'
          'm=np.array([.8,1.2,-.9,1.1])\n'
          'b=np.zeros((4,35)); b[:,1:4]=np.arange(12).reshape(4,3)/17-.3\n',
  'call': 'pullback_fixed_fmm(x.copy(), m.copy(), b.copy(), copy.deepcopy(tree), .22, 4, .9)',
  'gold_call': '_oracle_pullback_fixed_fmm(x.copy(), m.copy(), b.copy(), copy.deepcopy(tree), .22, 4, .9)',
  'tol': 1e-10}]
