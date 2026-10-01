"""
Evaluate the two-level frozen Cartesian interaction operator.

The operator is fixed: assignments, centers and the near/far split are
supplied and never recomputed from the state. Sources are general
particle-centered multipoles, not only masses.

Returns
-------
np.ndarray, shape (N, M)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_fixed_fmm(x: "np.ndarray", moments: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    """Evaluate a prescribed fixed hierarchy on monopole or generalized sources.

    Parameters
    ----------
    x : np.ndarray
        Positions of N >= 1 particles, shape (N,3).
    moments : np.ndarray
        Particle-centered raw multipoles, shape (N,M), where
        M=(p+1)(p+2)(p+3)/6, in kernel_derivatives order. Signed moments
        are allowed. Ordinary particles have only the mass component.
    tree : dict
        Fixed geometry: leaf_ids is an integer (N,) array; parent_ids is an
        integer (B,) array assigning each of B leaves to a parent; leaf_centers
        and parent_centers have shapes (B,3) and (A,3). IDs are in range.
        All arrays are finite; empty leaves are allowed. A >= 1, B >= 1.
        Different parents interact at parent level; particles in the same
        parent interact directly, excluding identical particle indices.
    epsilon : float
        Positive softening length.
    p : int
        Total interaction order, 0 <= p <= 6.
    G : float
        Finite interaction strength, including zero.

    Returns
    -------
    result : np.ndarray
        Particle-centered normalized locals, shape (N,M). Inter-parent
        translations retain total source-plus-destination degree <= p.
        Within-parent pairs use the unexpanded softened kernel at the
        particle displacement, with the same |source|+|local| <= p rule
        for generalized moments. Self contributions are excluded.
        The fixed schedule applies even if particles move across centers.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_fixed_fmm(x: "np.ndarray", moments: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    x = np.asarray(x, dtype=float)
    moments = np.asarray(moments, dtype=float)
    leaf = np.asarray(tree['leaf_ids'])
    parent = np.asarray(tree['parent_ids'])
    lc = np.asarray(tree['leaf_centers'])
    pc = np.asarray(tree['parent_centers'])
    size = len(_indices(p))
    parent_q = np.zeros((len(pc),size))
    for b in range(len(lc)):
        ids = np.flatnonzero(leaf == b)
        q = _oracle_particle_moments(x[ids],moments[ids,0],lc[b],p)
        for i in ids:
            higher = moments[i].copy()
            higher[0] = 0.0
            q += _oracle_translate_moments(higher,x[i]-lc[b],p)
        parent_q[parent[b]] += _oracle_translate_moments(q,lc[b]-pc[parent[b]],p)
    parent_l = np.zeros_like(parent_q)
    for d in range(len(pc)):
        for s in range(len(pc)):
            if s != d:
                parent_l[d] += _oracle_multipole_to_local(parent_q[s],pc[s]-pc[d],epsilon,p,G)
    result = np.zeros_like(moments)
    for b in range(len(lc)):
        local = _oracle_translate_locals(parent_l[parent[b]],lc[b]-pc[parent[b]],p)
        for i in np.flatnonzero(leaf == b):
            result[i] = _oracle_translate_locals(local,x[i]-lc[b],p)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and parent[leaf[i]] == parent[leaf[j]]:
                result[i] += _oracle_multipole_to_local(moments[j],x[j]-x[i],epsilon,p,G)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[1.05,-.08,.04],[1.2,.16,-.06]])\ntree=dict(leaf_ids=np.arange(4),parent_ids=np.array([0,0,1,1]),\n          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),\n          parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((4,20)); q[:,0]=[.8,1.2,.9,1.1]\n', 'call': 'evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .18, 3, .7)', 'gold_call': '_oracle_evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .18, 3, .7)', 'tol': 1e-10}, {'setup': 'import numpy as np\nimport copy\nx=np.zeros((1,3))\nq=np.array([[1.]])\ntree=dict(leaf_ids=np.array([0]),parent_ids=np.array([0]),leaf_centers=np.zeros((1,3)),parent_centers=np.zeros((1,3)))', 'call': 'evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .2, 0, 1.)', 'gold_call': '_oracle_evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .2, 0, 1.)', 'tol': 1e-10}, {'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.15,-.1,.05],[-.95,.13,-.02],[1.05,-.08,.04],[1.2,.16,-.06]])\ntree=dict(leaf_ids=np.arange(4),parent_ids=np.array([0,0,1,1]),\n          leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),\n          parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((4,20)); q[:,0]=[.8,1.2,.9,1.1]\nq[:,0]=0\nq[:,1:4]=np.arange(12).reshape(4,3)/17-.2\nx[1]=x[0]\n', 'call': 'evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .25, 3, .9)', 'gold_call': '_oracle_evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .25, 3, .9)', 'tol': 1e-10},
            {'setup': 'import numpy as np\nimport copy\nx=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\ntree=dict(leaf_ids=np.array([0,0,1,1,2,2,3,3]),parent_ids=np.array([0,0,1,1]),leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nq=np.zeros((8,35)); q[:,0]=[.8,1.1,.9,1.2,1.05,.75,1.15,.95]\n', 'call': 'evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .18, 4, .7)', 'gold_call': '_oracle_evaluate_fixed_fmm(x.copy(), q.copy(), copy.deepcopy(tree), .18, 4, .7)', 'tol': 1e-10}]
