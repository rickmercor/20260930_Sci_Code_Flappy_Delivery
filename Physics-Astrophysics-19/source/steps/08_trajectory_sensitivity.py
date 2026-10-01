"""
Compute the terminal-loss directional derivative through a DKD trajectory.

The loss is evaluated on a short trajectory of the fixed interaction
operator and differentiated exactly with respect to the perturbation
parameter. The model is finite and small; no GPU execution is needed.

Returns
-------
float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def trajectory_sensitivity(config: dict) -> float:
    """Return the analytic derivative of the specified discrete terminal loss.

    Parameters
    ----------
    config : dict
        x, v: finite (N,3) initial positions and velocities, N >= 1.
        masses: positive (N,) masses; weights: nonnegative (N,) loss weights.
        target: fixed finite (N,3) terminal target independent of perturbation.
        dx, dv, dm: finite perturbation directions of shapes (N,3), (N,3), (N,).
        tree: fixed geometry dictionary with the evaluate_fixed_fmm contract.
        epsilon > 0, integer 2 <= p <= 6, finite G >= 0, finite h >= 0,
        integer steps >= 0, finite lam >= 0.
        For parameter s, initial x,v,masses are perturbed by s times dx,dv,dm.
        Advance the state with steps Drift-Kick-Drift updates of length h
        driven by the per-unit-mass field acceleration, reusing the fixed
        tree at every evaluation including the terminal potential.

    Returns
    -------
    result : float
        dJ/ds at s=0, where J is half the weighted sum of squared terminal
        position residuals plus lam/2 times sum_i masses_i phi_i at the
        terminal positions. Include explicit and force-mediated mass effects.
        Zero steps is the unevolved terminal-loss sensitivity.
        All calculations use the finite-order interaction model.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _trajectory_adjoint(config):
    x = np.array(config['x'],dtype=float,copy=True)
    v = np.array(config['v'],dtype=float,copy=True)
    m = np.asarray(config['masses'],dtype=float)
    tree,eps,p,G = (config[k] for k in ('tree','epsilon','p','G'))
    h = config['h']
    q = np.zeros((len(x),len(_indices(p))))
    q[:,0] = m
    midpoints = []
    for _ in range(config['steps']):
        midpoint = x + .5*h*v
        local = _oracle_evaluate_fixed_fmm(midpoint,q,tree,eps,p,G)
        v = v-h*local[:,1:4]
        x = midpoint+.5*h*v
        midpoints.append(midpoint)
    local = _oracle_evaluate_fixed_fmm(x,q,tree,eps,p,G)
    residual = x-np.asarray(config['target'])
    weights = np.asarray(config['weights'])
    lam = config['lam']
    loss = .5*np.sum(weights[:,None]*residual**2)+.5*lam*np.dot(m,local[:,0])
    b = np.zeros_like(q)
    b[:,0] = .5*lam*m
    terminal = _oracle_pullback_fixed_fmm(x,m,b,tree,eps,p,G)
    bx = weights[:,None]*residual+terminal[:,:3]
    bv = np.zeros_like(v)
    bm = .5*lam*local[:,0]+terminal[:,3]
    for midpoint in reversed(midpoints):
        bv_new = bv+.5*h*bx
        b = np.zeros_like(q)
        b[:,1:4] = -h*bv_new
        force_pullback = _oracle_pullback_fixed_fmm(midpoint,m,b,tree,eps,p,G)
        bx = bx+force_pullback[:,:3]
        bm = bm+force_pullback[:,3]
        bv = bv_new+.5*h*bx
    derivative = np.sum(bx*config['dx'])+np.sum(bv*config['dv'])+np.dot(bm,config['dm'])
    return float(derivative),float(loss),bx,bv,bm,x,v

def _oracle_trajectory_sensitivity(config: dict) -> float:
    return _trajectory_adjoint(config)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nimport copy\nX=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\nV=np.array([[.04,.11,-.03],[-.08,.02,.06],[.03,-.09,.04],[.07,.01,-.05],[-.02,-.07,.05],[.06,.08,-.04],[-.05,.03,-.02],[.01,-.04,.07]])\ni=np.arange(8)\nU=.01*np.column_stack((i-3.5,(-1.)**i,i%3-1))\nW=.02*np.column_stack(((-1.)**(i+1),(i-2)/3,np.ones(8)))\nmu=.03*(-1.)**i\nY=X+.08*V+np.column_stack((.015*(-1.)**i,-.01*np.ones(8),.005*(i-3)))\ntree=dict(leaf_ids=np.array([0,0,1,1,2,2,3,3]),parent_ids=np.array([0,0,1,1]),leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nconfig=dict(x=X,v=V,masses=np.array([.8,1.1,.9,1.2,1.05,.75,1.15,.95]),weights=np.array([1,1.3,.7,1.1,.9,1.4,.8,1.2]),target=Y,dx=U,dv=W,dm=mu,tree=tree,epsilon=.18,p=4,G=.7,h=.025,steps=4,lam=.05)\n', 'call': 'trajectory_sensitivity(copy.deepcopy(config))', 'gold_call': '_oracle_trajectory_sensitivity(copy.deepcopy(config))', 'tol': 1e-09}, {'setup': "import numpy as np\nimport copy\nX=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\nV=np.array([[.04,.11,-.03],[-.08,.02,.06],[.03,-.09,.04],[.07,.01,-.05],[-.02,-.07,.05],[.06,.08,-.04],[-.05,.03,-.02],[.01,-.04,.07]])\ni=np.arange(8)\nU=.01*np.column_stack((i-3.5,(-1.)**i,i%3-1))\nW=.02*np.column_stack(((-1.)**(i+1),(i-2)/3,np.ones(8)))\nmu=.03*(-1.)**i\nY=X+.08*V+np.column_stack((.015*(-1.)**i,-.01*np.ones(8),.005*(i-3)))\ntree=dict(leaf_ids=np.array([0,0,1,1,2,2,3,3]),parent_ids=np.array([0,0,1,1]),leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nconfig=dict(x=X,v=V,masses=np.array([.8,1.1,.9,1.2,1.05,.75,1.15,.95]),weights=np.array([1,1.3,.7,1.1,.9,1.4,.8,1.2]),target=Y,dx=U,dv=W,dm=mu,tree=tree,epsilon=.18,p=4,G=.7,h=.025,steps=4,lam=.05)\nconfig['steps']=0\n", 'call': 'trajectory_sensitivity(copy.deepcopy(config))', 'gold_call': '_oracle_trajectory_sensitivity(copy.deepcopy(config))', 'tol': 1e-09}, {'setup': "import numpy as np\nimport copy\nX=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\nV=np.array([[.04,.11,-.03],[-.08,.02,.06],[.03,-.09,.04],[.07,.01,-.05],[-.02,-.07,.05],[.06,.08,-.04],[-.05,.03,-.02],[.01,-.04,.07]])\ni=np.arange(8)\nU=.01*np.column_stack((i-3.5,(-1.)**i,i%3-1))\nW=.02*np.column_stack(((-1.)**(i+1),(i-2)/3,np.ones(8)))\nmu=.03*(-1.)**i\nY=X+.08*V+np.column_stack((.015*(-1.)**i,-.01*np.ones(8),.005*(i-3)))\ntree=dict(leaf_ids=np.array([0,0,1,1,2,2,3,3]),parent_ids=np.array([0,0,1,1]),leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nconfig=dict(x=X,v=V,masses=np.array([.8,1.1,.9,1.2,1.05,.75,1.15,.95]),weights=np.array([1,1.3,.7,1.1,.9,1.4,.8,1.2]),target=Y,dx=U,dv=W,dm=mu,tree=tree,epsilon=.18,p=4,G=.7,h=.025,steps=4,lam=.05)\nconfig['steps']=1\nconfig['p']=2\nconfig['epsilon']=.35\nconfig['dx']=np.zeros_like(X)\nconfig['dv']=np.zeros_like(V)\n", 'call': 'trajectory_sensitivity(copy.deepcopy(config))', 'gold_call': '_oracle_trajectory_sensitivity(copy.deepcopy(config))', 'tol': 1e-09}, {'setup': 'import numpy as np\nimport copy\nX=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\nV=np.array([[.04,.11,-.03],[-.08,.02,.06],[.03,-.09,.04],[.07,.01,-.05],[-.02,-.07,.05],[.06,.08,-.04],[-.05,.03,-.02],[.01,-.04,.07]])\ni=np.arange(8)\nU=.01*np.column_stack((i-3.5,(-1.)**i,i%3-1))\nW=.02*np.column_stack(((-1.)**(i+1),(i-2)/3,np.ones(8)))\nmu=.03*(-1.)**i\nY=X+.08*V+np.column_stack((.015*(-1.)**i,-.01*np.ones(8),.005*(i-3)))\ntree=dict(leaf_ids=np.array([0,0,1,1,2,2,3,3]),parent_ids=np.array([0,0,1,1]),leaf_centers=np.array([[-1.1,-.1,0],[-1.1,.2,0],[1.1,-.1,0],[1.1,.2,0]]),parent_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]))\nconfig=dict(x=X,v=V,masses=np.array([.8,1.1,.9,1.2,1.05,.75,1.15,.95]),weights=np.array([1,1.3,.7,1.1,.9,1.4,.8,1.2]),target=Y,dx=U,dv=W,dm=mu,tree=tree,epsilon=.18,p=4,G=.7,h=.025,steps=4,lam=.05)\nconfig[\'steps\']=3\nconfig[\'dx\']=np.zeros_like(X)\nconfig[\'dm\']=np.zeros(8)\n', 'call': 'trajectory_sensitivity(copy.deepcopy(config))', 'gold_call': '_oracle_trajectory_sensitivity(copy.deepcopy(config))', 'tol': 1e-9},
 {'setup': 'import numpy as np\nimport copy\nX=np.array([[-1.18,-.16,.08],[-1.02,-.04,-.07],[-.91,.18,.04],[-1.13,.26,-.09],[1.06,-.12,.13],[1.24,.02,-.08],[.96,.19,.02],[1.17,.28,-.11]])\nV=np.array([[.04,.11,-.03],[-.08,.02,.06],[.03,-.09,.04],[.07,.01,-.05],[-.02,-.07,.05],[.06,.08,-.04],[-.05,.03,-.02],[.01,-.04,.07]])\ni=np.arange(8)\nU=.01*np.column_stack((i-3.5,(-1.)**i,i%3-1))\nW=.02*np.column_stack(((-1.)**(i+1),(i-2)/3,np.ones(8)))\nmu=.03*(-1.)**i\nY=X+.08*V+np.column_stack((.015*(-1.)**i,-.01*np.ones(8),.005*(i-3)))\ntree=dict(leaf_ids=np.array([0,0,0,0,1,1,1,1]),parent_ids=np.array([0,0]),leaf_centers=np.array([[-1.1,.05,0],[1.1,.05,0]]),parent_centers=np.array([[0.,.05,0]]))\nconfig=dict(x=X,v=V,masses=np.array([.8,1.1,.9,1.2,1.05,.75,1.15,.95]),weights=np.array([1,1.3,.7,1.1,.9,1.4,.8,1.2]),target=Y,dx=U,dv=W,dm=mu,tree=tree,epsilon=.18,p=4,G=.7,h=.025,steps=4,lam=.05)\nconfig[\'steps\']=2\nconfig[\'p\']=3\n', 'call': 'trajectory_sensitivity(copy.deepcopy(config))', 'gold_call': '_oracle_trajectory_sensitivity(copy.deepcopy(config))', 'tol': 1e-9}]
