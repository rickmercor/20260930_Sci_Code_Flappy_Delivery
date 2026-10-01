"""
Propagate material and design-field derivatives through the physical-geometric duality and compute the first and second derivatives of each device's worst-direction far-field disturbance.

Changing a physical design rotates its local gradient frame as well as changing the converted principal conductivities. Its worst-direction disturbance changes with the physical response and with the maximizing applied direction.

Returns
-------
np.ndarray: first and second $D^*$ derivatives for original and converted devices, shape (2,2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_duality_response(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity_jet: "np.ndarray",
    temperature_jet: "np.ndarray",
    background_conductivity: float,
    far_threshold: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "np.ndarray":
    r"""Return the first two ordinary derivatives of both angular maxima.

    The input jets specify smooth curves $k_e(\eta)>0$ and $T(\eta)$ at $\eta=0$
    through their value, first derivative and second derivative, without
    factorial scaling. The first device has isotropic conductivity $k_e$.
    The second has the duality conductivity assigned to $k_e$ and the P1
    gradient of $T$ on each triangle, for fixed reference gradient $(1,0)$
    and background_conductivity. Thus both principal values and local
    principal directions change with $\eta$. $T$ is the field generating the
    conversion; for a calibrated design it is the output of
    differentiate_compensation_design. This interface also accepts any
    smooth generating-field jet with a nonzero triangle gradient.

    For each $\eta$, evaluate both devices using the P1 conduction and far
    node conventions of compute_worst_direction_disturbance, with fixed
    boundary data $x\cos\varphi+y\sin\varphi$. Differentiate the maximum over
    all real $\varphi$, allowing the maximizing direction to change. Return
    derivatives at zero for each device, with the original first and the
    converted second. These are derivatives of $D^*$, not of $(D^*)^2$ or $\log D^*$.
    Use differentiate_duality_jacobian and then differentiate_duality_tensor
    for the converted constitutive jet,
    and differentiated equilibrium and angular-response equations.
    Coordinates follow $X_0+\eta V+\eta^2W/2$, with $X_0$ = nodes and $[V,W]$ given by
    node_derivatives; connectivity, boundary nodes and background stay fixed.
    Select the far-node indices at $\eta=0$ and track those nodes. At each
    $\eta$ the reference field is evaluated at their current coordinates.
    Nodal temperature jets are total derivatives along node trajectories.
    Differentiate element areas and physical P1 gradients as well as
    material values, the local frame, and the maximizing drive direction.

    Parameters
    ----------
    nodes : np.ndarray
        Coordinates of shape (n,2) in the square.
    triangles : np.ndarray
        Integer connectivity of shape (e,3), in either orientation.
    conductivity_jet : np.ndarray
        Finite array (3,e) of $[k,k',k'']$; row zero is strictly positive.
    temperature_jet : np.ndarray
        Finite array (3,n) of generating-field ordinary derivatives.
    background_conductivity : float
        Finite positive, constant background conductivity.
    far_threshold : float
        Finite max-norm far-field threshold in $[0,1)$, used at $\eta=0$.
    node_derivatives : np.ndarray or None
        Finite ordinary coordinate derivatives $[V,W]$, shape (2,n,2),
        exactly zero at boundary nodes. None means a fixed mesh.

    Returns
    -------
    np.ndarray
        Shape (2,2): rows original/converted; columns $[(D^*)',(D^*)'']$.

    Raises
    ------
    ValueError
        For invalid jets, background, threshold or node derivatives,
        nonzero boundary motion, rejected mesh data,
        a baseline generating gradient of norm at most $10^{-14}$, no far node,
        or a nonregular angular maximum for either device. Nonregular means
        the largest eigenvalue of its far-error Gram matrix is at most
        $10^{-24}$ or its eigenvalue gap is at most $10^{-10}$ times that eigenvalue.
    """
    return response_derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_differentiate_duality_response(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity_jet: "np.ndarray",
    temperature_jet: "np.ndarray",
    background_conductivity: float,
    far_threshold: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Differentiate the gradient projector, FEM equilibria and simple eigenvalue."""
    x,tri=np.asarray(nodes,float),np.asarray(triangles)
    k,t=np.asarray(conductivity_jet,float),np.asarray(temperature_jet,float)
    if (k.shape!=(3,len(tri)) or t.shape!=(3,len(x))
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(t)) or np.any(k[0]<=0)):
        raise ValueError('invalid material or temperature jets')
    if not np.isfinite(background_conductivity) or background_conductivity<=0:
        raise ValueError('background must be finite and positive')
    if not np.isfinite(far_threshold) or not 0<=far_threshold<1:
        raise ValueError('far threshold must be in [0,1)')
    identity=np.eye(2)
    original=k[:,:,None,None]*identity
    # The first reference solve validates mesh geometry before its reuse.
    original_x=_cloak_temperature_jet(x,tri,original,0.,node_derivatives)
    coordinates=_cloak_coordinate_jet(x,node_derivatives)
    area,basis=_cloak_geometry_jet(coordinates,tri)
    g=np.empty((3,len(tri),2))
    g[0]=np.einsum('ej,eja->ea',t[0,tri],basis[0])
    g[1]=(np.einsum('ej,eja->ea',t[1,tri],basis[0])
          +np.einsum('ej,eja->ea',t[0,tri],basis[1]))
    g[2]=(np.einsum('ej,eja->ea',t[2,tri],basis[0])
          +2*np.einsum('ej,eja->ea',t[1,tri],basis[1])
          +np.einsum('ej,eja->ea',t[0,tri],basis[2]))
    jacjet=_oracle_differentiate_duality_jacobian(k,g,background_conductivity)
    conv=_oracle_differentiate_duality_tensor(jacjet,background_conductivity)
    radius=np.max(np.abs(x),axis=1)
    far=(radius<1-1e-12)&(radius>far_threshold)
    if not np.any(far):
        raise ValueError('there is no far node')
    result=[]
    for tensors,tx in [(original,original_x),(conv,None)]:
        if tx is None:
            tx=_cloak_temperature_jet(x,tri,tensors,0.,node_derivatives)
        ty=_cloak_temperature_jet(x,tri,tensors,np.pi/2,node_derivatives)
        e=np.stack([tx[:,far],ty[:,far]],axis=-1)
        e-=coordinates[:,far]
        c=e[0].T@e[0]/far.sum()
        c1=(e[1].T@e[0]+e[0].T@e[1])/far.sum()
        c2=(e[2].T@e[0]+2*e[1].T@e[1]+e[0].T@e[2])/far.sum()
        eig,vec=np.linalg.eigh(c)
        lam,gap=eig[-1],eig[-1]-eig[0]
        if lam<=1e-24 or gap<=1e-10*lam:
            raise ValueError('nonregular worst direction')
        v,w=vec[:,1],vec[:,0]
        l1=v@c1@v
        l2=v@c2@v+2*(w@c1@v)**2/gap
        d=np.sqrt(lam)
        result.append([l1/(2*d),l2/(2*d)-l1*l1/(4*d**3)])
    return np.array(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Explicit normal, boundary and shape-derivative cases."""
    return [{'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(9,1.0)\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.0,0.55))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.0,0.55))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(12,0.3)\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.4,0.6))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.4,0.6))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(15,2.0)\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),0.7,0.4))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),0.7,0.4))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(10,0.8)\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.0,0.7))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.0,0.7))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(11,1.)\n'
               'K[1:]*=0; T[1:]*=0\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(11,1.)\n'
               'K[1:]*=-2; K[2]*=-2; T[1:]*=-2; T[2]*=-2\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(11,1.)\n'
               'T[0]*=0\n',
      'call': '_status(lambda: differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(11,1.)\n'
               'K[0]*=0\n',
      'call': '_status(lambda: differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,K,T=_data(11,1.)\n'
               'K[0]=1; K[1:]*=0; T[0]=X[:,0]; T[1:]*=0\n',
      'call': '_status(lambda: differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(12,.8)\n'
               'M=_motion(X)\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(10,1.2)\n'
               'M=_motion(X); M[0]*=0\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(13,1.)\n'
               'M=_motion(X)\n'
               'T[0]=X[:,0]+.3*X[:,1]\n'
               'T[1:]=M[:,:,0]+.3*M[:,:,1]\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(11,.7)\n'
               'M=_motion(X)\n'
               'K[1:]*=0\n'
               'T[0]=X[:,0]; T[1:]=M[:,:,0]\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(14,1.4)\n'
               'M=_motion(X)\n'
               'M[0]*=-2; M[1]*=4\n'
               'K[1]*=-2; K[2]*=4\n'
               'T[1]*=-2; T[2]*=4\n',
      'call': '_values(differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_values(_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(10,1.)\n'
               'M=_motion(X); M[1,0,1]=.01\n',
      'call': '_status(lambda: '
              'differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _data(n,scale):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()]); tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    v=np.exp(.9*np.cos(2*c[:,0]+.6*c[:,1])+.2*np.sin(c[:,0]+1.4*c[:,1]))*scale\n'
               '    h=.3*np.sin(c[:,0]-2*c[:,1])+.2*np.cos(1.3*c[:,0]-.7*c[:,1])\n'
               '    k=np.stack([v,v*h,v*(h*h+.2*np.cos(3*c[:,1]))])\n'
               '    xx,yy=x.T; env=(1-xx*xx)*(1-yy*yy)\n'
               '    t=np.stack([xx+.12*env*np.sin(xx+yy),\n'
               '                .23*env*np.cos(2*xx-yy),.17*env*np.sin(xx-3*yy)])\n'
               '    tr[::4]=tr[::4,::-1]\n'
               '    return x,tr,k,t\n'
               'def _values(v):\n'
               '    v=np.asarray(v)\n'
               '    assert v.shape==(2,2) and np.all(np.isfinite(v))\n'
               '    return v.ravel()\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               '\n'
               'def _motion(x):\n'
               '    u,v=x.T\n'
               '    b=(1-u*u)*(1-v*v)\n'
               '    return np.stack([b[:,None]*np.column_stack([.2*u+.17*v, -.11*u+.14*v]),\n'
               '                     b[:,None]*np.column_stack([.12*np.sin(u+2*v), .08*np.cos(2*u-v)])])\n'
               'X,TR,K,T=_data(10,1.)\n'
               'M=np.zeros((3,len(X),2))\n',
      'call': '_status(lambda: '
              'differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_duality_response(X.copy(),TR.copy(),K.copy(),T.copy(),1.,.55,node_derivatives=M.copy()))'}]
