"""
Differentiate the physically calibrated unidirectional cloak twice as its inner-shell conductivity varies exponentially and the compensation shell is retuned to keep the exterior dipole moment fixed.

Material robustness depends on both the changing inner shell and the compensation required to retain the original physical cancellation. Differentiating the discrete equilibrium and its scalar cancellation constraint on a deforming mesh couples material, shape and sampling-point derivatives.

Returns
-------
tuple[np.ndarray, np.ndarray]: shell derivatives (3,) and temperature derivatives (3,n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def differentiate_compensation_design(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    outer_conductivity: float,
    background_conductivity: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    r"""Return ordinary derivatives of the retuned shell and design temperature.

    At $\eta=0$ the isotropic conductivity is ``background_conductivity`` in
    regions 0 and 3, ``outer_conductivity`` in region 1 and ``inner_conductivity``
    in region 2. For nearby $\eta$, region 2 has ``inner_conductivity`` $\cdot\,e^{\eta}$.
    Region 1 has $q(\eta)$, with $q(0)$ = ``outer_conductivity``, chosen so that
    $\operatorname{mean}[x\,(T_\eta-x)]$ over the exterior nodes stays equal to its value at
    $\eta=0$. Exterior nodes and the $x$-drive P1 solve have the definitions in
    calibrate_compensation_shell. A calibrated input therefore stays on
    the zero-moment branch, up to its supplied root tolerance.

    Compute derivatives of these discrete equations at zero, holding
    connectivity, labels, background and boundary nodes fixed. Interior
    nodes follow $X(\eta)=X_0+\eta V+\eta^2W/2$, where $X_0$ = ``nodes`` and $[V,W]$ =
    ``node_derivatives``. Element labels are transported with their triangles.
    The moment is $\operatorname{mean}[X_x(\eta)\,(T(\eta)-X_x(\eta))]$ over the same exterior
    node indices. Its weights and reference field therefore move as well.
    Differentiate physical element areas and P1 basis gradients, including
    all geometry/material cross terms. These are
    derivatives of the implicit moment constraint, not derivatives of
    bisection decisions. A jet stores ordinary derivatives $[f,f',f'']$
    (value, first, second), so its second row/entry is not divided by 2. Use differentiated
    linear systems; the input contains no finite-difference step size.

    Parameters
    ----------
    nodes : np.ndarray
        Mesh coordinates, shape (n,2), as in solve_steady_conduction.
    triangles : np.ndarray
        Integer connectivity, shape (e,3); either orientation is allowed.
    labels : np.ndarray
        Integer region labels of shape (e,), taking values 0 through 3.
    inner_conductivity : float
        Finite positive inner-shell conductivity at $\eta=0$.
    outer_conductivity : float
        Finite positive compensation conductivity at $\eta=0$.
    background_conductivity : float
        Finite positive, $\eta$-independent core/background conductivity.
    node_derivatives : np.ndarray or None
        Ordinary coordinate derivatives $[V,W]$, shape (2,n,2), finite and
        exactly zero on boundary nodes. None means a fixed mesh. The
        baseline mesh is nondegenerate; only a local curve is required.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        q_jet of shape (3,), followed by temperature_jet of shape (3,n).
        The temperature derivatives are zero on the boundary.

    Raises
    ------
    ValueError
        For invalid region labels, conductivities or node derivatives,
        nonzero boundary motion, rejected mesh inputs,
        no exterior node, or absolute partial derivative of the exterior
        moment with respect to $q$ at most $10^{-14}$ (no regular local branch).
    """
    return shell_jet, temperature_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

def _cloak_coordinate_jet(nodes, node_derivatives):
    jet = np.zeros((3, len(nodes), 2))
    jet[0] = nodes
    if node_derivatives is not None:
        motion = np.asarray(node_derivatives, float)
        if motion.shape != (2, len(nodes), 2) or not np.all(np.isfinite(motion)):
            raise ValueError('node derivatives must be finite (2,n,2)')
        boundary = np.max(np.abs(nodes), axis=1) >= 1-1e-12
        if np.any(motion[:, boundary] != 0):
            raise ValueError('boundary nodes must stay fixed')
        jet[1:] = motion
    return jet

def _cloak_geometry_jet(coordinates, triangles):
    p = coordinates[:, triangles]
    a, b = p[:, :, 1]-p[:, :, 0], p[:, :, 2]-p[:, :, 0]
    jac = np.stack([a, b], axis=-1)
    inv = np.empty_like(jac)
    inv[0] = np.linalg.inv(jac[0])
    inv[1] = -inv[0] @ jac[1] @ inv[0]
    inv[2] = 2*inv[0] @ jac[1] @ inv[0] @ jac[1] @ inv[0] - inv[0] @ jac[2] @ inv[0]
    reference = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    basis = np.einsum('ia,reab->reib', reference, inv)
    def _cross(u, v):
        return u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    determinant = np.stack([_cross(a[0], b[0]),
                            _cross(a[1], b[0])+_cross(a[0], b[1]),
                            _cross(a[2], b[0])+2*_cross(a[1], b[1])+_cross(a[0], b[2])])
    area = .5*np.sign(determinant[0])*determinant
    return area, basis

def _cloak_matrix_jet(triangles, area, basis, tensors, size):
    def _term(i, j, k):
        return np.einsum('eia,eab,ejb->eij', basis[i], tensors[j], basis[k])
    local0 = _term(0, 0, 0)
    local1 = _term(1, 0, 0)+_term(0, 1, 0)+_term(0, 0, 1)
    local2 = (_term(2, 0, 0)+_term(0, 2, 0)+_term(0, 0, 2)
              +2*(_term(1, 1, 0)+_term(1, 0, 1)+_term(0, 1, 1)))
    local = [area[0, :, None, None]*local0,
             area[1, :, None, None]*local0+area[0, :, None, None]*local1,
             area[2, :, None, None]*local0+2*area[1, :, None, None]*local1
             +area[0, :, None, None]*local2]
    rows = np.broadcast_to(triangles[:, :, None], local0.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], local0.shape).ravel()
    return [coo_matrix((v.ravel(), (rows, cols)), shape=(size, size)).tocsr() for v in local]

def _cloak_temperature_jet(nodes, triangles, tensor_jet, angle, node_derivatives=None):
    t0 = _oracle_solve_steady_conduction(nodes, triangles, tensor_jet[0], angle)
    coordinates = _cloak_coordinate_jet(nodes, node_derivatives)
    area, basis = _cloak_geometry_jet(coordinates, triangles)
    mats = _cloak_matrix_jet(triangles, area, basis, tensor_jet, len(nodes))
    interior = np.flatnonzero(np.max(np.abs(nodes), axis=1) < 1-1e-12)
    factor = splu(mats[0][interior][:, interior].tocsc())
    jet = np.zeros((3, len(nodes)))
    jet[0] = t0
    jet[1, interior] = factor.solve(-(mats[1] @ t0)[interior])
    jet[2, interior] = factor.solve(-(mats[2] @ t0+2*mats[1] @ jet[1])[interior])
    return jet

def _oracle_differentiate_compensation_design(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    outer_conductivity: float,
    background_conductivity: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Differentiate the moving-domain equilibrium and moment constraint."""
    x, tri, lab = np.asarray(nodes, float), np.asarray(triangles), np.asarray(labels)
    if (lab.shape != (len(tri),) or not np.issubdtype(lab.dtype, np.integer)
            or np.any((lab < 0) | (lab > 3))):
        raise ValueError('invalid region labels')
    vals = np.asarray([background_conductivity, outer_conductivity, inner_conductivity], float)
    if vals.shape != (3,) or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError('conductivities must be finite positive scalars')
    k = np.array([vals[0], vals[1], vals[2], vals[0]])[lab]
    t0 = _oracle_solve_steady_conduction(x, tri, k[:, None, None]*np.eye(2), 0.)
    coordinates = _cloak_coordinate_jet(x, node_derivatives)
    interior = np.flatnonzero(np.max(np.abs(x), axis=1) < 1-1e-12)
    touched = np.zeros(len(x), bool)
    touched[tri[lab != 0].ravel()] = True
    exterior = (~touched) & (np.max(np.abs(x), axis=1) < 1-1e-12)
    if not np.any(exterior):
        raise ValueError('there is no exterior node')
    area, basis = _cloak_geometry_jet(coordinates, tri)
    kfixed = np.stack([k, (lab == 2)*vals[2], (lab == 2)*vals[2]])
    mats = _cloak_matrix_jet(tri, area, basis, kfixed[:, :, None, None]*np.eye(2), len(x))
    outer = np.zeros_like(kfixed)
    outer[0] = lab == 1
    outer_mats = _cloak_matrix_jet(tri, area, basis, outer[:, :, None, None]*np.eye(2), len(x))
    factor = splu(mats[0][interior][:, interior].tocsc())
    def _response(rhs):
        z = np.zeros(len(x))
        z[interior] = factor.solve(rhs[interior])
        return z
    def _moment(z):
        return np.mean(x[exterior, 0]*z[exterior])
    tq = _response(-outer_mats[0] @ t0)
    slope = _moment(tq)
    if abs(slope) <= 1e-14:
        raise ValueError('singular compensation constraint')
    tf1 = _response(-mats[1] @ t0)
    xx, vx, wx = coordinates[:, exterior, 0]
    m1_fixed = np.mean(vx*(t0[exterior]-xx)+xx*(tf1[exterior]-vx))
    q1 = -m1_fixed/slope
    t1 = tf1+q1*tq
    a1 = mats[1]+q1*outer_mats[0]
    # The second total stiffness includes shape/material mixed derivatives.
    a2_fixed = mats[2]+2*q1*outer_mats[1]
    tf2 = _response(-a2_fixed @ t0-2*a1 @ t1)
    m2_fixed = np.mean(wx*(t0[exterior]-xx)+2*vx*(t1[exterior]-vx)
                       +xx*(tf2[exterior]-wx))
    q2 = -m2_fixed/slope
    t2 = tf2+q2*tq
    return np.array([outer_conductivity, q1, q2]), np.stack([t0, t1, t2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Explicit normal, boundary and shape-derivative cases."""
    return [{'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(11)\n'
               'TR[::3]=TR[::3,::-1]\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.1,2.0,1.0),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.1,2.0,1.0),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(14)\n'
               'TR[::3]=TR[::3,::-1]\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.03,3.0,1.0),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.03,3.0,1.0),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(13)\n'
               'TR[::3]=TR[::3,::-1]\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.3,2.2,1.7),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.3,2.2,1.7),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(9)\n'
               'TR[::3]=TR[::3,::-1]\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.6,1.4,1.0),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),0.6,1.4,1.0),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(11)\n'
               'L[:]=0\n',
      'call': '_status(lambda: differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(11)\n'
               'L[:]=2\n',
      'call': '_status(lambda: differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
               'def _status(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    return 0\n'
               'X,TR,L=_device(11)\n'
               'L[0]=4\n',
      'call': '_status(lambda: differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(12)\n'
               'M=_motion(X)\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.4,1.,node_derivatives=M.copy()),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.4,1.,node_derivatives=M.copy()),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(10)\n'
               'M=_motion(X); M[0]*=0\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.25,1.8,1.,node_derivatives=M.copy()),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.25,1.8,1.,node_derivatives=M.copy()),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(13)\n'
               'X+=.3*_motion(X)[0]\n'
               'TR[::2]=TR[::2,::-1]\n'
               'M=_motion(X)\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.06,3.1,1.3,node_derivatives=M.copy()),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.06,3.1,1.3,node_derivatives=M.copy()),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(11)\n'
               'M=np.zeros((2,len(X),2))\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(14)\n'
               'M=_motion(X); M[0]*=-1.7; M[1]*=.4\n',
      'call': '_packed(differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.21,1.9,.8,node_derivatives=M.copy()),len(X))',
      'gold_call': '_packed(_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.21,1.9,.8,node_derivatives=M.copy()),len(X))',
      'tol': 2e-08},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(9)\n'
               'M=_motion(X); M[0,0,0]=.1\n',
      'call': '_status(lambda: '
              'differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()))'},
     {'setup': '\n'
               'import numpy as np\n'
               'def _device(n):\n'
               '    a=np.linspace(-1,1,n); xx,yy=np.meshgrid(a,a)\n'
               '    x=np.column_stack([xx.ravel(),yy.ravel()])\n'
               '    tr=[]\n'
               '    for j in range(n-1):\n'
               '        for i in range(n-1):\n'
               '            p=j*n+i; tr.extend([[p,p+1,p+n+1],[p,p+n+1,p+n]])\n'
               '    tr=np.array(tr); c=x[tr].mean(1)\n'
               '    rad=np.hypot(c[:,0],1.2*c[:,1])\n'
               '    lab=np.where(rad<.18,3,np.where(rad<.36,2,np.where(rad<.62,1,0)))\n'
               '    return x,tr,lab\n'
               'def _packed(out,n):\n'
               '    q,t=map(np.asarray,out)\n'
               '    assert q.shape==(3,) and t.shape==(3,n)\n'
               '    assert np.all(np.isfinite(q)) and np.all(np.isfinite(t))\n'
               '    return np.r_[q,t.ravel()]\n'
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
               'X,TR,L=_device(9)\n'
               'M=_motion(X); M[1,12,1]=np.nan\n',
      'call': '_status(lambda: '
              'differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()))',
      'gold_call': '_status(lambda: '
                   '_oracle_differentiate_compensation_design(X.copy(),TR.copy(),L.copy(),.1,2.,1.,node_derivatives=M.copy()))'}]
