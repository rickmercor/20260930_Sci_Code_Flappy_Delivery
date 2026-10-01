"""
Motion of the coefficient region allowed by formation physicality

The supplied affine formation-family jets define $$h_i(\boldsymbol u,\varepsilon)=h_{i,0}(\varepsilon)+\sum_{a=1}^3h_{i,a}(\varepsilon)u_a$$, with each coefficient expanded as $$h(\varepsilon)=h^{(0)}+\varepsilon h^{(1)}+\tfrac12\varepsilon^2h^{(2)}+o(\varepsilon^2)$$. Intersect all selected inequalities $$h_i\ge0$$ with the fixed latent box $$\boldsymbol\ell\le\boldsymbol u\le\boldsymbol r$$. On the supported moving branch, the polytope is full dimensional with a locally unchanged set of faces and vertices, and every vertex is the intersection of three independent active supporting planes. Return each vertex and its first two true derivatives, tracking those active planes along the perturbation. Facets remain planar through second order; a triangulation of a base facet therefore continues to describe the same local moving region. Fully static input jets may describe non-simple vertices. A locally persistent empty or zero-volume region returns an empty vertex jet.

Returns
-------
np.ndarray of shape (3,V,3), giving vertex positions, first derivatives and second derivatives in the original latent coordinates; shape (3,0,3) represents a locally persistent zero-probability region.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def physicality_vertex_jet(formation_family_jet: np.ndarray,
                           window_mask: np.ndarray, lower: np.ndarray,
                           upper: np.ndarray) -> np.ndarray:
    r'''Track the vertices of a smooth physicality-region branch.

    Parameters
    ----------
    formation_family_jet : np.ndarray
        Finite real shape (3,4,N), N >= 1. Axis 0 gives true derivative orders zero, one and two with respect to the same dimensionless perturbation. Within each order, row 0 is the constant formation coefficient and rows 1–3 multiply the latent coordinates; all coefficients have rate units.
    window_mask : np.ndarray
        Boolean shape (N,), selecting at least one inequality that must remain nonnegative.
    lower, upper : np.ndarray
        Fixed finite real box bounds of shape (3,), with lower < upper coordinatewise.

    Returns
    -------
    vertex_jet : np.ndarray
        Finite shape (3,V,3), containing distinct extreme vertices with true derivative orders on axis 0. Vertex order is unrestricted, but the same vertex must occupy each column of all three derivative orders. A locally persistent empty or zero-volume intersection returns shape (3,0,3).

    Raises
    ------
    ValueError
        If finite real shapes, the nonempty boolean window or ordered box contracts fail, or numerical geometry or linear systems cannot produce finite vertex jets. Supported moving regions have simple vertices and locally fixed face incidence; topology changes and non-simple moving vertices are outside this input domain. Fully static input jets may describe non-simple vertices. These local geometric preconditions need not be numerically certified.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

def _oracle_physicality_vertex_jet(formation_family_jet: np.ndarray,
                                    window_mask: np.ndarray, lower: np.ndarray,
                                    upper: np.ndarray) -> np.ndarray:
    if any(not np.isrealobj(value) for value in (formation_family_jet, lower, upper)):
        raise ValueError("formation jets and box bounds must be real")
    try:
        family, lo, hi = [np.asarray(value, dtype=float)
                          for value in (formation_family_jet, lower, upper)]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("formation jets and box bounds must be finite real arrays") from exc
    if family.ndim != 3 or family.shape[:2] != (3, 4) or family.shape[2] < 1:
        raise ValueError("formation_family_jet must have shape (3,4,N), N >= 1")
    if lo.shape != (3,) or hi.shape != (3,) or not np.all(lo < hi):
        raise ValueError("box bounds must have shape (3,) with lower < upper")
    if not all(np.all(np.isfinite(value)) for value in (family, lo, hi)):
        raise ValueError("formation jets and box bounds must be finite")
    mask = np.asarray(window_mask)
    if mask.shape != (family.shape[2],) or mask.dtype != np.bool_ or not np.any(mask):
        raise ValueError("window_mask must be boolean, matching and nonempty")
    midpoint = lo + (hi - lo) / 2.0
    halfwidth = (hi - lo) / 2.0
    selected = family[:, :, mask]
    intercepts = selected[:, 0] + np.einsum("kan,a->kn", selected[:, 1:], midpoint)
    coefficients = selected[:, 1:].transpose(0, 2, 1) * halfwidth
    zero = np.all(coefficients[0] == 0.0, axis=1)
    if np.any(intercepts[0, zero] < 0.0):
        return np.empty((3, 0, 3), dtype=float)
    necessary = (~zero) & (intercepts[0] - np.sum(np.abs(coefficients[0]), axis=1) < 0.0)
    coefficients, intercepts = coefficients[:, necessary], intercepts[:, necessary]
    count = coefficients.shape[1]
    planes = np.zeros((3, count + 6, 4), dtype=float)
    planes[:, :count, :3] = -coefficients
    planes[:, :count, 3] = -intercepts
    planes[0, count:, :3] = np.vstack((np.eye(3), -np.eye(3)))
    planes[0, count:, 3] = -1.0
    scale = np.linalg.norm(planes[0, :, :3], axis=1)
    planes = planes / scale[None, :, None]
    if not np.all(np.isfinite(planes)):
        raise ValueError("scaled constraint jets must be finite")
    base = planes[0]
    fit = linprog(np.array([0.0, 0.0, 0.0, -1.0]),
                  A_ub=np.column_stack((base[:, :3], np.ones(len(base)))),
                  b_ub=-base[:, 3], bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
    if fit.status == 2:
        return np.empty((3, 0, 3), dtype=float)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("could not determine a finite interior point")
    if fit.x[3] <= 0.0:
        return np.empty((3, 0, 3), dtype=float)
    try:
        region = HalfspaceIntersection(base, fit.x[:3])
        if not np.any(family[1:]):
            positions = np.unique(midpoint + halfwidth * region.intersections, axis=0)
            result = np.zeros((3, len(positions), 3), dtype=float)
            result[0] = positions
            if not np.all(np.isfinite(result)):
                raise ValueError("static vertices must be finite")
            return result
        vertices = []
        for active in region.dual_facets:
            if len(active) != 3:
                raise ValueError("supported moving vertices must be simple")
            indices = np.asarray(active, dtype=int)
            matrix = planes[:, indices, :3]
            constant = planes[:, indices, 3]
            value = np.linalg.solve(matrix[0], -constant[0])
            first = np.linalg.solve(matrix[0], -constant[1] - matrix[1] @ value)
            second = np.linalg.solve(matrix[0], -constant[2] - matrix[2] @ value - 2.0 * matrix[1] @ first)
            vertices.append(np.array([midpoint + halfwidth * value,
                                      halfwidth * first, halfwidth * second]))
    except Exception as exc:
        raise ValueError("the moving polytope could not be evaluated") from exc
    result = np.asarray(vertices).transpose(1, 0, 2)
    if not np.all(np.isfinite(result)):
        raise ValueError("vertex jets must be finite")
    order = np.lexsort((result[0, :, 2], result[0, :, 1], result[0, :, 0]))
    return result[:, order]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = "import numpy as np\nfrom scipy.optimize import linprog\nfrom scipy.spatial import HalfspaceIntersection,ConvexHull\n"
    metric = '''
def geometry_response(fn):
    v=np.asarray(fn(j.copy(),m.copy(),lo.copy(),hi.copy()))
    if v.ndim!=3 or v.shape[0]!=3 or v.shape[2]!=3:
        raise ValueError("vertex jet must have shape (3,V,3)")
    directions=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[-1,-1,-1],[1,-1,1],[-1,1,-1],[2,1,-1],[-2,-1,1],[1,-2,3],[-1,2,-3]],dtype=float)
    if v.shape[1]==0:
        return np.zeros((3,58))
    hull=ConvexHull(v[0]);center=v.mean(axis=1);volume=np.zeros(3)
    for face in hull.simplices:
        b=(v[:,face]-center[:,None,:]).transpose(0,2,1)
        inverse=np.linalg.inv(b[0]);x=inverse@b[1]
        first=np.trace(x);second=np.trace(inverse@b[2]-x@x)
        volume+=abs(np.linalg.det(b[0]))*np.array([1.,first,second+first*first])/6
    projections=np.einsum('kvi,di->kvd',v,directions)
    signatures=[volume,np.array([v.shape[1],0.,0.])]
    for d in range(len(directions)):
        value,first,second=projections[:,:,d]
        for power in (1,2,3,4):
            curvature=second.sum() if power==1 else power*np.sum((power-1)*value**(power-2)*first**2+value**(power-1)*second)
            signatures.append(np.array([np.sum(value**power),power*np.sum(value**(power-1)*first),curvature]))
    return np.array(signatures).T
'''
    return [
        {"setup": imports + "j=np.zeros((3,4,1));j[0,0,0]=1.;m=np.ones(1,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.array([[[.31],[-1.],[-.4],[-.2]],[[.2],[.05],[-.1],[.03]],[[.1],[-.02],[.04],[.01]]]);m=np.ones(1,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.array([[[.37,.8,.6],[-1.,.2,.1],[-.4,-1.,.2],[-.2,.3,-1.]],[[.2,-.1,.15],[.05,-.04,.02],[-.1,.03,.08],[.03,-.02,.05]],[[.1,.04,-.06],[-.02,.01,-.03],[.04,-.01,.02],[.01,.03,-.02]]]);m=np.ones(3,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.zeros((3,4,4));j[0]=np.array([[0.,0.,0.,.8],[1.,0.,0.,-1.],[0.,1.,0.,-1.],[0.,0.,1.,-1.]]);j[1,:,3]=np.array([.12,.04,-.03,.02]);j[2,:,3]=np.array([-.05,.02,.01,-.03]);m=np.ones(4,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.zeros((3,4,2));j[0]=np.array([[-1.,.2],[0.,1.],[0.,-.5],[0.,.3]]);j[1,:,1]=[.1,.03,.02,-.01];j[2,:,1]=[-.04,.01,-.02,.03];m=np.array([False,True]);lo=np.array([-1.2,-.8,-1.]);hi=np.array([.7,1.,.9])\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.zeros((3,4,1));j[0,0,0]=-2.;j[0,1,0]=1.;m=np.ones(1,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "j=np.zeros((3,4,2));j[0,1]=[1.,-1.];m=np.ones(2,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
        {"setup": imports + "normals=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]]);j=np.zeros((3,4,8));j[0,0]=.8;j[0,1:]=-normals.T;m=np.ones(8,dtype=bool);lo=-np.ones(3);hi=np.ones(3)\n" + metric,
         "call": "geometry_response(physicality_vertex_jet)", "gold_call": "geometry_response(_oracle_physicality_vertex_jet)", "tol": 1e-6},
    ]
