"""
Infer the finite-quadrature QCD target and propagate its correlated uncertainty.

Implement the complete inverse and predictive problem in the task. This public orchestrator must call and combine every preceding public function, directly or transitively. Construct the physical MV input, apply initial matching, evolve and match each probe, fit Q0 and a in the stated rectangle, then propagate the absolute-error information covariance. The admissible datasets have an identifiable interior minimizer. This inverse experiment and the probe family are task-defined extensions of the source method.

Returns
-------
return result  # ndarray, shape (6,), ordered [Q0, a, chi2, predicted M, correlation rho, uncertainty delta_M]. Only Q0 has units (GeV); all other entries are dimensionless. The six entries follow the task definitions and precision rules.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_rotated_cgc(points: "np.ndarray", weights: "np.ndarray", measurements: "np.ndarray", prediction: "np.ndarray", mu2: float = 1.3, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """points : ndarray, shape (n,2), distinct transverse coordinates in GeV^-1, n >= 3.
    weights : ndarray, shape (n,), nonnegative weights for d^2x/(2*pi), in GeV^-2.
    measurements : ndarray, shape (m,5), m >= 3; columns [Y,b,epsilon,observed M,sigma].
    prediction : ndarray, shape (3,), [Y,b,epsilon].
    mu2 : positive float, collinear scale squared in GeV^2.
    nf : nonnegative integer flavor count.
    nc : integer color count >= 2.
    Probe domains are Y >= 0, b > 0, abs(epsilon) < 1, sigma > 0.
    Lambda=0.2 GeV; bounds are Q0 in [0.55,1.4] GeV, a in [0.08,0.32].

    Returns
    -------
    ndarray, shape (6,), ordered [Q0, a, chi2, predicted M, correlation rho, uncertainty delta_M].
    Only Q0 has units (GeV); all other entries are dimensionless.
    The six entries follow the task definitions and precision rules.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _geometry(points, weights):
    points=np.asarray(points,dtype=float);weights=np.asarray(weights,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or weights.shape!=(len(points),):
        raise ValueError('Expected points (n,2) and weights (n,), n >= 3.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Coordinates and nonnegative quadrature weights must be finite.')
    d=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    if np.any(d[np.triu_indices(len(points),1)]<=0):raise ValueError('Nodes must be distinct.')
    return points,weights,d


def _oracle_infer_rotated_cgc(points: "np.ndarray", weights: "np.ndarray", measurements: "np.ndarray", prediction: "np.ndarray", mu2: float = 1.3, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """Fit the MV saturation scale/coupling and propagate the absolute observation covariance."""
    import numpy as np
    from scipy.optimize import least_squares
    p,w,d=_geometry(points,weights);data=np.asarray(measurements,dtype=float);pred=np.asarray(prediction,dtype=float)
    if data.ndim!=2 or data.shape[1]!=5 or len(data)<3 or pred.shape!=(3,):raise ValueError('Invalid measurement table or prediction.')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(pred)) or np.any(data[:,4]<=0):raise ValueError('Invalid observations.')
    queries=np.vstack([data[:,:3],pred])
    if np.any(queries[:,0]<0) or np.any(queries[:,1]<=0) or np.any(abs(queries[:,2])>=1):raise ValueError('Invalid probe parameters.')
    one=_oracle_emission_kernels(p,w,mu2,nf,nc);two=_oracle_pair_kernels(p,w,nf,nc)
    dx=p[:,None,0]-p[None,:,0];dy=p[:,None,1]-p[None,:,1]
    probes=[]
    for Y,b,eps in queries:
        h=np.triu(np.exp(-d/(2*b*b))*(d+eps*(dx*dx-dy*dy)),1)
        probes.append(h/h.sum())
    radius=np.sqrt(d);safe=np.where(radius==0,1,radius)
    mv_exponent=-.25*d*np.log(1/(.2*safe)+np.e)
    def _model(theta,indices):
        Q0,alpha=theta
        physical=np.exp(Q0*Q0*mv_exponent)
        initial=_oracle_composite_matching(physical,one,alpha,-1)
        values=[]
        for i in indices:
            evolved=_oracle_evolve_rotated_dipole(initial,one,two,alpha,queries[i,0])
            matched=_oracle_composite_matching(evolved,one,alpha,1)
            values.append(np.sum(probes[i]*(1-matched)))
        return np.array(values)
    ids=np.arange(len(data));future=[len(data)]
    def _residual(theta):return (_model(theta,ids)-data[:,3])/data[:,4]
    fits=[]
    for start in ([.72,.12],[.98,.21],[1.25,.29]):
        fit=least_squares(_residual,start,bounds=([.55,.08],[1.4,.32]),xtol=2e-12,ftol=2e-12,gtol=2e-8,diff_step=2e-5,max_nfev=120)
        fits.append(fit)
    fit=min(fits,key=lambda r:(np.dot(r.fun,r.fun),r.x[0],r.x[1]))
    theta=fit.x
    if not fit.success or np.any(theta<np.array([.55001,.08001])) or np.any(theta>np.array([1.39999,.31999])):raise ValueError('An interior identifiable fit is required.')
    jac=np.zeros((len(data),2));g=np.zeros(2)
    # Symmetric five-point derivatives checked independently against complex-step tangents.
    for j in range(2):
        step=np.zeros(2);step[j]=2e-4
        jac[:,j]=(-_residual(theta+2*step)+8*_residual(theta+step)-8*_residual(theta-step)+_residual(theta-2*step))/(12*step[j])
        g[j]=(-_model(theta+2*step,future)[0]+8*_model(theta+step,future)[0]-8*_model(theta-step,future)[0]+_model(theta-2*step,future)[0])/(12*step[j])
    information=jac.T@jac
    if np.linalg.eigvalsh(information)[0]<=1e-9:raise ValueError('The fit is not identifiable.')
    covariance=np.linalg.inv(information)
    rho=covariance[0,1]/np.sqrt(covariance[0,0]*covariance[1,1])
    error=np.sqrt(g@covariance@g)
    return np.array([theta[0],theta[1],np.dot(fit.fun,fit.fun),_model(theta,future)[0],rho,error])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\ndata=np.array([[0.35, 0.65, 0.3, 0.427358983, 0.0003], [1.1, 1.25, -0.4, 0.578560082, 0.00025], [0.7, 0.9, 0.55, 0.514189898, 0.0002], [1.6, 0.75, -0.2, 0.483489813, 0.0003]], dtype=float)\npred=np.array([2.0, 0.95, 0.45], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\ndata=np.array([[0.1, 0.6, 0.2, 0.3077514399, 0.00028], [0.4, 1.1, -0.3, 0.4399753268, 0.00022], [0.8, 0.8, 0.5, 0.3787436874, 0.00031], [1.2, 1.4, -0.2, 0.4777823823, 0.00026]], dtype=float)\npred=np.array([1.5, 0.7, 0.6], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\ndata=np.array([[0.3, 0.7, -0.2, 0.5595509968, 0.00028], [0.7, 1.2, 0.4, 0.6860342064, 0.00022], [1.3, 0.9, -0.5, 0.6395149087, 0.00031], [1.7, 1.3, 0.1, 0.7072896736, 0.00026]], dtype=float)\npred=np.array([2.2, 1.1, -0.4], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,1.3,0,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,1.3,0,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\ndata=np.array([[0.2, 0.65, 0.3, 0.3862467301, 0.00028], [0.8, 1.2, -0.4, 0.528282465, 0.00022], [1.1, 0.85, 0.5, 0.4680979747, 0.00031], [1.5, 1.1, -0.2, 0.5250324387, 0.00026]], dtype=float)\npred=np.array([1.8, 0.75, 0.2], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,0.55,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,0.55,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14], dtype=float)\ndata=np.array([[0.3, 0.6, 0.2, 0.6351511356, 0.00028], [0.7, 1.3, -0.3, 0.7648211936, 0.00022], [1.2, 0.9, 0.5, 0.727550536, 0.00031], [1.8, 1.5, -0.4, 0.7779065768, 0.00026]], dtype=float)\npred=np.array([2.1, 0.85, -0.1], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,1.3,3,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\ndata=np.array([[0.2, 0.55, 0.4, 0.5850149251, 0.00028], [0.5, 1.2, -0.5, 0.7635165112, 0.00022], [1.0, 0.95, 0.2, 0.7314191416, 0.00031], [1.4, 1.4, -0.3, 0.7903514979, 0.00026]], dtype=float)\npred=np.array([1.9, 0.8, 0.55], dtype=float)\n', 'call': 'infer_rotated_cgc(p,w,data,pred,1.3,5,3)*np.array([1.,1.,1.,1.,1.,100.])', 'gold_call': '_oracle_infer_rotated_cgc(p,w,data,pred,1.3,5,3)*np.array([1.,1.,1.,1.,1.,100.])', 'tol': 3e-06}]
