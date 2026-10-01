#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def discrete_skew_metric(nodes, weights):
    """Construct the discrete beta=1 polynomial skew metric.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless abscissae, m>=1, in any supplied order.
    weights : real array, (m,)
        Nonnegative dimensionless masses. Zero masses are permitted here.

    Returns
    -------
    float64 array, (m,m)
        B such that f.T@B@g is one half of the sum of
        f(x_i)g(x_j)sign(x_j-x_i)w_i*w_j over all ordered pairs.
        The diagonal sign is zero. Supplied node order is retained.

    Raises
    ------
    ValueError
        For nonfinite/nonreal inputs, wrong shapes, repeated nodes,
        an empty domain, or negative weights.
    """
    x=_real(nodes,1); w=_real(weights,1)
    if len(x)==0 or w.shape!=x.shape or len(np.unique(x))!=len(x) or np.any(w<0):
        raise ValueError('Invalid discrete measure')
    return .5*np.sign(x[None,:]-x[:,None])*w[:,None]*w[None,:]

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def skew_project(basis, vector, metric, gauge=2):
    """Extend an ordered skew-orthogonal polynomial basis by one vector.

    Parameters
    ----------
    basis : real array, (m,k)
        Independent columns with complete consecutive pairs normalized to J.
        If k is odd, its last column is unpaired and skew-orthogonal to the
        earlier pairs. Under gauge 1 or 2 each even-indexed column has Euclidean
        norm one. Empty basis (m,0) is allowed.
    vector : real array, (m,)
        Candidate vector. The prescribed residual has a nonzero normalization.
    metric : real array, (m,m)
        Skew-symmetric bilinear metric B; no conjugation is used.
    gauge : int in {1,2,3}
        ESR1, ESR2, or ESR3m, respectively. For an unpaired candidate choose
        r11=||v||_2 (1,2) or r11=1 (3). To pair with the last column choose
        r12=last.T@v for ESR2 and zero for ESR1/ESR3m, then r22=last.T@B@v.

    Returns
    -------
    float64 array, (m+k+1,)
        Normalized new vector q, then k projection coefficients h, then scale d.
        The original vector equals basis@h+d*q. Apply two complete modified
        skew-projection passes, pairing consecutive columns. Each pass removes
        its own correction; h is their accumulated sum. Normalize only after
        both passes. The final partner uses the stated ESR gauge on each pass.
        All entries are dimensionless. Equivalent computations are accepted.

    Raises
    ------
    ValueError
        For invalid dimensions, nonfinite/nonreal values, gauge outside {1,2,3},
        a metric failing skew symmetry at atol=1e-12, or zero final scale.
        Basis normalization/independence are caller preconditions.
    """
    s=_real(basis,2);v=_real(vector,1).copy();b=_real(metric,2)
    m,k=s.shape
    if v.shape!=(m,) or b.shape!=(m,m) or gauge not in (1,2,3) or not np.allclose(b,-b.T,rtol=0,atol=1e-12):
        raise ValueError('Invalid projection contract')
    h=np.zeros(k)
    for _ in range(2):
        for j in range(0,k-1,2):
            a=-(s[:,j+1]@b@v);c=s[:,j]@b@v
            v-=a*s[:,j]+c*s[:,j+1];h[j]+=a;h[j+1]+=c
        if k%2:
            a=s[:,-1]@v if gauge==2 else 0.
            v-=a*s[:,-1];h[-1]+=a
    d=s[:,-1]@b@v if k%2 else (1. if gauge==3 else np.linalg.norm(v))
    if not np.isfinite(d) or d==0:raise ValueError('Skew breakdown')
    return np.r_[v/d,h,d]

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def symplectic_arnoldi(nodes, weights, count, gauge=2):
    """Construct sampled SOPs by the paper's symplectic Arnoldi map.

    Parameters
    ----------
    nodes, weights : real arrays, (m,)
        Distinct dimensionless nodes in supplied order, strictly positive masses.
    count : even int
        Number of polynomial columns, 2<=count<=m. Column j has degree j.
    gauge : int in {1,2,3}
        Same ESR choice as skew_project. The same gauge is used for all columns.

    Returns
    -------
    float64 array, (m,count)
        S[i,j]=p_j(nodes[i]), with S.T@B@S=J_(count/2). Start from the constant
        vector and use multiplication by nodes to extend the polynomial space.
        The two-pass skew_project convention fixes the otherwise free pair gauge.
        Nodes remain in supplied order; columns are ordered by polynomial degree.

    Raises
    ------
    ValueError
        For invalid metric inputs, nonpositive masses, invalid count/gauge,
        or a zero skew normalization during the construction.
    """
    b=discrete_skew_metric(nodes,weights);x=_real(nodes,1);w=_real(weights,1)
    if not isinstance(count,(int,np.integer)) or count<2 or count%2 or count>len(x) or np.any(w<=0) or gauge not in (1,2,3):
        raise ValueError('Invalid polynomial count or gauge')
    s=np.empty((len(x),0));v=np.ones(len(x))
    for _ in range(count):
        q=skew_project(s,v,b,gauge)[:len(x)]
        s=np.column_stack((s,q));v=x*q
    return s

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def orthogonal_ensemble_kernel(nodes, weights, basis):
    """Build the beta=1 Pfaffian correlation kernel from sampled SOPs.

    Parameters
    ----------
    nodes, weights : real arrays, (m,)
        Distinct dimensionless abscissae and nonnegative masses.
    basis : real array, (m,n)
        Even n>=2, sampled SOP columns normalized to the consecutive-pair J
        under discrete_skew_metric. Normalization is a caller precondition.

    Returns
    -------
    float64 array, (2*m,2*m)
        Direct Pfaffian kernel (not the quaternion-determinant kernel), with
        adjacent components (2*i,2*i+1) for node i, in the block orientation
        [[I(x,y),S(y,x)],[-S(x,y),-D(x,y)]] of Section 4.2.1.
        Use the discrete sign transform psi_j(x)=sum_y sign(x-y)w_y p_j(y)/2
        and the bilinear symplectic pairing of consecutive SOPs. I includes
        the bare -sign(x-y)/2 term. No quadrature factor is appended: weights
        already define the exact finite discrete ensemble. All entries are
        dimensionless; Pf(K_A) is the inclusion probability of node set A.

    Raises
    ------
    ValueError
        For invalid metric inputs or a nonfinite/nonreal basis of wrong shape
        or with a nonpositive/odd column count.
    """
    discrete_skew_metric(nodes,weights)
    x=_real(nodes,1);w=_real(weights,1);s=_real(basis,2);m=len(x)
    if s.shape[0]!=m or s.shape[1]<2 or s.shape[1]%2:raise ValueError('Invalid SOP shape')
    eps=.5*np.sign(x[None,:]-x[:,None]);j=_j(s.shape[1]//2)
    f=w[:,None]*s;psi=-eps@f
    ss=-f@j@psi.T;dd=f@j@f.T;ii=-psi@j@psi.T+eps
    out=np.empty((2*m,2*m));out[0::2,0::2]=ii;out[0::2,1::2]=ss.T
    out[1::2,0::2]=-ss;out[1::2,1::2]=-dd
    return (out-out.T)/2

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def signed_pfaffian(matrix):
    """Evaluate a signed, possibly complex Pfaffian.

    Parameters
    ----------
    matrix : real or complex array, (2*m,2*m)
        Finite skew-symmetric matrix A=-A.T, including singular and empty cases.

    Returns
    -------
    complex scalar
        Pf(A), with Pf([[0,a],[-a,0]])=a and Pf(empty)=1. This is a bilinear
        Pfaffian, not a Hermitian determinant. Preserve its sign/complex phase.

    Raises
    ------
    ValueError
        For a nonfinite matrix, nonsquare/odd shape, or failure of A=-A.T
        at absolute tolerance 1e-12 (rtol=0).
    """
    a=np.array(matrix,dtype=complex,copy=True)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or len(a)%2 or not np.isfinite(a).all() or not np.allclose(a,-a.T,atol=1e-12,rtol=0):
        raise ValueError('Invalid skew matrix')
    value=1.+0j
    for i in range(0,len(a),2):
        pidx=i+1+int(np.argmax(abs(a[i,i+1:])))
        if a[i,pidx]==0:return 0j
        if pidx!=i+1:
            a[[i+1,pidx],:]=a[[pidx,i+1],:];a[:,[i+1,pidx]]=a[:,[pidx,i+1]];value=-value
        pivot=a[i,i+1];value*=pivot
        u=a[i,i+2:].copy();v=a[i+1,i+2:].copy()
        a[i+2:,i+2:]+=(np.outer(v,u)-np.outer(u,v))/pivot
    return complex(value)

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def condition_spectral_record(kernel, record):
    """Apply sequential binary observations using conditional Pfaffian measures.

    Parameters
    ----------
    kernel : real array, (2*m,2*m)
        Valid finite Pfaffian kernel in adjacent node-component order.
    record : integer array, (r,2)
        Distinct ORIGINAL node labels and observed occupancy (0 or 1), in the
        requested elimination order. Unobserved nodes are unmeasured, not absent.
        Empty record is allowed. The joint observed event must have positive mass.

    Returns
    -------
    float64 array, (1+4*(m-r)**2,)
        First the joint probability of the record, then the conditional kernel
        on remaining original node labels in ascending order, flattened in C order.
        Use Proposition 2.5: for elimination block A subtract
        K_RA @ inv(K_A-(1-occupancy)*J_1) @ K_AR from K_RR.
        J_1=[[0,1],[-1,0]], hence inv(p*J_1)=-J_1/p. This equation fixes the
        sign even if a displayed pseudocode update uses a conflicting sign.
        Conditional probabilities within 1e-12 of [0,1] may be clipped for
        roundoff; exact allowed deterministic outcomes contribute probability 1.

    Raises
    ------
    ValueError
        For invalid dimensions, nonfinite/nonreal data, nonintegral/repeated or
        out-of-range labels, nonbinary observations, failure of skew symmetry
        at atol=1e-12, or a zero-probability requested observation. Also raise
        if any encountered conditional occupancy lies outside [-1e-12,1+1e-12].
    """
    k=_real(kernel,2).copy();r=_real(record,2)
    if k.shape[0]!=k.shape[1] or len(k)%2 or not np.allclose(k,-k.T,rtol=0,atol=1e-12) or r.shape[1]!=2:
        raise ValueError('Invalid observation dimensions')
    m=len(k)//2
    if not np.equal(r,np.round(r)).all() or len(set(r[:,0]))!=len(r) or np.any(r[:,0]<0) or np.any(r[:,0]>=m) or np.any((r[:,1]!=0)&(r[:,1]!=1)):
        raise ValueError('Invalid record')
    labels=list(range(m));prob=1.
    for label,occupied in r.astype(int):
        i=labels.index(label);a=[2*i,2*i+1];remain=[q for q in range(len(k)) if q not in a]
        p=k[a[0],a[1]]
        if p < -1e-12 or p > 1+1e-12:raise ValueError('Invalid conditional probability')
        p=float(np.clip(p,0.,1.));event=p if occupied else 1-p
        if event==0:raise ValueError('Impossible observation')
        prob*=event
        block=k[np.ix_(a,a)]-(1-occupied)*_j(1)
        k=k[np.ix_(remain,remain)]-k[np.ix_(remain,a)]@np.linalg.solve(block,k[np.ix_(a,remain)])
        k=(k-k.T)/2;labels.pop(i)
    return np.r_[prob,k.ravel()]

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def zero_one_level_probabilities(kernel):
    """Return zero- and one-level probabilities on a restricted domain.

    Parameters
    ----------
    kernel : real or complex array, (2*m,2*m)
        Valid Pfaffian correlation kernel restricted to the queried nodes,
        with adjacent components. Complex symplectic gauge representations are
        allowed when their inclusion probabilities describe a real probability law.

    Returns
    -------
    float64 array, (2,)
        [P(number of nodes occupied=0), P(number of nodes occupied=1)]. These
        are the first two coefficients of Pf(J_m+(z-1)*K). This includes cases
        where J_m-K is singular and the empty domain gives [1,0]. Equivalent
        polynomial, coefficient, or nonsingular derivative calculations are valid.
        Outputs below zero by at most 2e-12 may be clipped to zero.

    Raises
    ------
    ValueError
        For the same finite even skew-matrix violations as signed_pfaffian,
        or coefficient imaginary part above 2e-10, or real coefficients outside
        [-2e-12,1+2e-12]. Validity of the entire point process is a precondition.
    """
    k=np.asarray(kernel,dtype=complex);signed_pfaffian(k)
    m=len(k)//2
    if m==0:return np.array([1.,0.])
    count=m+1;j=_j(m)
    zs=np.exp(2j*np.pi*np.arange(count)/count)
    values=np.array([signed_pfaffian(j+(z-1)*k) for z in zs])
    coeff=np.fft.fft(values)/count
    if np.max(abs(coeff[:2].imag))>2e-10 or np.any(coeff[:2].real < -2e-12) or np.any(coeff[:2].real>1+2e-12):
        raise ValueError('Invalid probability coefficients')
    return np.clip(coeff[:2].real,0,1)

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def _pf_curve(nodes,count,record,threshold,theta,tilt):
    x=np.asarray(nodes,float);w=np.exp(-x*x/4-theta*x**4/150+tilt*x)
    s=symplectic_arnoldi(x,w,count,2)
    k=orthogonal_ensemble_kernel(x,w,s)
    packed=condition_spectral_record(k,record)
    ids=np.array([i for i in range(len(x)) if i not in set(np.asarray(record)[:,0])],int)
    kc=packed[1:].reshape(2*len(ids),2*len(ids))
    rows=np.where(x[ids]>threshold)[0];pair=np.column_stack((2*rows,2*rows+1)).ravel()
    return np.r_[packed[0],zero_one_level_probabilities(kc[np.ix_(pair,pair)])]

def infer_confinement(nodes, count, record, threshold, target, bracket, tilt=0.07):
    """Infer a quartic confinement parameter from a conditional tail probability.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes, supplied order fixes observation labels.
    count : even int, 2<=count<=m
        Fixed total number of levels before conditioning.
    record : integer array, (r,2)
        Original node labels and binary observations as in condition_spectral_record.
    threshold : finite float
        Queried unobserved nodes satisfy x>threshold (strict inequality).
    target : float in (0,1)
        Observed conditional probability of at most one level in that domain.
    bracket : real array, (2,)
        Finite increasing nonnegative parameter endpoints. The user-supplied
        equation is continuous with a unique root in this bracket, possibly
        at an endpoint. Endpoint probability residual <=1e-13 counts as a root.
    tilt : finite float
        Linear coefficient in log w=-x^2/4-theta*x^4/150+tilt*x.

    Returns
    -------
    float
        Dimensionless theta at that root, with parameter accuracy 1e-9 or better.
        Use the exact finite beta=1 ensemble and condition on the full record.
        Positive-mass observations and well-conditioned SOPs throughout the
        bracket are caller preconditions. This inverse problem is a task-specific
        application of the source method, not a fitted parameter from the paper.

    Raises
    ------
    ValueError
        For invalid scalar/bracket inputs, a root not bracketed, or any invalid
        contract propagated from symplectic_arnoldi/condition_spectral_record.
    """
    bounds=_real(bracket,1)
    if bounds.shape!=(2,) or bounds[0]<0 or bounds[0]>=bounds[1] or not np.isfinite([threshold,target,tilt]).all() or not 0<target<1:
        raise ValueError('Invalid inverse problem')
    def residual(theta):return np.sum(_pf_curve(nodes,count,record,threshold,theta,tilt)[1:])-target
    lo,hi=map(float,bounds);fl=residual(lo);fh=residual(hi)
    if abs(fl)<=1e-13:return lo
    if abs(fh)<=1e-13:return hi
    if fl*fh>0:raise ValueError('Root not bracketed')
    return float(brentq(residual,lo,hi,xtol=2e-10,rtol=1e-12))

import numpy as np
from scipy.optimize import brentq

def _real(a, ndim):
    a=np.asarray(a)
    if a.ndim!=ndim or not np.isfinite(a).all() or np.any(a.imag):
        raise ValueError('Expected a finite real array of the declared dimension')
    return a.real.astype(float)

def _j(n):
    return np.kron(np.eye(n),np.array([[0.,1.],[-1.,0.]]))

def spectral_tail_surprisal(nodes,count,record,calibration_cut,target,prediction_cut,bracket,tilt=0.07):
    """Assemble the inferred finite-ensemble joint-event surprisal.

    Parameters
    ----------
    nodes : real array, (m,)
        Distinct dimensionless nodes in supplied label order.
    count : even int, 4<=count<=m
        Fixed number of levels. The final assembly extends a count-2 SOP prefix
        by the last pair, using discrete_skew_metric and skew_project.
    record : integer array, (r,2)
        Original labels and occupancies as in condition_spectral_record.
    target : float in (0,1)
        Conditional at-most-one calibration probability.
    bracket : real array, (2,)
        Increasing nonnegative confinement bounds containing the unique root.
    tilt : finite float
        Linear coefficient of the weight, as in infer_confinement.
    calibration_cut : finite float
        Strict upper-domain boundary for inference.
    prediction_cut : finite float
        Strict upper-domain boundary for prediction, larger than calibration_cut.
        All recorded-present nodes must be <= both cuts, so the resulting event
        states that the second-largest level is <=prediction_cut.

    Returns
    -------
    float
        -ln P(record AND at most one level above prediction_cut), at the inferred
        theta; natural logarithm, dimensionless nats. Compose the preceding
        public functions on the submission path, including through their calls
        to earlier steps. This final orchestrator must use every earlier function.
        No rounding is fed back into the calculation.

    Raises
    ------
    ValueError
        For nonfinite/misordered cuts, a recorded-present node above either cut,
        a nonpositive computed joint probability, or an earlier contract violation.
    """
    if not np.isfinite([calibration_cut,prediction_cut]).all() or prediction_cut<=calibration_cut:
        raise ValueError('Invalid prediction interval')
    x=_real(nodes,1);r=_real(record,2)
    if not isinstance(count,(int,np.integer)) or count<4 or count%2:raise ValueError('At least two polynomial pairs required')
    theta=infer_confinement(x,count,r,calibration_cut,target,bracket,tilt)
    if any(x[int(i)]>calibration_cut for i,b in r if b==1):raise ValueError('Recorded level exceeds calibration cut')
    weights=np.exp(-x*x/4-theta*x**4/150+tilt*x)
    metric=discrete_skew_metric(x,weights)
    basis=symplectic_arnoldi(x,weights,count-2,2)
    for _ in range(2):
        extension=skew_project(basis,x*basis[:,-1],metric,2)[:len(x)]
        basis=np.column_stack((basis,extension))
    kernel=orthogonal_ensemble_kernel(x,weights,basis)
    packed=condition_spectral_record(kernel,r)
    labels=np.array([i for i in range(len(x)) if i not in set(r[:,0])],int)
    kc=packed[1:].reshape(2*len(labels),2*len(labels))
    rows=np.where(x[labels]>prediction_cut)[0];pair=np.column_stack((2*rows,2*rows+1)).ravel()
    restricted=kc[np.ix_(pair,pair)]
    q1=zero_one_level_probabilities(restricted)[1]
    q0=signed_pfaffian(_j(len(rows))-restricted).real
    p=packed[0]*(q0+q1)
    if p<=0:raise ValueError('Zero predicted joint probability')
    return float(-np.log(p))
SCICODE_GOLD_EOF
