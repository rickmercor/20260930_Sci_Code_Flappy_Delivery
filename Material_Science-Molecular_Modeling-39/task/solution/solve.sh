#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def field_matrices(z: float, field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np
    from math import erf as _erf_scalar
    erf = np.vectorize(_erf_scalar, otypes=[float])

    for _name, _value in (('z', z), ('field', field), ('positions', positions), ('widths', widths), ('parameters', parameters), ('curvature', curvature),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)
    z = _scalar(z, 'z')
    field = _array(field, (3,), 'field')
    try:
        positions = np.asarray(positions, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('positions') from exc
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions)<1 or not np.isfinite(positions).all():
        raise ValueError('positions')
    n = len(positions)
    widths = _array(widths, (n,), 'widths')
    p = _array(parameters, (n,9), 'parameters')
    curvature = _array(curvature, (2,3,3), 'curvature')
    if np.any(widths<=0) or not np.allclose(curvature, curvature.swapaxes(1,2), atol=1e-12, rtol=0):
        raise ValueError('widths or curvature')
    v = positions-positions.mean(axis=0)
    d = v@field
    chi,h,s,u,t,w,eta,l,k = p.T
    distances = np.linalg.norm(positions[:,None,:]-positions[None,:,:], axis=2)
    spread = np.sqrt(2*(widths[:,None]**2+widths[None,:]**2))
    G = np.empty((n,n))
    np.divide(erf(distances/spread), distances, out=G, where=distances!=0)
    G[distances==0] = (2/np.sqrt(np.pi)/spread)[distances==0]
    A = G+np.diag(h+eta*z+l*d+0.5*k*d*d)
    A1 = np.zeros((3,n,n)); A2 = np.zeros((3,3,n,n))
    ids = np.arange(n)
    A1[:,ids,ids] = ((l+k*d)[:,None]*v).T
    vv = np.einsum('ia,ib->abi',v,v)
    A2[:,:,ids,ids] = vv*k
    b = chi+s*z+(u+w*z)*d+0.5*t*d*d
    b1 = (((u+w*z)+t*d)[:,None]*v).T
    b2 = vv*t
    return A,A1,A2,b,b1,b2,-curvature[0]-z*curvature[1]

def constrained_response(A: 'np.ndarray', A1: 'np.ndarray', b: 'np.ndarray', b1: 'np.ndarray', total_charge: float) -> 'tuple[np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('A', A), ('A1', A1), ('b', b), ('b1', b1), ('total_charge', total_charge),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)
    try:
        A = np.asarray(A, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('A') from exc
    if A.ndim!=2 or A.shape[0]<1 or A.shape[0]!=A.shape[1]:
        raise ValueError('A')
    n=len(A)
    A=_array(A,(n,n),'A'); A1=_array(A1,(3,n,n),'A1')
    b=_array(b,(n,),'b'); b1=_array(b1,(3,n),'b1')
    Q=_scalar(total_charge,'total_charge')
    if not np.allclose(A,A.T,atol=1e-12,rtol=0) or not np.allclose(A1,A1.swapaxes(1,2),atol=1e-12,rtol=0):
        raise ValueError('symmetry')
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise ValueError('A must be positive definite') from exc
    K=np.zeros((n+1,n+1)); K[:n,:n]=A; K[n,:n]=1; K[:n,n]=1
    q=np.linalg.solve(K,np.r_[-b,Q])[:n]
    rhs=np.zeros((n+1,3)); rhs[:n]=-(np.einsum('aij,j->ai',A1,q)+b1).T
    dq=np.linalg.solve(K,rhs)[:n].T
    return q,dq

def relaxed_polarizability(A1: 'np.ndarray', A2: 'np.ndarray', b1: 'np.ndarray', b2: 'np.ndarray', c2: 'np.ndarray', q: 'np.ndarray', dq: 'np.ndarray') -> 'np.ndarray':
    import numpy as np

    for _name, _value in (('A1', A1), ('A2', A2), ('b1', b1), ('b2', b2), ('c2', c2), ('q', q), ('dq', dq),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    try:
        q=np.asarray(q,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('q') from exc
    if q.ndim!=1 or len(q)<1:
        raise ValueError('q')
    n=len(q); q=_array(q,(n,),'q')
    A1=_array(A1,(3,n,n),'A1'); A2=_array(A2,(3,3,n,n),'A2')
    b1=_array(b1,(3,n),'b1'); b2=_array(b2,(3,3,n),'b2')
    c2=_array(c2,(3,3),'c2'); dq=_array(dq,(3,n),'dq')
    tangent=b1+np.einsum('aij,j->ai',A1,q)
    H=c2+np.einsum('abi,i->ab',b2,q)+0.5*np.einsum('i,abij,j->ab',q,A2,q)+tangent@dq.T
    return -0.5*(H+H.T)

def projected_spectrum(alpha: 'np.ndarray', projector: 'np.ndarray', time_step: float) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('alpha', alpha), ('projector', projector), ('time_step', time_step),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)
    try:
        alpha=np.asarray(alpha,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('alpha') from exc
    if alpha.ndim!=4 or alpha.shape[2:]!=(3,3) or alpha.shape[0]<1 or alpha.shape[1]<4 or not np.isfinite(alpha).all():
        raise ValueError('alpha')
    projector=_array(projector,(3,3),'projector'); h=_scalar(time_step,'time_step')
    if h<=0: raise ValueError('time_step')
    M,N=alpha.shape[:2]
    x=np.einsum('mnab,ab->mn',alpha,projector)
    x=x-x.mean(axis=1,keepdims=True)
    X=np.fft.rfft(x,axis=1)[:,1:]
    I=h*np.mean(abs(X)**2,axis=0)/(2*N)
    f=np.arange(1,N//2+1)/(N*h)
    nu=np.full(len(f),2*M,dtype=float)
    if N%2==0:nu[-1]=M
    return f,I,nu

def pade_likelihood(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', weights: 'np.ndarray') -> 'tuple[float, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('p', p), ('frequency', frequency), ('observed', observed), ('dof', dof), ('weights', weights),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x


    def _spectrum(f, observed, dof):
        try:
            f = np.asarray(f, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('frequency') from exc
        if f.ndim != 1 or len(f)<4 or not np.isfinite(f).all() or np.any(f<0) or np.any(np.diff(f)<=0):
            raise ValueError('frequency')
        y = _array(observed, f.shape, 'observed')
        nu = _array(dof, f.shape, 'dof')
        if np.any(y<=0) or np.any(nu<=0):
            raise ValueError('observed and dof must be positive')
        return f,y,nu

    def _pade_jet(p, f):
        x = f*f
        num = p[0]+p[1]*x
        den = 1+p[2]*x
        if np.any(num<=0) or np.any(den<=0):
            raise ValueError('nonpositive Pade numerator or denominator')
        m = num/den
        J = np.stack((1/den,x/den,-num*x/den**2),axis=1)
        D = np.zeros((len(f),3,3))
        D[:,0,2]=D[:,2,0]=-x/den**2
        D[:,1,2]=D[:,2,1]=-x*x/den**2
        D[:,2,2]=2*num*x*x/den**3
        return m,J,D
    f,y,nu=_spectrum(frequency,observed,dof)
    p=_array(p,(3,),'p'); w=_array(weights,f.shape,'weights')
    if np.any(w<0) or not np.any(w>0):raise ValueError('weights')
    m,J,D=_pade_jet(p,f)
    a=w*nu/2
    first=a*(m-y)/m**2
    second=a*(2*y-m)/m**3
    value=np.sum(a*(np.log(m)+y/m))
    gradient=J.T@first
    hessian=J.T@(second[:,None]*J)+np.einsum('k,kab->ab',first,D)
    return float(value),gradient,hessian

def fit_cutoff(frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', cutoff: float, initial: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('frequency', frequency), ('observed', observed), ('dof', dof), ('cutoff', cutoff), ('initial', initial),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)

    def _spectrum(f, observed, dof):
        try:
            f = np.asarray(f, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('frequency') from exc
        if f.ndim != 1 or len(f)<4 or not np.isfinite(f).all() or np.any(f<0) or np.any(np.diff(f)<=0):
            raise ValueError('frequency')
        y = _array(observed, f.shape, 'observed')
        nu = _array(dof, f.shape, 'dof')
        if np.any(y<=0) or np.any(nu<=0):
            raise ValueError('observed and dof must be positive')
        return f,y,nu

    def _pade_jet(p, f):
        x = f*f
        num = p[0]+p[1]*x
        den = 1+p[2]*x
        if np.any(num<=0) or np.any(den<=0):
            raise ValueError('nonpositive Pade numerator or denominator')
        m = num/den
        J = np.stack((1/den,x/den,-num*x/den**2),axis=1)
        D = np.zeros((len(f),3,3))
        D[:,0,2]=D[:,2,0]=-x/den**2
        D[:,1,2]=D[:,2,1]=-x*x/den**2
        D[:,2,2]=2*num*x*x/den**3
        return m,J,D
    f,y,nu=_spectrum(frequency,observed,dof)
    c=_scalar(cutoff,'cutoff'); initial=_array(initial,(3,),'initial')
    if c<=0:raise ValueError('cutoff')
    _pade_jet(initial,f)
    with np.errstate(over='ignore'):
        w=1/(1+(f/c)**8)
    w[w<0.001]=0
    if np.count_nonzero(w)<4:raise ValueError('too few frequencies')
    scale=float(np.sum(w*y)/np.sum(w))
    transform=np.array([scale,scale/c**2,1/c**2])
    x=f/c; Y=y/scale; p=initial/transform
    for iteration in range(250):
        value,g,H=pade_likelihood(p,x,Y,nu,w)
        ev=np.linalg.eigvalsh(H)
        shift=max(0.0,1e-7*np.max(abs(ev))-ev[0])
        Hstep=H+shift*np.eye(3)
        d=-np.linalg.solve(Hstep,g)
        descent=float(g@d)
        if ev[0]>0 and -descent<1e-19*(1+np.sum(w*nu)):
            break
        rate=1.0
        accepted=False
        for trial in range(70):
            candidate=p+rate*d
            try:
                newvalue,_,_=pade_likelihood(candidate,x,Y,nu,w)
            except ValueError:
                newvalue=np.inf
            # The slack only permits a final Newton correction at roundoff.
            if np.isfinite(newvalue) and newvalue<=value+1e-4*rate*descent+1e-12*(1+abs(value)):
                p=candidate; accepted=True; break
            rate*=0.5
        if not accepted:raise ValueError('line search did not converge')
    else:
        raise ValueError('optimizer did not converge')
    _,g,H=pade_likelihood(p,x,Y,nu,w)
    try:
        np.linalg.cholesky(H)
        C=np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        raise ValueError('nonpositive Hessian') from exc
    return p*transform,C*transform[:,None]*transform[None,:]

def cv2l_score(p: 'np.ndarray', frequency: 'np.ndarray', observed: 'np.ndarray', dof: 'np.ndarray', cutoff: float) -> 'tuple[float, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('p', p), ('frequency', frequency), ('observed', observed), ('dof', dof), ('cutoff', cutoff),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    def _scalar(x, name):
        try:
            a = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if a.shape != () or not np.isfinite(a):
            raise ValueError(name)
        return float(a)

    def _spectrum(f, observed, dof):
        try:
            f = np.asarray(f, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('frequency') from exc
        if f.ndim != 1 or len(f)<4 or not np.isfinite(f).all() or np.any(f<0) or np.any(np.diff(f)<=0):
            raise ValueError('frequency')
        y = _array(observed, f.shape, 'observed')
        nu = _array(dof, f.shape, 'dof')
        if np.any(y<=0) or np.any(nu<=0):
            raise ValueError('observed and dof must be positive')
        return f,y,nu

    def _pade_jet(p, f):
        x = f*f
        num = p[0]+p[1]*x
        den = 1+p[2]*x
        if np.any(num<=0) or np.any(den<=0):
            raise ValueError('nonpositive Pade numerator or denominator')
        m = num/den
        J = np.stack((1/den,x/den,-num*x/den**2),axis=1)
        D = np.zeros((len(f),3,3))
        D[:,0,2]=D[:,2,0]=-x/den**2
        D[:,1,2]=D[:,2,1]=-x*x/den**2
        D[:,2,2]=2*num*x*x/den**3
        return m,J,D
    f,y,nu=_spectrum(frequency,observed,dof)
    p=_array(p,(3,),'p'); c=_scalar(cutoff,'cutoff')
    if c<=0:raise ValueError('cutoff')
    m,J,_=_pade_jet(p,f)
    # Column scaling preserves the specified parameterization of the result.
    scales=np.array([p[0],p[0]/c**2,1/c**2])
    JS=J*scales
    V=2*m*m/nu
    with np.errstate(over='ignore'):
        w1=1/(1+(f/(1.25*c/2))**8)
        w2=1/(1+(f/(1.25*c))**8)-w1
    w1[w1<0.001]=0; w2[w2<0.001]=0
    matrices=[]
    try:
        for w in (w1,w2):
            rhs=JS.T*(w/V)
            gram=rhs@JS
            np.linalg.cholesky(gram)
            matrices.append(np.linalg.solve(gram,rhs))
        D=matrices[0]-matrices[1]
        d=D@(y-m); Cd=(D*V)@D.T
        np.linalg.cholesky(Cd)
        d=d*scales; Cd=Cd*scales[:,None]*scales[None,:]
        sign,logdet=np.linalg.slogdet(Cd)
        if sign<=0:raise np.linalg.LinAlgError('covariance')
        criterion=0.5*(3*np.log(2*np.pi)+logdet+d@np.linalg.solve(Cd,d))
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular or indefinite covariance') from exc
    return float(criterion),d,Cd

def lorentz_marginal(parameters: 'np.ndarray', covariance: 'np.ndarray', criteria: 'np.ndarray') -> 'tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    import numpy as np

    for _name, _value in (('parameters', parameters), ('covariance', covariance), ('criteria', criteria),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")

    def _array(x, shape, name):
        try:
            x = np.asarray(x, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(name) from exc
        if x.shape != shape or not np.all(np.isfinite(x)):
            raise ValueError(name)
        return x

    try:
        p=np.asarray(parameters,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('parameters') from exc
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():
        raise ValueError('parameters')
    J=len(p); C=_array(covariance,(J,3,3),'covariance'); crit=_array(criteria,(J,),'criteria')
    if not np.allclose(C,C.swapaxes(1,2),atol=1e-12,rtol=0):raise ValueError('covariance symmetry')
    try:np.linalg.cholesky(C)
    except np.linalg.LinAlgError as exc:raise ValueError('covariance') from exc
    values=np.zeros((J,4)); cov=np.zeros((J,4,4)); ratios=np.zeros(J); valid=np.zeros(J,dtype=int)
    for j,(a,b,c) in enumerate(p):
        if a<=0 or c<=0 or a*c<=b:continue
        tau=np.sqrt(c)/(2*np.pi); C0=b/c; C1=np.pi*(a-C0)/np.sqrt(c)
        T=np.zeros((4,3)); T[0,0]=1; T[1,2]=1/(4*np.pi*np.sqrt(c))
        T[2,1]=1/c; T[2,2]=-b/c**2
        T[3]=[np.pi/np.sqrt(c),-np.pi/c**1.5,np.pi*(-0.5*a/c**1.5+1.5*b/c**2.5)]
        values[j]=[a,tau,C0,C1]; cov[j]=T@C[j]@T.T
        ratios[j]=(np.sqrt(cov[j,1,1])/tau)/(np.sqrt(cov[j,0,0])/a)
        valid[j]=int(ratios[j]<=100)
    ids=np.flatnonzero(valid)
    if len(ids)==0:raise ValueError('no admissible Lorentz fit')
    logs=-crit[ids]-ratios[ids]; w=np.zeros(J)
    w[ids]=np.exp(logs-logs.max()); w/=w.sum()
    mean=np.sum(w[:,None]*values,axis=0)
    centered=values-mean
    mixture=np.einsum('j,jab->ab',w,cov)+np.einsum('j,ja,jb->ab',w,centered,centered)
    return mean,mixture,w,ratios,valid

def polarizability_integral(states: 'np.ndarray', field: 'np.ndarray', positions: 'np.ndarray', widths: 'np.ndarray', parameters: 'np.ndarray', curvature: 'np.ndarray', total_charge: float, projector: 'np.ndarray', time_step: float, cutoffs: 'np.ndarray', initial: 'np.ndarray') -> float:
    import numpy as np

    for _name, _value in (('states', states), ('field', field), ('positions', positions), ('widths', widths), ('parameters', parameters), ('curvature', curvature), ('total_charge', total_charge), ('projector', projector), ('time_step', time_step), ('cutoffs', cutoffs), ('initial', initial),):
        if np.iscomplexobj(_value):
            raise ValueError(_name + " must be real")


    try:
        states=np.asarray(states,dtype=float); cutoffs=np.asarray(cutoffs,dtype=float)
    except (ValueError,TypeError,OverflowError) as exc:
        raise ValueError('states or cutoffs') from exc
    if states.ndim!=2 or states.shape[0]<1 or states.shape[1]<8 or not np.isfinite(states).all():raise ValueError('states')
    if cutoffs.ndim!=1 or len(cutoffs)<1 or not np.isfinite(cutoffs).all() or np.any(cutoffs<=0) or np.any(np.diff(cutoffs)<=0):raise ValueError('cutoffs')
    alpha=np.empty(states.shape+(3,3))
    for m in range(states.shape[0]):
        for t in range(states.shape[1]):
            A,A1,A2,b,b1,b2,c2=field_matrices(states[m,t],field,positions,widths,parameters,curvature)
            q,dq=constrained_response(A,A1,b,b1,total_charge)
            alpha[m,t]=relaxed_polarizability(A1,A2,b1,b2,c2,q,dq)
    f,y,nu=projected_spectrum(alpha,projector,time_step)
    rows=[]; cov=[]; scores=[]
    for cutoff in cutoffs:
        p,C=fit_cutoff(f,y,nu,cutoff,initial)
        score,_,_=cv2l_score(p,f,y,nu,cutoff)
        rows.append(p);cov.append(C);scores.append(score)
    mean,_,_,_,_=lorentz_marginal(rows,cov,scores)
    return float(mean[0])
SCICODE_GOLD_EOF
