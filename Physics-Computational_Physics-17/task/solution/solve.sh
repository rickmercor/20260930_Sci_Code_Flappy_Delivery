#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def contact_response(squared_separations, radius, strength):
    import numpy as np
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(strength) or strength<0:
        raise ValueError('invalid contact inputs')
    p=(2*np.pi*(radius**2+d)/3)**(-1.5)
    s=-3*strength*p/(radius**2+d)
    np.fill_diagonal(s,0)
    s-=np.diag(s.sum(axis=1))
    return s

def averaged_mobility(squared_separations, radius, self_mobility):
    import numpy as np
    from scipy.special import erf
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(self_mobility) or self_mobility<=0:
        raise ValueError('invalid mobility inputs')
    z=d/(6*radius**2)
    m=np.ones_like(z)
    mask=z>0
    m[mask]=erf(1/np.sqrt(z[mask]))-np.sqrt(z[mask]/np.pi)*(-np.expm1(-1/z[mask]))
    return self_mobility*m

def screened_response(squared_separations, radius, screening_length, couplings):
    import numpy as np
    from scipy.special import erfcx
    d=np.asarray(squared_separations,dtype=float); b=np.asarray(couplings,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or b.shape!=d.shape or not np.all(np.isfinite(d)) or not np.all(np.isfinite(b)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.allclose(np.diag(b),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(screening_length) or screening_length<=0:
        raise ValueError('invalid screening inputs')
    z=(d+radius**2)/(6*screening_length**2)
    k=-(4*np.pi*z**3)**(-0.5)+(np.pi*z)**(-0.5)-erfcx(np.sqrt(z))
    u=b*k/(3*screening_length**3)
    np.fill_diagonal(u,0)
    u-=np.diag(u.sum(axis=1))
    return u

def averaged_spatial_covariance(squared_separations, amplitude, length):
    import numpy as np
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(amplitude) or amplitude<0 or not np.isfinite(length) or length<=0:
        raise ValueError('invalid spatial covariance inputs')
    return amplitude*(1+d/(3*length**2))**(-1.5)

def drift_and_noise(mobility, spring, contact, chemical, force_covariance):
    import numpy as np
    m,k,s,u,c=[np.asarray(v,dtype=float) for v in (mobility,spring,contact,chemical,force_covariance)]
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or any(v.shape!=m.shape or not np.all(np.isfinite(v)) for v in (m,k,s,u,c)) or not np.allclose(m,m.T) or not np.allclose(c,c.T):
        raise ValueError('invalid coefficient inputs')
    return np.stack((m@(k+s+u),m@c@m))

def evolve_covariance(initial_covariance, end_time, coefficient_function):
    import numpy as np
    from scipy.integrate import solve_ivp
    x=np.asarray(initial_covariance,dtype=float)
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not np.all(np.isfinite(x)) or not np.allclose(x,x.T) or np.linalg.eigvalsh(x).min()<-1e-10 or not np.isfinite(end_time) or end_time<0 or not callable(coefficient_function):
        raise ValueError('invalid evolution inputs')
    if end_time==0: return x.copy()
    def rhs(t,y):
        state=y.reshape(x.shape)
        j,q=coefficient_function(state)
        return (j@state+state@j.T+q).ravel()
    sol=solve_ivp(rhs,(0.,end_time),x.ravel(),method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success: raise ValueError('integration failed')
    return sol.y[:,-1].reshape(x.shape)

def evolve_cross_covariance(earlier_covariance, lag, coefficient_function):
    import numpy as np
    from scipy.integrate import solve_ivp
    x=np.asarray(earlier_covariance,dtype=float)
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not np.all(np.isfinite(x)) or not np.allclose(x,x.T) or np.linalg.eigvalsh(x).min()<-1e-10 or not np.isfinite(lag) or lag<0 or not callable(coefficient_function):
        raise ValueError('invalid cross-covariance inputs')
    if lag==0: return x.copy()
    def rhs(t,y):
        same,cross=y.reshape((2,)+x.shape)
        j,q=coefficient_function(same)
        return np.stack((j@same+same@j.T+q,j@cross)).ravel()
    sol=solve_ivp(rhs,(0.,lag),np.stack((x,x)).ravel(),method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success: raise ValueError('integration failed')
    return sol.y[:,-1].reshape((2,)+x.shape)[1]

def correlation_asymmetry(activity=4.0, attraction=-3.0):
    import numpy as np
    if not np.isfinite(activity) or activity<0 or not np.isfinite(attraction):
        raise ValueError('invalid instance parameters')
    n=12
    p=np.eye(n)-np.ones((n,n))/n
    d=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    x0=-0.5*p@d@p
    k=0.5*(np.eye(n,k=1)+np.eye(n,k=-1)); k-=np.diag(k.sum(axis=1))
    c=np.eye(n); c[2,2]=activity
    b=np.zeros((n,n)); b[8,2]=attraction; b[2,8]=1.
    def coefficients(x):
        d=np.maximum(np.diag(x)[:,None]+np.diag(x)[None,:]-x-x.T,0)
        np.fill_diagonal(d,0)
        s=contact_response(d,0.5,1.)
        m=averaged_mobility(d,0.5,1.)
        u=screened_response(d,0.5,2.,b)
        coefficients=drift_and_noise(m,k,s,u,c)
        coefficients[1]+=averaged_spatial_covariance(d,4.,1.5)
        return coefficients
    x1=evolve_covariance(x0,1.,coefficients)
    y=evolve_cross_covariance(x1,1.,coefficients)
    u=np.eye(n)[2]-np.eye(n)[1]; v=np.eye(n)[8]-np.eye(n)[7]
    return float(u@y@v-v@y@u)
SCICODE_GOLD_EOF
