#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def surface_geometry(points: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple') -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,dtype=float);a=np.asarray(axes,dtype=float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or a.shape!=(3,):
        raise ValueError('invalid geometry shapes')
    if not np.isfinite(x).all() or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('invalid geometry values')
    if np.max(abs(np.sum((x/a)**2,axis=1)-1))>1e-8:
        raise ValueError('points must lie on ellipsoid')
    g=x/a**2;n=g/np.linalg.norm(g,axis=1)[:,None]
    out=np.empty((len(x),4));out[:,:3]=n
    for i in range(len(x)):
        p=np.eye(3)-np.outer(n[i],n[i])
        shape=p@np.diag(1/a**2)@p/np.linalg.norm(g[i])
        eig=np.linalg.eigvalsh(shape)
        out[i,3]=np.linalg.norm(eig[1:])
    return out

def characteristic_lengths(points: 'np.ndarray | list | tuple', curvature: 'np.ndarray | list | tuple', h0: float, tau: float, kref: float, radius: float) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);c=np.asarray(curvature,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or c.shape!=(len(x),):
        raise ValueError('invalid length-field shapes')
    if not np.isfinite(x).all() or not np.isfinite(c).all() or np.any(c<0):
        raise ValueError('invalid length-field data')
    if not np.isfinite([h0,tau,kref,radius]).all() or h0<=0 or radius<=0 or tau<0 or kref<0:
        raise ValueError('invalid length parameters')
    raw=h0/np.sqrt(1+tau*abs(c-kref))
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    h=np.min(np.where(d<=radius,raw[None,:],np.inf),axis=1)
    return np.column_stack((raw,h))

def occupied_support(points: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', packing: float=0.933) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);h=np.asarray(lengths,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or h.shape!=(len(x),):
        raise ValueError('invalid support shapes')
    if not np.isfinite(x).all() or not np.isfinite(h).all() or np.any(h<=0) or not np.isfinite(packing) or packing<=0:
        raise ValueError('invalid support values')
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    if np.any(d[np.triu_indices(len(x),1)]<1e-12):
        raise ValueError('distinct coincident samples')
    r=2*h
    kernel=3/(np.pi*r[:,None]**2)*np.maximum(1-d/r[:,None],0)
    rho=np.sum(kernel*(h[None,:]/h[:,None])**2,axis=1)
    return np.column_stack((1/rho,packing*h*h*rho))

def event_proposal(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', normals: 'np.ndarray | list | tuple', supports: 'np.ndarray | list | tuple', target: int, lower: float, upper: float, mu: 'np.ndarray | list | tuple') -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);labels=np.asarray(ids);h=np.asarray(lengths,float)
    n=np.asarray(normals,float);s=np.asarray(supports,float);m=np.asarray(mu,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<2 or n.shape!=x.shape:
        raise ValueError('invalid proposal point shapes')
    if labels.shape!=(len(x),) or h.shape!=(len(x),) or s.shape!=(len(x),) or m.shape!=(3,):
        raise ValueError('invalid proposal arrays')
    if not all(np.isfinite(z).all() for z in (x,labels,h,n,s,m)):
        raise ValueError('nonfinite proposal data')
    if np.any(labels<0) or np.any(labels!=np.floor(labels)) or len(np.unique(labels))!=len(labels) or target not in labels:
        raise ValueError('invalid identifiers')
    if np.any(h<=0) or np.any(s<0) or np.any(abs(m)>.5) or np.any(abs(np.linalg.norm(n,axis=1)-1)>1e-8):
        raise ValueError('invalid proposal fields')
    if not np.isfinite([lower,upper]).all() or lower<0 or upper<=lower:
        raise ValueError('invalid thresholds')
    dvec=x[:,None,:]-x[None,:,:];d=np.linalg.norm(dvec,axis=2)
    if np.any(d[np.triu_indices(len(x),1)]<1e-12):
        raise ValueError('coincident samples')
    k=int(np.flatnonzero(labels==target)[0])
    if s[k]>upper:
        distance=d[k].copy();distance[k]=np.inf
        j=int(np.lexsort((labels,distance))[0])
        return np.r_[-1.,float(labels[j]),(x[k]+x[j])/2]
    if s[k]<lower:
        r=2*h[k]
        unit=np.divide(dvec[k],d[k,:,None],out=np.zeros_like(x),where=d[k,:,None]>0)
        weights=(h/h[k])**2
        grad=np.sum((-3/(np.pi*r**3))*weights[:,None]*unit*(d[k]<r)[:,None],axis=0)
        grad-=n[k]*np.dot(n[k],grad)
        norm=np.linalg.norm(grad)
        if norm<=1e-10:
            raise ValueError('active birth requires nonzero tangent gradient')
        return np.r_[1.,-1.,x[k]-h[k]*(1+m)*grad/norm]
    return np.r_[0.,-1.,x[k]]

def surface_return(trial: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', tolerance: float = 1e-13) -> 'np.ndarray':
    import numpy as np
    from scipy.optimize import brentq
    y = np.asarray(trial, dtype=float)
    a = np.asarray(axes, dtype=float)
    if (y.shape != (3,) or a.shape != (3,) or not np.isfinite(y).all()
            or not np.isfinite(a).all() or np.any(a <= 0)):
        raise ValueError('invalid projection data')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid projection tolerance')
    b = a*a
    residual0 = float(np.sum((y/a)**2)-1)
    if abs(residual0) <= 2e-15:
        return y.copy()
    options = {'xtol': np.nextafter(0., 1.), 'rtol': 4*np.finfo(float).eps, 'maxiter': 1000}
    if residual0 > 0:
        def _residual(lam):
            return float(np.sum((a*y/(b+lam))**2)-1)
        upper = max(1., float(np.linalg.norm(y)*max(a)))
        while _residual(upper) > 0:
            upper *= 2
        lam = brentq(_residual, 0., upper, **options)
        return b*y/(b+lam)

    # The shifted multiplier avoids subtracting nearly equal values at a pole.
    minimum = float(min(b))
    short = b == minimum
    delta = b-minimum
    if np.all(y[short] == 0):
        point = np.zeros(3)
        point[~short] = b[~short]*y[~short]/delta[~short]
        filled = float(np.sum((point[~short]/a[~short])**2))
        if filled <= 1:
            first = int(np.flatnonzero(short)[0])
            point[first] = a[first]*np.sqrt(max(0., 1-filled))
            return point
        lower = 0.
    else:
        lower = float(np.sqrt(minimum)*np.linalg.norm(y[short]))

    def _residual_shifted(shift):
        denominator = delta+shift
        active = denominator != 0
        return float(np.sum((a[active]*y[active]/denominator[active])**2)-1)

    shift = lower if _residual_shifted(lower) <= 0 else brentq(_residual_shifted, lower, minimum, **options)
    return b*y/(delta+shift)

def inspected_state(points: 'np.ndarray | list | tuple', ids: 'np.ndarray | list | tuple', axes: 'np.ndarray | list | tuple', h0: float, tau: float, radius: float, lower: float, upper: float, mu: 'np.ndarray | list | tuple', target: int, new_id: int, kref: float=0.0, packing: float=0.933) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);labels=np.asarray(ids)
    if not np.isscalar(new_id) or not np.isfinite(new_id) or new_id<0 or new_id!=int(new_id) or new_id in labels:
        raise ValueError('invalid new identifier')
    geo=surface_geometry(x,axes)
    lengths=characteristic_lengths(x,geo[:,3],h0,tau,kref,radius)[:,1]
    support=occupied_support(x,lengths,packing)[:,1]
    proposal=event_proposal(x,labels,lengths,geo[:,:3],support,target,lower,upper,mu)
    kind=int(proposal[0])
    if kind==0:return np.column_stack((labels,x))
    point=surface_return(proposal[2:],axes)
    if kind==-1:
        keep=(labels!=target)&(labels!=int(proposal[1]))
        x=x[keep];labels=labels[keep]
    return np.column_stack((np.r_[labels,new_id],np.vstack((x,point))))

def adapted_moment(tau: float=0.5, radius: float=0.55, lower: float=0.7, upper: float=1.25, ring_radius: float=0.12) -> float:
    import numpy as np
    if not np.isfinite([tau,radius,lower,upper,ring_radius]).all() or tau<0 or radius<=0 or ring_radius<=0 or lower<0 or upper<=lower:
        raise ValueError('invalid instance parameters')
    axes=np.array([1.,.8,.6]);ids=np.arange(48)
    t=1-2*(ids+.5)/48;phi=ids*np.pi*(3-np.sqrt(5))
    x=np.column_stack((np.sqrt(1-t*t)*np.cos(phi),np.sqrt(1-t*t)*np.sin(phi),t))*axes
    keep=~((x[:,0]>.6)&(x[:,2]>.1));x=x[keep];ids=ids[keep]
    index=int(np.flatnonzero(ids==34)[0]);center=x[index]
    normal=surface_geometry(x,axes)[index,:3]
    e=np.cross(normal,[0.,0.,1.]);e/=np.linalg.norm(e);f=np.cross(normal,e)
    ring=[]
    for k in range(5):
        trial=center+ring_radius*(np.cos(2*np.pi*k/5)*e+np.sin(2*np.pi*k/5)*f)
        ring.append(surface_return(trial,axes))
    x=np.vstack((x,ring));ids=np.r_[ids,np.arange(100,105)]
    for target,new_id in [(26,200),(34,201),(26,202)]:
        # Redundant API consistency check; this adds no scientific novelty credit.
        geo=surface_geometry(x,axes)
        lengths=characteristic_lengths(x,geo[:,3],.42,tau,0.,radius)[:,1]
        supports=occupied_support(x,lengths,.933)[:,1]
        proposal=event_proposal(x,ids,lengths,geo[:,:3],supports,target,lower,upper,[.13,-.21,.07])
        expected_count=len(x)+int(proposal[0])
        packed=inspected_state(x,ids,axes,.42,tau,radius,lower,upper,[.13,-.21,.07],target,new_id)
        if len(packed)!=expected_count:
            raise ValueError('proposal and state update disagree')
        ids=packed[:,0].astype(int);x=packed[:,1:]
    geo=surface_geometry(x,axes)
    h=characteristic_lengths(x,geo[:,3],.42,tau,0.,radius)[:,1]
    areas=occupied_support(x,h,.933)[:,0]
    return float(np.dot(areas,x[:,0]**2)/np.sum(areas))
SCICODE_GOLD_EOF
