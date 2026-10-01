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
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def collision_coefficients(hamiltonian,bath):
    try:
        h=_hamiltonian(hamiltonian);xi=_density(bath,'bath',dimension=2);d=h.shape[0]//2
        def coefficient(e,k):
            b=np.kron(e,xi);c=h@b-b@h
            return _ptr(-1j*c if k==1 else -(h@c-c@h)/2,d)
        return np.stack([_super(lambda e:coefficient(e,1),d),_super(lambda e:coefficient(e,2),d)])
    except ValueError as exc:
        raise ValueError('collision_coefficients: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _pow(a,p):
    v,u=np.linalg.eigh((a+a.conj().T)/2)
    return (u*v**p)@u.conj().T

def normalizer_coefficients(prior,collision):
    try:
        g=_density(prior,'prior',positive_definite=True);d=len(g);c=_finite_array(collision,'collision',(2,d*d,d*d))
        g1=(c[0]@g.reshape(-1,order='F')).reshape(d,d,order='F')
        g2=(c[1]@g.reshape(-1,order='F')).reshape(d,d,order='F')
        root=_pow(g,.5);w0=_pow(g,-.5)
        s1=solve_sylvester(root,root,g1)
        s2=solve_sylvester(root,root,g2-s1@s1)
        w1=-w0@s1@w0
        w2=w0@s1@w0@s1@w0-w0@s2@w0
        return np.stack([w0,w1,w2])
    except ValueError as exc:
        raise ValueError('normalizer_coefficients: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _pow(a,p):
    v,u=np.linalg.eigh((a+a.conj().T)/2)
    return (u*v**p)@u.conj().T

def petz_coefficients(prior,collision,normalizer):
    try:
        g=_density(prior,'prior',positive_definite=True);d=len(g)
        c=_finite_array(collision,'collision',(2,d*d,d*d));w=_finite_array(normalizer,'normalizer',(3,d,d))
        r=_pow(g,.5);lift=np.kron(r.conj(),r)
        b0=np.kron(w[0].T,w[0])
        b1=np.kron(w[1].T,w[0])+np.kron(w[0].T,w[1])
        b2=np.kron(w[2].T,w[0])+np.kron(w[1].T,w[1])+np.kron(w[0].T,w[2])
        return np.stack([lift@(b1+c[0].conj().T@b0),lift@(b2+c[0].conj().T@b1+c[1].conj().T@b0)])
    except ValueError as exc:
        raise ValueError('petz_coefficients: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _paulis():
    return np.array([np.eye(2), [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],complex)

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def reverse_coefficients(hamiltonian):
    try:
        h=_hamiltonian(hamiltonian);d=len(h)//2;basis=_paulis()/2
        out=np.empty((2,4,d*d,d*d),complex)
        for a,b in enumerate(basis):
            def f(e,k):
                x=np.kron(e,b);c=h@x-x@h
                return _ptr(1j*c if k==1 else -(h@c-c@h)/2,d)
            out[0,a]=_super(lambda e:f(e,1),d)
            out[1,a]=_super(lambda e:f(e,2),d)
        return out
    except ValueError as exc:
        raise ValueError('reverse_coefficients: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _forward(h,xi,t):
    d=h.shape[0]//2;u=expm(-1j*t*h)
    return _super(lambda e:_ptr(u@np.kron(e,xi)@u.conj().T,d),d)

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _sequence_inputs(hamiltonians, bath, initial_prior, times):
    g = _density(initial_prior, 'initial_prior', positive_definite=True)
    xi = _density(bath, 'bath', dimension=2)
    hs = _finite_array(hamiltonians, 'hamiltonians')
    if hs.ndim != 3 or len(hs) < 1 or hs.shape[1:] != (2*len(g), 2*len(g)):
        raise ValueError('hamiltonians has an invalid shape')
    for h in hs:
        _hamiltonian(h, len(g))
    ts = _finite_array(times, 'times', (len(hs),), real=True)
    if np.any(ts < 0):
        raise ValueError('times must be nonnegative')
    return hs, xi, g, ts

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def prior_trajectory(hamiltonians,bath,initial_prior,times):
    try:
        hs,bath,g,ts=_sequence_inputs(hamiltonians,bath,initial_prior,times);d=len(g)
        out=[g.copy()]
        for h,t in zip(hs,ts):
            g=(_forward(h,bath,t)@g.reshape(-1,order='F')).reshape(d,d,order='F')
            g=(g+g.conj().T)/2
            _density(g,'propagated prior',positive_definite=True)
            out.append(g)
        return np.array(out)
    except ValueError as exc:
        raise ValueError('prior_trajectory: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _bloch(value, name):
    a = _finite_array(value, name, (3,), real=True)
    if np.linalg.norm(a) > 1 + 1e-10:
        raise ValueError(name + ' must have norm at most one')
    return a

def _channel_dimension(a):
    d = int(round(np.sqrt(a.shape[-1])))
    if d < 1 or a.shape[-2:] != (d*d, d*d):
        raise ValueError('superoperator dimensions must be equal perfect squares')
    return d

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _real_scalar(value, name, positive=False, nonnegative=False):
    a = _finite_array(value, name, (), real=True)
    x = float(a)
    if (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(name + ' violates its sign domain')
    return x

def shared_reset_objective(petz,reverse,times,penalty,nominal):
    try:
        p=_finite_array(petz,'petz')
        if p.ndim!=4 or len(p)<1 or p.shape[1]!=2:raise ValueError('petz has an invalid shape')
        d=_channel_dimension(p);n=len(p)
        a=_finite_array(reverse,'reverse',(n,2,4,d*d,d*d))
        ts=_finite_array(times,'times',(n,),real=True)
        if np.any(ts<0):raise ValueError('times must be nonnegative')
        penalty=_real_scalar(penalty,'penalty',positive=True);nominal=_bloch(nominal,'nominal')
        n=len(ts);q=float(penalty)*np.eye(3);b=float(penalty)*np.asarray(nominal,float);c=float(penalty)*float(np.dot(nominal,nominal))
        for k,t in enumerate(ts):
            target=(p[k,0]+t*p[k,1]-a[k,0,0]-t*a[k,1,0]).reshape(-1)
            design=np.stack([(a[k,0,j]+t*a[k,1,j]).reshape(-1) for j in range(1,4)],axis=1)
            q+=(design.conj().T@design).real/n
            b+=(design.conj().T@target).real/n
            c+=float(np.vdot(target,target).real)/n
        out=np.zeros((4,4),float);out[:3,:3]=q;out[:3,3]=out[3,:3]=b;out[3,3]=c
        return out
    except ValueError as exc:
        raise ValueError('shared_reset_objective: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _bounds(radius, z_floor):
    r = _real_scalar(radius, 'radius', positive=True)
    z = _real_scalar(z_floor, 'z_floor')
    if r > 1 or not -r <= z <= r:
        raise ValueError('Require 0 < radius <= 1 and -radius <= z_floor <= radius')
    return r, z

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _real_scalar(value, name, positive=False, nonnegative=False):
    a = _finite_array(value, name, (), real=True)
    x = float(a)
    if (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(name + ' violates its sign domain')
    return x

def physical_reset(objective,radius,z_floor):
    try:
        m=_finite_array(objective,'objective',(4,4),real=True)
        _hermitian(m,'objective')
        if np.linalg.eigvalsh(m[:3,:3])[0]<=0:raise ValueError('Q must be positive definite')
        r,z=_bounds(radius,z_floor)
        q=m[:3,:3];b=m[:3,3]
        def ball(q,b,r):
            if r==0:return np.zeros(len(b))
            s=np.linalg.solve(q,b)
            if np.linalg.norm(s)<=r:return s
            def f(mu):return np.linalg.norm(np.linalg.solve(q+mu*np.eye(len(b)),b))-r
            hi=1.
            while f(hi)>0:hi*=2
            mu=brentq(f,0,hi,xtol=5e-15,rtol=5e-15)
            return np.linalg.solve(q+mu*np.eye(len(b)),b)
        s=ball(q,b,r)
        if s[2]<z:
            xy=ball(q[:2,:2],b[:2]-q[:2,2]*z,np.sqrt(max(0,r*r-z*z)))
            s=np.r_[xy,z]
        return s
    except ValueError as exc:
        raise ValueError('physical_reset: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _bloch(value, name):
    a = _finite_array(value, name, (3,), real=True)
    if np.linalg.norm(a) > 1 + 1e-10:
        raise ValueError(name + ' must have norm at most one')
    return a

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _forward(h,xi,t):
    d=h.shape[0]//2;u=expm(-1j*t*h)
    return _super(lambda e:_ptr(u@np.kron(e,xi)@u.conj().T,d),d)

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _paulis():
    return np.array([np.eye(2), [[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],complex)

def _ptr(a,d):
    return np.trace(a.reshape(d,2,d,2),axis1=1,axis2=3)

def _real_scalar(value, name, positive=False, nonnegative=False):
    a = _finite_array(value, name, (), real=True)
    x = float(a)
    if (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(name + ' violates its sign domain')
    return x

def _super(action,d):
    out=np.empty((d*d,d*d),complex)
    for j in range(d):
        for i in range(d):
            e=np.zeros((d,d),complex);e[i,j]=1
            out[:,i+d*j]=action(e).reshape(-1,order='F')
    return out

def transition_recovery_maps(hamiltonian,bath,prior,next_prior,time,reset):
    try:
        g=_density(prior,'prior',positive_definite=True);d=len(g)
        h=_hamiltonian(hamiltonian,d);xi=_density(bath,'bath',dimension=2)
        gn=_density(next_prior,'next_prior',positive_definite=True,dimension=d)
        time=_real_scalar(time,'time',nonnegative=True);reset=_bloch(reset,'reset')
        expected=(_forward(h,xi,time)@g.reshape(-1,order='F')).reshape(d,d,order='F')
        if not np.allclose(gn,expected,atol=1e-10,rtol=0):raise ValueError('next_prior must equal the forward-evolved prior')
        d=len(g);u=expm(-1j*float(time)*h)
        eta=np.einsum('a,aij->ij',np.r_[1.,reset],_paulis())/2
        r,v=np.linalg.eigh(g);rn,vn=np.linalg.eigh(gn);p,e=np.linalg.eigh(xi);pp,ep=np.linalg.eigh(eta)
        phi=(np.kron(vn,ep).conj().T@u@np.kron(v,e)).reshape(d,2,d,2)
        pe=np.zeros((d*d,d*d),complex);te=pe.copy()
        for j in range(2):
            for k in range(2):
                rev=phi[:,j,:,k].conj().T
                kp=np.sqrt(max(p[k],0))*np.sqrt(r[:,None]/rn[None,:])*rev
                kt=np.sqrt(max(pp[j],0))*rev
                pe+=np.kron(kp.conj(),kp);te+=np.kron(kt.conj(),kt)
        left=np.kron(v.conj(),v);right=np.kron(vn.T,vn.conj().T)
        return np.stack([_forward(h,xi,time),left@pe@right,left@te@right])
    except ValueError as exc:
        raise ValueError('transition_recovery_maps: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _channel_dimension(a):
    d = int(round(np.sqrt(a.shape[-1])))
    if d < 1 or a.shape[-2:] != (d*d, d*d):
        raise ValueError('superoperator dimensions must be equal perfect squares')
    return d

def _channel_maps(value):
    a = _finite_array(value, 'maps')
    if a.ndim != 4 or len(a) < 1 or a.shape[1] != 3:
        raise ValueError('maps has an invalid shape')
    d = _channel_dimension(a)
    trace = np.eye(d).reshape(-1, order='F')
    for m in a.reshape(-1, d*d, d*d):
        if not np.allclose(trace @ m, trace, atol=1e-10, rtol=0):
            raise ValueError('maps must be trace preserving')
        choi = np.empty((d*d, d*d), complex)
        for i in range(d):
            for j in range(d):
                choi[i*d:(i+1)*d, j*d:(j+1)*d] = m[:, i+d*j].reshape(d, d, order='F')/d
        _hermitian(choi, 'channel Choi matrix')
        if np.linalg.eigvalsh(choi)[0] < -1e-10:
            raise ValueError('maps must be completely positive')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def reversal_error(maps):
    try:
        a=_channel_maps(maps);d=_channel_dimension(a);f=np.eye(d*d,dtype=complex);p=f.copy();r=f.copy()
        for stage in a:
            f=stage[0]@f;p=p@stage[1];r=r@stage[2]
        delta=(r-p)@f
        choi=np.zeros((d*d,d*d),complex)
        for i in range(d):
            for j in range(d):
                block=delta[:,i+d*j].reshape(d,d,order='F')/d
                choi[i*d:(i+1)*d,j*d:(j+1)*d]=block
        return float(np.sum(np.abs(np.linalg.eigvalsh((choi+choi.conj().T)/2)))/2)
    except ValueError as exc:
        raise ValueError('reversal_error: ' + str(exc)) from exc

import numpy as np
from scipy.linalg import expm, solve_sylvester
from scipy.optimize import brentq

def _bloch(value, name):
    a = _finite_array(value, name, (3,), real=True)
    if np.linalg.norm(a) > 1 + 1e-10:
        raise ValueError(name + ' must have norm at most one')
    return a

def _bounds(radius, z_floor):
    r = _real_scalar(radius, 'radius', positive=True)
    z = _real_scalar(z_floor, 'z_floor')
    if r > 1 or not -r <= z <= r:
        raise ValueError('Require 0 < radius <= 1 and -radius <= z_floor <= radius')
    return r, z

def _density(value, name, positive_definite=False, dimension=None):
    a = _hermitian(value, name)
    if dimension is not None and a.shape != (dimension, dimension):
        raise ValueError(name + ' has an invalid dimension')
    if abs(np.trace(a) - 1) > 1e-10:
        raise ValueError(name + ' must have unit trace')
    low = np.linalg.eigvalsh(a)[0]
    if (positive_definite and low <= 0) or (not positive_definite and low < -1e-10):
        raise ValueError(name + ' violates its positivity domain')
    return a

def _finite_array(value, name, shape=None, real=False):
    try:
        a = np.asarray(value, dtype=complex)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a numerical array') from exc
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must contain finite values')
    if shape is not None and a.shape != shape:
        raise ValueError(name + ' has an invalid shape')
    if real:
        if np.any(a.imag != 0):
            raise ValueError(name + ' must be real')
        a = a.real
    return a

def _hamiltonian(value, dimension=None):
    a = _hermitian(value, 'hamiltonian')
    if a.shape[0] % 2 or (dimension is not None and a.shape != (2*dimension, 2*dimension)):
        raise ValueError('hamiltonian must act on system then qubit ancilla')
    return a

def _hermitian(value, name):
    a = _finite_array(value, name)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[0] != a.shape[1]:
        raise ValueError(name + ' must be a nonempty square matrix')
    if not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError(name + ' must be Hermitian')
    return a

def _real_scalar(value, name, positive=False, nonnegative=False):
    a = _finite_array(value, name, (), real=True)
    x = float(a)
    if (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(name + ' violates its sign domain')
    return x

def _sequence_inputs(hamiltonians, bath, initial_prior, times):
    g = _density(initial_prior, 'initial_prior', positive_definite=True)
    xi = _density(bath, 'bath', dimension=2)
    hs = _finite_array(hamiltonians, 'hamiltonians')
    if hs.ndim != 3 or len(hs) < 1 or hs.shape[1:] != (2*len(g), 2*len(g)):
        raise ValueError('hamiltonians has an invalid shape')
    for h in hs:
        _hamiltonian(h, len(g))
    ts = _finite_array(times, 'times', (len(hs),), real=True)
    if np.any(ts < 0):
        raise ValueError('times must be nonnegative')
    return hs, xi, g, ts

def design_reversal(hamiltonians,bath,initial_prior,times,penalty,nominal,radius,z_floor):
    try:
        hamiltonians,bath,initial_prior,times=_sequence_inputs(hamiltonians,bath,initial_prior,times)
        penalty=_real_scalar(penalty,'penalty',positive=True);nominal=_bloch(nominal,'nominal')
        radius,z_floor=_bounds(radius,z_floor)
        path=prior_trajectory(hamiltonians,bath,initial_prior,times)
        p=[];a=[]
        for k,h in enumerate(hamiltonians):
            c=collision_coefficients(h,bath)
            w=normalizer_coefficients(path[k],c)
            p.append(petz_coefficients(path[k],c,w))
            a.append(reverse_coefficients(h))
        obj=shared_reset_objective(p,a,times,penalty,nominal)
        s=physical_reset(obj,radius,z_floor)
        maps=[transition_recovery_maps(h,bath,path[k],path[k+1],times[k],s) for k,h in enumerate(hamiltonians)]
        return reversal_error(maps)
    except ValueError as exc:
        raise ValueError('design_reversal: ' + str(exc)) from exc
SCICODE_GOLD_EOF
