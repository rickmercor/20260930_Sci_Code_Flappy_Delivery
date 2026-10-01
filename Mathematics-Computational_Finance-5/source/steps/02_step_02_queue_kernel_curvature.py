"""
Differentiate the persistent marked-queue transition law twice.



Q increases by one at rate lambda(alpha)+alpha*Q and decreases by one at

rate beta*Q. Every increase adds an independent N(mu_J,sigma_J**2) mark to

the uncompensated sum M. The initial state q is fixed. lambda_jet contains

lambda, d lambda/d alpha, and d2 lambda/d alpha2 at the nominal alpha.

The remaining model parameters and the time interval are held fixed.



For q,j=0,...,max_state, and each frequency v[k], compute the value and

first two ordinary alpha derivatives of

E[exp(1j*v[k]*(M(tau)-M(0))) * 1{Q(tau)=j} | Q(0)=q].

The last axis enumerates the real frequencies, the middle two enumerate

initial and terminal queue states, and the leading axis contains derivative

orders 0,1,2. Order two is the second derivative, not half of it.



The state process is infinite; trajectories above max_state that return

before tau contribute. Do not renormalize the retained endpoint states.

Obtain the derivatives from the exact joint generating function through

analytic differentiation. Finite differences, numerical Fourier inversion,

simulation, time integration and finite-state generator exponentiation are

not permitted in this function. No diffusion factor or discount is included.

The marked transition law retains dependence between terminal activation and accumulated marks. Parameter derivatives must retain that dependence.

Returns
-------
return np.zeros((3,max_state+1,max_state+1,np.size(v)),dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def queue_kernel_curvature(v,tau,lambda_jet,alpha,beta,mu_J,sigma_J,max_state):
    """Return a complex array with shape (3,max_state+1,max_state+1,len(v)).

    v: nonempty finite 1-D real array. lambda_jet: finite real shape (3,)
    with positive entry zero; its derivative entries may have either sign.
    Scalars: tau >= 0, beta > alpha > 0, sigma_J >= 0, finite mu_J.
    max_state: nonnegative integer. All scalar inputs must be finite.
    At tau=0, order zero is the identity and both derivatives are zero.
    Raise ValueError for invalid shapes, nonfinite inputs or invalid ranges.
    """
    return np.zeros((3,max_state+1,max_state+1,np.size(v)),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_queue_kernel_curvature(v,tau,lambda_jet,alpha,beta,mu_J,sigma_J,max_state):
    np=__import__('numpy')
    v=np.asarray(v,float);lj=np.asarray(lambda_jet,float)
    if v.ndim!=1 or not v.size or not np.all(np.isfinite(v)) or lj.shape!=(3,) or not np.all(np.isfinite(lj)):
        raise ValueError('array inputs')
    if not np.all(np.isfinite([tau,alpha,beta,mu_J,sigma_J])) or tau<0 or sigma_J<0 or not beta>alpha>0 or lj[0]<=0:
        raise ValueError('model inputs')
    if isinstance(max_state,(bool,np.bool_)) or not np.isscalar(max_state) or not np.isfinite(max_state) or int(max_state)!=max_state or max_state<0:
        raise ValueError('max_state')
    n=int(max_state)+1
    out=np.zeros((3,n,n,len(v)),complex)
    if tau==0:
        out[0,np.arange(n),np.arange(n),:]=1
        return out
    def const(x):
        x=np.asarray(x,complex)
        if x.ndim==0:x=x.reshape(1)
        return np.stack([x,np.zeros_like(x),np.zeros_like(x)])
    def mul(a,b):
        return np.stack([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])
    def inv(a):
        return np.stack([1/a[0],-a[1]/a[0]**2,2*a[1]**2/a[0]**3-a[2]/a[0]**2])
    def div(a,b):return mul(a,inv(b))
    def exp(a):
        z=np.exp(a[0]);return np.stack([z,z*a[1],z*(a[2]+a[1]**2)])
    def log(a):
        return np.stack([np.log(a[0]),a[1]/a[0],a[2]/a[0]-(a[1]/a[0])**2])
    al=np.array([[alpha],[1.],[0.]],complex)
    lam=lj[:,None].astype(complex)
    mark=const(np.exp(1j*mu_J*v-.5*sigma_J*sigma_J*v*v))
    f=exp(.5*log(mul(al+const(beta),al+const(beta))-4*beta*mul(al,mark)))
    e=exp(-tau*f);one=const(1)
    h=mul(f,one+e)+mul(const(beta)+al,one-e)
    p=div(2*mul(mul(al,mark),one-e),h)
    u=div(2*beta*(one-e),h)
    w=div(mul(f,one+e)-mul(const(beta)+al,one-e),h)
    r=div(lam,al)
    out[:,0,0,:]=exp(.5*tau*mul(r,const(beta)-al-f)+mul(r,log(div(2*f,h))))
    for j in range(1,n):
        out[:,0,j,:]=mul(mul(out[:,0,j-1,:],r+const(j-1)),p)/j
    ratio=np.zeros((3,n,len(v)),complex);ratio[:,0,:]=u
    if n>1:ratio[:,1,:]=w+mul(u,p)
    for j in range(2,n):ratio[:,j,:]=mul(ratio[:,j-1,:],p)
    for q in range(1,n):
        for j in range(n):
            out[:,q,j,:]=sum(mul(out[:,q-1,k,:],ratio[:,j-k,:]) for k in range(j+1))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'import numpy as np\n'
               'args=([0.0, 0.3, 1.1, -2.0], 0.8, [0.4, -0.12, 0.08], 0.35, 0.9, 0.15, 0.22, 5)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(queue_kernel_curvature(*args))',
      'gold_call': 'pack(_oracle_queue_kernel_curvature(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([0.0, 0.7, -0.7, 2.4], 2.2, [0.3, -0.05, -0.12], 0.55, 0.75, -0.12, 0.3, 12)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(queue_kernel_curvature(*args))',
      'gold_call': 'pack(_oracle_queue_kernel_curvature(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([0.0, 0.2, 1.7], 1.3, [0.15, 0.0, 0.0], 0.4, 1.1, 0.2, 0.15, 9)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(queue_kernel_curvature(*args))',
      'gold_call': 'pack(_oracle_queue_kernel_curvature(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([0.0, 0.4, 1.2], 0.6, [0.8, -0.6, 0.9], 0.08, 1.7, -0.18, 0.28, 7)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(queue_kernel_curvature(*args))',
      'gold_call': 'pack(_oracle_queue_kernel_curvature(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([0.0, 0.4], 0.0, [0.4, -0.12, 0.08], 0.35, 0.9, 0.15, 0.22, 5)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(queue_kernel_curvature(*args))',
      'gold_call': 'pack(_oracle_queue_kernel_curvature(*args))'},
     {'setup': 'def check(f):\n'
               '    try:\n'
               '        f([0.,1.],1.,[.4,-.1],.35,.9,.1,.2,4)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(queue_kernel_curvature)',
      'gold_call': 'check(_oracle_queue_kernel_curvature)'}]
