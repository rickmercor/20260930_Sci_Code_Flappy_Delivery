"""
Project the compound payoff and its clustering derivatives analytically. For activation state q, C[d,q,k] is the d-th ordinary parameter derivative of the complex coefficient of a finite continuation curve V_q(x), d=0,1,2. The curve uses Re(C*exp(1j*omega*(x-a_next))), with k=0 half weighted. At nominal parameter alpha the supplied root r_q solves V_q(r_q)=cost, with positive x derivative. The cost, intervals and frequencies are fixed as alpha varies; the coefficients and exercise boundaries vary.

Return the value and first two alpha derivatives of the normalized outer projection 2/(b-a) integral from r_q(alpha) to b of (V_q(x,alpha)-cost)*cos(nu[n]*(x-a)) dx, for every q and outer mode n. Evaluate the integrals and boundary-motion terms analytically, without numerical quadrature or finite differences. Equal inner and outer frequencies, and both zero frequencies, are admissible.

A moving exercise boundary contributes to the second derivative even though the nominal payoff vanishes at the crossing.

Returns
-------
return np.zeros((3,np.shape(C)[1],np.size(nu)))
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def queue_payoff_jet(C,omega,nu,a_next,a,b,roots,cost):
    """Return finite float shape (3,Q,len(nu)), in derivative order 0,1,2.

    C: finite complex (3,Q,N), Q,N>0. omega: finite (N,) and nu: finite
    nonempty 1-D arrays, both starting at zero and strictly increasing.
    roots: finite (Q,), with a<roots[q]<b. Scalars a_next,a,b,cost are
    finite, b>a, cost>0. The nominal root identity is guaranteed for valid
    input. Raise ValueError for invalid shapes/ranges, nonfinite inputs,
    or nonpositive continuation slope at a supplied root.
    """
    return np.zeros((3,np.shape(C)[1],np.size(nu)))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_queue_payoff_jet(C,omega,nu,a_next,a,b,roots,cost):
    np=__import__('numpy');C=np.asarray(C,complex);w=np.asarray(omega,float);nu=np.asarray(nu,float);roots=np.asarray(roots,float)
    if C.ndim!=3 or C.shape[0]!=3 or min(C.shape)<1 or w.shape!=(C.shape[2],) or roots.shape!=(C.shape[1],) or nu.ndim!=1 or not nu.size:
        raise ValueError('shapes')
    if not all(np.all(np.isfinite(x)) for x in [C,w,nu,roots]) or not np.all(np.isfinite([a_next,a,b,cost])):
        raise ValueError('nonfinite inputs')
    if b<=a or cost<=0 or w[0]!=0 or nu[0]!=0 or np.any(np.diff(w)<=0) or np.any(np.diff(nu)<=0) or np.any(roots<=a) or np.any(roots>=b):
        raise ValueError('ranges')
    prime=np.ones(len(w));prime[0]=.5
    out=np.zeros((3,len(roots),len(nu)))
    minus=w[None,:]-nu[:,None];plus=w[None,:]+nu[:,None]
    phase_minus=w[None,:]*a_next-nu[:,None]*a
    phase_plus=w[None,:]*a_next+nu[:,None]*a
    for q,r in enumerate(roots):
        length=b-r;middle=(b+r)/2
        sm=np.sinc(minus*length/(2*np.pi));sp=np.sinc(plus*length/(2*np.pi))
        Ic=length/(b-a)*(np.cos(minus*middle-phase_minus)*sm+np.cos(plus*middle-phase_plus)*sp)
        Is=length/(b-a)*(np.sin(minus*middle-phase_minus)*sm+np.sin(plus*middle-phase_plus)*sp)
        Io=2*length/(b-a)*np.cos(nu*(middle-a))*np.sinc(nu*length/(2*np.pi))
        for d in range(3):out[d,q]=(Ic*prime)@C[d,q].real-(Is*prime)@C[d,q].imag
        out[0,q]-=cost*Io
        phase=np.exp(1j*w*(r-a_next))
        slope=np.real(C[0,q]*1j*w*phase)@prime
        first=np.real(C[1,q]*phase)@prime
        if slope<=0:raise ValueError('nonpositive slope')
        out[2,q]+=2/(b-a)*np.cos(nu*(r-a))*first*first/slope
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(13)\n'
               'q,n,m=2,5,7\n'
               'w=np.arange(n)*.8;nu=np.arange(m)*.8\n'
               'roots=np.linspace(.7,1.8,q);C=np.zeros((3,q,n),complex)\n'
               'C[0,:,0]=2*(3.5+2*np.cos(.8*(roots+1)));C[0,:,1]=-2\n'
               'C[1:]=.15*(rng.normal(size=(2,q,n))+1j*rng.normal(size=(2,q,n)))\n'
               'args=(C,w,nu,-1.,0.,2.8,roots,3.5)\n',
      'call': 'queue_payoff_jet(*args).ravel().tolist()',
      'gold_call': '_oracle_queue_payoff_jet(*args).ravel().tolist()'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(23)\n'
               'q,n,m=3,8,8\n'
               'w=np.arange(n)*.8;nu=np.arange(m)*.8\n'
               'roots=np.linspace(.7,1.8,q);C=np.zeros((3,q,n),complex)\n'
               'C[0,:,0]=2*(3.5+2*np.cos(.8*(roots+1)));C[0,:,1]=-2\n'
               'C[1:]=.15*(rng.normal(size=(2,q,n))+1j*rng.normal(size=(2,q,n)))\n'
               'args=(C,w,nu,-1.,0.,2.8,roots,3.5)\n',
      'call': 'queue_payoff_jet(*args).ravel().tolist()',
      'gold_call': '_oracle_queue_payoff_jet(*args).ravel().tolist()'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(31)\n'
               'q,n,m=1,3,4\n'
               'w=np.arange(n)*.8;nu=np.arange(m)*.8\n'
               'roots=np.linspace(.7,1.8,q);C=np.zeros((3,q,n),complex)\n'
               'C[0,:,0]=2*(3.5+2*np.cos(.8*(roots+1)));C[0,:,1]=-2\n'
               'C[1:]=.15*(rng.normal(size=(2,q,n))+1j*rng.normal(size=(2,q,n)))\n'
               'args=(C,w,nu,-1.,0.,2.8,roots,3.5)\n',
      'call': 'queue_payoff_jet(*args).ravel().tolist()',
      'gold_call': '_oracle_queue_payoff_jet(*args).ravel().tolist()'},
     {'setup': 'import numpy as np\n'
               'def check(f):\n'
               '    try:\n'
               '        f(np.ones((3,1,3)),[0.,1.,2.],[0.,1.],0.,0.,2.,[3.],1.)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(queue_payoff_jet)',
      'gold_call': 'check(_oracle_queue_payoff_jet)'}]
