"""
Run warm-started masked flooding min-sum. Obtain the residual syndrome from residual_syndrome(H,syndrome,fixed). On each live edge, the check message is (-1) to the residual bit power times the product of signs and minimum absolute value of the other live incoming fault messages; sign(0)=0. A live posterior is its prior plus incoming check messages. Its outgoing message excludes the recipient's check message. Accumulate signed posteriors starting at zero for this call only. Complete all outgoing updates before testing the residual syndrome, using live decisions posterior<=0 and zeros in masked positions. Stop after the first match or max_iters, never before iteration one. Return (messages,posterior_sum,iterations,correction,success), with fixed values restored even on failure, masked edge messages unchanged and nonedges zero. The matrix, syndrome and mask satisfy the residual-syndrome domain. Priors have length n and magnitude<=16, messages have shape (m,n), magnitude<=1e18 and zero nonedges; all are finite real numeric arrays, including zero-imaginary complex arrays. Treat input binary64 values as exact real inputs to the recurrence, without intermediate rounding that changes a sign, minimum or convergence decision. Return binary64 arrays with absolute error<=1e-9 plus 1e-12 times the reference magnitude in each entry. This includes small residuals after cancellation of large warm messages. max_iters is a nonboolean integer or integer-valued real scalar in [1,32]. Invalid inputs raise ValueError. Do not mutate inputs.

The conditional run removes masked variables from every parity sign and minimum, not just from the hard decision. Resetting the accumulator while retaining the parent's extrinsic messages separates present conditional evidence from inherited initialization. Its actual iteration count includes a converging iteration and later normalizes predictive reliability.

Returns
-------
tuple, conditional trajectory state with numeric arrays and integers
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def masked_run(H, prior, syndrome, messages, fixed, max_iters):
    """Run one conditional min-sum trajectory.

    Parameters
    ----------
    H : array_like
        Binary detector matrix, 1<=m<=32 and 2<=n<=64.
    prior : array_like
        Finite real log odds, shape (n,), magnitude at most 16.
    syndrome : array_like
        Original binary syndrome, shape (m,).
    messages : array_like
        Incoming fault-to-detector snapshot, shape (m,n), bounded as above.
    fixed : array_like
        Mask vector of -1, 0 or 1 leaving at least two live faults per detector.
    max_iters : int
        Nonboolean integer-valued iteration budget in [1,32].

    Returns
    -------
    tuple
        Final messages, signed posterior sum, executed count, full correction
        and integer success flag. Array shapes are (m,n), (n,) and (n,).

    Raises
    ------
    ValueError
        If any stated domain condition fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_masked_run(H, prior, syndrome, messages, fixed, max_iters):
    from fractions import Fraction
    residual=_oracle_residual_syndrome(H,syndrome,fixed)
    h=np.asarray(H).real.astype(int);f=np.asarray(fixed).real.astype(int)
    n=h.shape[1]
    vals=[]
    for x,shape,bound in ((prior,(n,),16),(messages,h.shape,1e18)):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(a.imag!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if a.shape!=shape or not np.all(np.isfinite(a)) or np.any(np.abs(a)>bound):raise ValueError('numeric domain')
        vals.append(a.copy())
    L,E=vals
    if np.any(E[h==0]!=0):raise ValueError('nonedge message')
    q=np.asarray(max_iters)
    if q.ndim!=0 or q.dtype.kind not in 'iuf' or not np.isfinite(q) or q!=np.floor(q) or not 1<=q<=32:raise ValueError('iteration budget')
    live=f<0
    L=[Fraction(float(x)) for x in L]
    E=np.array([[Fraction(float(x)) for x in row] for row in E],dtype=object)
    acc=np.array([Fraction(0) for _ in range(n)],dtype=object)
    err=np.zeros(n,dtype=int);ok=False;t=0
    for t in range(1,int(q)+1):
        D=np.full(h.shape,Fraction(0),dtype=object)
        for i in range(h.shape[0]):
            nb=np.flatnonzero(h[i]&live)
            for j in nb:
                v=E[i,nb[nb!=j]]
                sign=-1 if residual[i] else 1
                for x in v:sign*=1 if x>0 else (-1 if x<0 else 0)
                D[i,j]=sign*min(abs(x) for x in v)
        posterior=np.array([Fraction(0) for _ in range(n)],dtype=object);new=E.copy()
        for j in np.flatnonzero(live):
            nb=np.flatnonzero(h[:,j]);posterior[j]=L[j]+sum(D[i,j] for i in nb)
            for i in nb:new[i,j]=L[j]+sum(D[k,j] for k in nb if k!=i)
        E=new;acc+=posterior
        err=((posterior<=0)&live).astype(int)
        ok=np.array_equal((h@err)%2,residual)
        if ok:break
    err[~live]=f[~live]
    return E.astype(float),acc.astype(float),int(t),err,int(ok)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,0],[0,1,1,1]]);L=np.array([1.,-.5,2.,.3]);s=np.array([1,0]);E=np.array([[.5,-1,2,0],[0,1.5,-.7,.2]]);f=np.array([-1,-1,-1,1])',
            'call': 'masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),5)',
            'gold_call': '_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),5)',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\nH=np.ones((1,2),int);L=np.zeros(2);s=np.array([0]);E=np.zeros((1,2));f=np.array([-1,-1])',
            'call': 'masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),1)',
            'gold_call': '_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),1)',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,1,0],[0,1,1,1,1]]);L=np.array([2.,-1.,.25,3.,-.5]);s=np.array([0,1]);E=H*np.array([.5,0,-.25,2,1]);f=np.array([1,-1,-1,0,-1])',
            'call': 'masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),8)',
            'gold_call': '_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),8)',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: masked_run([[1,1]],[0,0],[0],[[0,0]],[-1,-1],True)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_masked_run([[1,1]],[0,0],[0],[[0,0]],[-1,-1],True)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\nH=np.ones((3,4),int);L=np.array([.25,-.5,.75,-1.25]);s=np.zeros(3,int);E=np.array([[1e16]*4,[1.]*4,[-1e16,1e16,1e16,1e16]]);f=np.full(4,-1)',
            'call': 'masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),1)',
            'gold_call': '_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),1)',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\nH=np.ones((3,4),int);L=np.array([.25,-.5,.75,-1.25]);s=np.zeros(3,int);E=np.array([[1e16]*4,[1.]*4,[-1e16,1e16,1e16,1e16]]);f=np.full(4,-1)',
            'call': '(lambda out: np.array([out[0][1,0],out[2],out[4]]))(masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),5))',
            'gold_call': '(lambda out: np.array([out[0][1,0],out[2],out[4]]))(_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),5))',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
        {
            'setup': 'import numpy as np\nH=np.ones((3,5),int);L=np.array([.25,-.5,.75,-1.25,2.]);s=np.ones(3,int);E=np.array([[1e17]*5,[.125]*5,[-1e17,1e17,1e17,1e17,1e17]]);f=np.array([-1,-1,-1,-1,1])',
            'call': 'masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),3)',
            'gold_call': '_oracle_masked_run(H.copy(),L.copy(),s.copy(),E.copy(),f.copy(),3)',
            'atol': 1e-09,
            'rtol': 1e-12,
        },
    ]
