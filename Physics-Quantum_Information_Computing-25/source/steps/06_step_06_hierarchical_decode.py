"""
Compute the full correction profile of the hierarchical decoder under affine calibration uncertainty.

Local and global decisions interact: a candidate changes the residual syndrome, which changes the block profile that determines its total cost.

Returns
-------
tuple: (breakpoints, errors), where breakpoints is a tuple of reduced integer (numerator, positive_denominator) pairs covering gain_interval, and errors is an integer 0/1 array of shape (len(breakpoints) - 1, n), with corrections in original D-column order and adjacent identical corrections merged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hierarchical_decode(
    syndrome: "np.ndarray",
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    t_matrix: "np.ndarray",
    n_blocks: int,
    block_cols: int,
    weights: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return the exact gain-dependent hierarchical correction profile.

    Each weight row is the integer pair $(a, b)$ representing the
    affine objective coefficient $a + b\alpha$ at calibration gain $\alpha$.
    ``gain_interval`` is ``((lo_num, lo_den), (hi_num, hi_den))``, with
    positive denominators and $0 < \mathrm{lo} < \mathrm{hi}$. At each fixed
    gain use the specified monotone greedy decoder. Return its complete
    piecewise-constant correction profile, including every change on an open
    interval. Values at isolated breakpoints and interval endpoints do not
    contribute to the requested average and are excluded from the profile
    contract.

    The result is ``(breakpoints, errors)``. ``breakpoints`` is a tuple of
    reduced integer numerator/positive-denominator pairs in strictly
    increasing order, starting at $\mathrm{lo}$ and ending at $\mathrm{hi}$.
    Row $i$ of the integer 0/1 array ``errors`` applies between breakpoint
    $i$ and $i+1$. Adjacent rows must differ; merge adjacent intervals with
    the same full correction even when internal greedy decisions changed.
    Integer affine coefficients make the breakpoints rational. A sampled gain
    grid does not define this exact profile.

    $D' = T D_{\pi} \bmod 2$, where $D_{\pi}$ = ``D[:, perm]``, contains
    $K$ = ``n_blocks`` diagonal blocks $(I \mid B_i)$ followed by $A$. Each
    fixed-gain outer search starts from $r = 0$ completed by all block
    decodings. Each round tries every candidate that sets one currently-zero
    $r$ bit to one; bits are never cleared. For each candidate, compute
    $(T s + A r) \bmod 2$, where $s$ = ``syndrome``, and decode every block on
    its own residual rows using ``greedy_guess_block`` at that gain with the
    same ``max_iter``. Compare the complete objective of all block errors and
    $r$. Accept the lowest-cost candidate only on a strict improvement,
    resolving equal costs by the lowest newly set $A$-column index; otherwise
    stop. Stop after ``max_iter`` rounds or when no zero bits remain. The
    zero-completed incumbent is the explicit correction to the source's
    infinite-incumbent initialization.

    Weights refer to original $D$ columns; keep them attached through ``perm``.
    Each returned correction is in original column order. Use the block
    profiles from ``greedy_guess_block`` and the residual syndromes from
    ``compute_left_syndromes``. Both inner and outer decision changes affect
    the returned profile; discard only boundaries where the full returned
    correction stays identical on the two neighboring open intervals.

    Parameters
    ----------
    syndrome : np.ndarray
        Binary length-$m$ syndrome $s$ in the original row basis.
    d_matrix : np.ndarray
        Binary array $D$ of shape $(m, n)$.
    perm : np.ndarray
        Integer permutation of ``range(n)``.
    t_matrix : np.ndarray
        Binary invertible $(m, m)$ row transformation $T$.
    n_blocks : int
        Positive block count $K$ dividing $m$.
    block_cols : int
        Columns per block $b$, strictly larger than $m/K$; $K b \le n$.
    weights : np.ndarray
        Integer affine coefficients of shape $(n, 2)$, in original column order.
    max_iter : int
        Non-negative round cap at both levels.
    gain_interval : tuple
        Two positive rational endpoints as defined above.

    Returns
    -------
    tuple
        Rational breakpoints and binary errors of shape $(N_{\mathrm{int}},\ n)$,
        where $N_{\mathrm{int}}$ is the number of intervals. Every correction
        $e$ satisfies $D e = s \pmod 2$.

    Raises
    ------
    ValueError
        If shapes, binary entries, integer coefficients, cap, interval or
        decoupled layout violate the stated requirements.

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction
from bisect import bisect_right

def _oracle_hierarchical_decode(
    syndrome: "np.ndarray",
    d_matrix: "np.ndarray",
    perm: "np.ndarray",
    t_matrix: "np.ndarray",
    n_blocks: int,
    block_cols: int,
    weights: "np.ndarray",
    max_iter: int,
    gain_interval: tuple,
) -> tuple:
    d = np.asarray(d_matrix)
    if d.ndim != 2 or min(d.shape) < 1 or not np.all((d == 0) | (d == 1)):
        raise ValueError("D must be a nonempty binary matrix")
    d = d.astype(np.int64);m,n = d.shape
    s = np.asarray(syndrome)
    if s.shape != (m,) or not np.all((s == 0) | (s == 1)):
        raise ValueError("syndrome must be binary with m entries")
    s = s.astype(np.int64)
    p = np.asarray(perm);t = np.asarray(t_matrix)
    if p.shape != (n,) or not np.issubdtype(p.dtype,np.integer) or not np.array_equal(np.sort(p),np.arange(n)):
        raise ValueError("perm must be a permutation")
    if t.shape != (m,m) or not np.all((t == 0) | (t == 1)):
        raise ValueError("T must be binary with shape (m,m)")
    if isinstance(n_blocks,(bool,np.bool_)) or not isinstance(n_blocks,(int,np.integer)) or n_blocks < 1 or m%n_blocks:
        raise ValueError("invalid block count")
    rows=m//int(n_blocks)
    if isinstance(block_cols,(bool,np.bool_)) or not isinstance(block_cols,(int,np.integer)) or block_cols <= rows or n_blocks*block_cols > n:
        raise ValueError("invalid block width")
    if isinstance(max_iter,(bool,np.bool_)) or not isinstance(max_iter,(int,np.integer)) or max_iter < 0:
        raise ValueError("invalid cap")
    w=_affine_array(weights,n);lo,hi=_gain_bounds(gain_interval)
    k,width=int(n_blocks),int(block_cols)
    dp=(t.astype(np.int64)@d[:,p])%2;wp=w[p]
    for i in range(k):
        part=dp[:,i*width:(i+1)*width]
        if np.delete(part,np.s_[i*rows:(i+1)*rows],axis=0).any() or not np.array_equal(part[i*rows:(i+1)*rows,:rows],np.eye(rows,dtype=int)):
            raise ValueError("matrix is not decoupled in the stated layout")
    a=dp[:,k*width:];cache={}
    def complete(r,x):
        residuals=_oracle_compute_left_syndromes(s,t,a,r,k)
        pieces=[];end=hi
        for i,block_s in enumerate(residuals):
            key=(i,tuple(int(v) for v in block_s))
            if key not in cache:
                offset=i*width
                bp,es=_oracle_greedy_guess_block(dp[i*rows:(i+1)*rows,offset+rows:offset+width],block_s,wp[offset:offset+rows],wp[offset+rows:offset+width],max_iter,gain_interval)
                cache[key]=([Fraction(*pair) for pair in bp],es)
            edges,es=cache[key];idx=bisect_right(edges,x)-1
            pieces.append(es[idx]);end=min(end,edges[idx+1])
        error=np.concatenate(pieces+[r])
        return error,_profile_cost(wp,error),end
    edges,errors,x=[lo],[],lo
    while x < hi:
        r=np.zeros(a.shape[1],dtype=np.int64);e,incumbent,end=complete(r,x)
        for _ in range(int(max_iter)):
            candidates=[]
            for j in np.flatnonzero(r==0):
                cr=r.copy();cr[j]=1
                ce,cost,ce_end=complete(cr,x);end=min(end,ce_end)
                candidates.append((cost,int(j),cr,ce))
            if not candidates:break
            cost,_,cr,ce=min(candidates,key=lambda item:(_right_key(item[0],x),item[1]))
            for other,_,_,_ in candidates:end=_crossing(x,end,cost,other)
            if _right_key(cost,x) < _right_key(incumbent,x):
                end=_crossing(x,end,cost,incumbent);r,e,incumbent=cr,ce,cost
            else:
                end=_crossing(x,end,incumbent,cost);break
        original=np.empty(n,dtype=np.int64);original[p]=e
        _profile_append(edges,errors,end,original);x=end
    return _profile_result(edges,errors)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return explicit differential cases for Studio contract inspection."""
    return [{'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),3,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),3,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode((T@np.array([1,1,1,0]))%2,SCR.copy(),P.copy(),T.copy(),2,3,WS.copy(),3,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode((T@np.array([1,1,1,0]))%2,SCR.copy(),P.copy(),T.copy(),2,3,WS.copy(),3,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,0,1,1]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),1,((1,4),(3,1)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,0,1,1]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),1,((1,4),(3,1)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.zeros(4,dtype=int),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),3,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode(np.zeros(4,dtype=int),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),3,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),0,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,W.copy(),0,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,1]),np.array([[1,0,1],[0,1,1]]),np.arange(3),np.eye(2,dtype=int),1,3,np.array([[5,0],[8,0],[0,7]]),3,((1,2),(2,1)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,1]),np.array([[1,0,1],[0,1,1]]),np.arange(3),np.eye(2,dtype=int),1,3,np.array([[5,0],[8,0],[0,7]]),3,((1,2),(2,1)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,np.array([[6,0],[6,0],[0,6],[6,0],[6,0],[0,6],[0,6],[0,6]]),3,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,1,1,0]),D.copy(),np.arange(8),np.eye(4,dtype=int),2,3,np.array([[6,0],[6,0],[0,6],[6,0],[6,0],[0,6],[0,6],[0,6]]),3,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.ones(12,dtype=int),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((3,4),(5,4)))',
      'gold_call': '_oracle_hierarchical_decode(np.ones(12,dtype=int),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((3,4),(5,4)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.array([1,0,1,0,0,1,0,0,0,0,0,0]),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((1,2),(3,2)))',
      'gold_call': '_oracle_hierarchical_decode(np.array([1,0,1,0,0,1,0,0,0,0,0,0]),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((1,2),(3,2)))',
      'tol': 1e-12},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,0,1,0,0,0,1,0],[0,1,1,0,0,0,0,1],[0,0,0,1,0,1,1,0],[0,0,0,0,1,1,0,1]])\n'
               'W=np.array([[8,0],[6,0],[0,7],[9,0],[5,0],[0,8],[0,4],[1,3]])\n'
               'P=np.array([4,0,6,1,7,3,2,5])\n'
               'T=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]])\n'
               'SCR=np.empty_like(D);SCR[:,P]=(T@D)%2\n'
               'WS=np.empty_like(W);WS[P]=W\n'
               'H1=np.eye(3,dtype=int)|np.roll(np.eye(3,dtype=int),1,axis=1)\n'
               'H2=np.eye(4,dtype=int)|np.roll(np.eye(4,dtype=int),1,axis=1)\n'
               'DL=np.hstack((np.kron(H1,np.eye(4,dtype=int)),np.kron(np.eye(3,dtype=int),H2.T),np.eye(12,dtype=int)))\n'
               'PL=np.array([24,25,26,27,12,13,14,15,28,29,30,31,16,17,18,19,32,33,34,35,20,21,22,23,0,1,2,3,4,5,6,7,8,9,10,11])\n'
               'import math\n'
               'probs=np.r_[.010+.001*np.arange(24),.006+.0005*np.arange(12)]\n'
               'WL=np.zeros((36,2),dtype=int)\n'
               'for j,p in enumerate(probs): WL[j,1 if j<24 else 0]=round(1000000*math.log((1-p)/p))\n',
      'call': 'hierarchical_decode(np.ones(12,dtype=int),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((999563,1000000),(199913,200000)))',
      'gold_call': '_oracle_hierarchical_decode(np.ones(12,dtype=int),DL.copy(),PL.copy(),np.eye(12,dtype=int),3,8,WL.copy(),3,((999563,1000000),(199913,200000)))',
      'tol': 1e-12}]
