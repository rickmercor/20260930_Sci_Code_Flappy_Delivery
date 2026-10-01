"""
Return the exact affine terminal score and its executed-comparison cell for uncertain priors base+direction*x. H is binary shape(m,n), 1<=m<=8, 2<=n<=24, n>rounds and row degree>=rounds+2. base and direction have length n, are finite real binary64 inputs interpreted exactly and have magnitudes<=16 and<=4. syndrome is binary length m. sample is an integer pair[numerator,positive_denominator]; bounds has two such rows and satisfies -1<=lower<sample<upper<=1. initial_iters and inner_iters are nonboolean integers in[1,6], rounds in[1,3] and width in[1,2]. Return four reduced rational pairs for cell_lower,cell_upper,intercept,slope, with positive denominators. Invalid domains or a nonconstant comparison expression vanishing at sample raise ValueError. Identically zero expressions return sign zero without narrowing. Do not mutate inputs.

Use exact affine min-sum arithmetic and the conditional branching prescription: initialize outgoing messages to priors, use syndrome sign (-1)^s times excluded-live-neighbor sign product and minimum magnitude, form posteriors and extrinsic outgoing sums simultaneously, accumulate signed posteriors only within the current run and check syndrome after complete updates with posterior<=0 decoded as one. Masked-one columns flip original syndrome and every masked edge is excluded. Children inherit final outgoing messages, not accumulators. Select the smallest unmasked index minimizing accumulated magnitude; score is their magnitude sum divided by actual iterations. Expand zero then one, retain converged children and sort by descending score then lexicographic(index,value)history. The root score is zero. To define a unique open cell, intersect bounds with the sign-preserving half-line for every nonconstant sign query: query every other-neighbor message sign; scan minimum magnitudes in ascending fault index by candidate-minus-current; query every live posterior sign; query every accumulated posterior sign before absolute value; scan next-node magnitudes in ascending index by candidate-minus-current; query every pair of child score differences i<j before sorting. Equal minima keep the earlier entry. No other comparisons narrow the cell. Return the first retained terminal affine score. The cell fixes the entire comparison trace, not merely a coincidentally affine final value.

Returns
-------
list, four exact rational pairs for an affine trace cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def trace_score_cell(H, base, direction, syndrome, sample, bounds, initial_iters=4, inner_iters=3, rounds=3, width=2):
    """Return a rational affine score cell for conditional decoding.

    Parameters
    ----------
    H : array_like
        Binary incidence within the stated size and live-degree bounds.
    base : array_like
        Base log odds, interpreted as exact binary64 values.
    direction : array_like
        Affine log-odds perturbation coefficients.
    syndrome : array_like
        Binary original detector syndrome.
    sample : array_like
        Exact rational interior point as an integer numerator-denominator pair.
    bounds : array_like
        Two rational endpoint pairs in [-1,1].
    initial_iters : int
        Initial run budget in [1,6].
    inner_iters : int
        Child run budget in [1,6].
    rounds : int
        Branching depth in [1,3].
    width : int
        Retained population width in [1,2].

    Returns
    -------
    list
        Four reduced rational pairs encoding cell endpoints and affine score.

    Raises
    ------
    ValueError
        If the domain fails or a nonconstant queried expression is zero at sample.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_trace_score_cell(H, base, direction, syndrome, sample, bounds, initial_iters=4, inner_iters=3, rounds=3, width=2):
    from fractions import Fraction as F
    for val,high in ((initial_iters,6),(inner_iters,6),(rounds,3),(width,2)):
        if isinstance(val,(bool,np.bool_)) or not isinstance(val,(int,np.integer)) or not 1<=val<=high:raise ValueError('budget')
    vals=[]
    for x in (H,base,direction,syndrome):
        a=np.asarray(x)
        if a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):raise ValueError('real numeric array')
        vals.append(a)
    h,b,d,s=vals
    if h.ndim!=2 or not 1<=h.shape[0]<=8 or not 2<=h.shape[1]<=24:raise ValueError('matrix shape')
    m,n=h.shape
    if b.shape!=(n,) or d.shape!=(n,) or s.shape!=(m,):raise ValueError('vector shape')
    if np.any((h!=0)&(h!=1)) or np.any((s!=0)&(s!=1)):raise ValueError('binary')
    if n<=rounds or np.any(h.sum(1)<rounds+2) or np.any(np.abs(b)>16) or np.any(np.abs(d)>4):raise ValueError('bounds')
    def rational_table(v,shape):
        a=np.asarray(v,dtype=object)
        if a.shape!=shape:raise ValueError('rational shape')
        for x in a.flat:
            if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)):raise ValueError('integer pair')
        return a
    q=rational_table(sample,(2,));rr=rational_table(bounds,(2,2))
    if q[1]<=0 or any(row[1]<=0 for row in rr):raise ValueError('denominator')
    x=F(int(q[0]),int(q[1]));lo=F(int(rr[0,0]),int(rr[0,1]));hi=F(int(rr[1,0]),int(rr[1,1]))
    if not -1<=lo<x<hi<=1:raise ValueError('interval')
    cell=[lo,hi];cache={};zero=(F(0),F(0))
    def add(a,b):return a[0]+b[0],a[1]+b[1]
    def sub(a,b):return a[0]-b[0],a[1]-b[1]
    def mul(a,k):return a[0]*k,a[1]*k
    def total(values):
        a=zero
        for v in values:a=add(a,v)
        return a
    def sign(a):
        if a in cache:return cache[a]
        intercept,slope=a
        v=intercept+slope*x
        if not slope:z=(v>0)-(v<0)
        else:
            if v==0:raise ValueError('sample at nonconstant comparison boundary')
            z=1 if v>0 else -1;root=-intercept/slope
            if (z>0)==(slope>0):cell[0]=max(cell[0],root)
            else:cell[1]=min(cell[1],root)
        cache[a]=z
        return z
    def minimum(values):
        best=values[0]
        for v in values[1:]:
            if sign(sub(v,best))<0:best=v
        return best
    def argmin(values):
        best=0
        for k in range(1,len(values)):
            if sign(sub(values[k],values[best]))<0:best=k
        return best
    priors=[(F(float(b[j])),F(float(d[j]))) for j in range(n)]
    Nd=[[j for j in range(n) if h[i,j]] for i in range(m)]
    Ne=[[i for i in range(m) if h[i,j]] for j in range(n)]
    def bp(edge,path,budget):
        fixed=dict(path);res=[int(v) for v in s]
        for j,val in path:
            if val:
                for i in Ne[j]:res[i]^=1
        live=[j for j in range(n) if j not in fixed]
        E=edge.copy();acc=[zero for _ in range(n)]
        for t in range(1,budget+1):
            D={}
            for i in range(m):
                nb=[j for j in Nd[i] if j not in fixed]
                for j in nb:
                    values=[E[k,i] for k in nb if k!=j]
                    signs=[sign(v) for v in values]
                    factor=(-1)**res[i]
                    for z in signs:factor*=z
                    D[i,j]=mul(minimum([mul(v,z) for v,z in zip(values,signs)]),factor)
            new=E.copy();err=[0]*n
            for j in live:
                post=add(priors[j],total(D[i,j] for i in Ne[j]))
                for i in Ne[j]:new[j,i]=add(priors[j],total(D[k,j] for k in Ne[j] if k!=i))
                acc[j]=add(acc[j],post);err[j]=int(sign(post)<=0)
            E=new
            if all(sum(err[j] for j in Nd[i] if j not in fixed)%2==res[i] for i in range(m)):break
        mags=[mul(acc[j],sign(acc[j])) for j in live]
        node=live[argmin(mags)]
        return E,node,mul(total(mags),F(1,t))
    E0={(j,i):priors[j] for j in range(n) for i in Ne[j]}
    E,node,_=bp(E0,(),initial_iters)
    beam=[(E,(),node,zero)]
    for _ in range(rounds):
        children=[]
        for edge,path,j,score in beam:
            for val in (0,1):
                pp=path+((j,val),);ee,jj,ss=bp(edge,pp,inner_iters)
                children.append((ee,pp,jj,ss))
        for i in range(len(children)):
            for j in range(i+1,len(children)):sign(sub(children[i][3],children[j][3]))
        children.sort(key=lambda p:(-(p[3][0]+p[3][1]*x),p[1]));beam=children[:width]
    return [[int(v.numerator),int(v.denominator)] for v in [cell[0],cell[1],*beam[0][3]]]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nH=np.ones((2,4),int);b=np.array([.25,1.,-.5,2.]);d=np.array([0.,.5,0.,0.]);s=np.array([1,0])',
            'call': 'trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[1,7],[[-1,2],[1,2]],2,2,1,2)',
            'gold_call': '_oracle_trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[1,7],[[-1,2],[1,2]],2,2,1,2)',
        },
        {
            'setup': 'import numpy as np\nH=np.ones((1,3),int);b=np.zeros(3);d=np.zeros(3);s=np.array([0])',
            'call': 'trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[0,1],[[-1,1],[1,1]],1,1,1,1)',
            'gold_call': '_oracle_trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[0,1],[[-1,1],[1,1]],1,1,1,1)',
        },
        {
            'setup': 'import numpy as np\nH=np.ones((2,5),int);b=np.array([.5,-.75,1.25,2.,.125]);d=np.array([.25,0.,-.5,0.,.25]);s=np.array([1,1])',
            'call': 'trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[2,7],[[-1,1],[1,1]],3,2,2,2)',
            'gold_call': '_oracle_trace_score_cell(H.copy(),b.copy(),d.copy(),s.copy(),[2,7],[[-1,1],[1,1]],3,2,2,2)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: trace_score_cell([[1,1,1]],[0,0,0],[1,0,0],[0],[0,1],[[-1,1],[1,1]],1,1,1,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_trace_score_cell([[1,1,1]],[0,0,0],[1,0,0],[0],[0,1],[[-1,1],[1,1]],1,1,1,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
