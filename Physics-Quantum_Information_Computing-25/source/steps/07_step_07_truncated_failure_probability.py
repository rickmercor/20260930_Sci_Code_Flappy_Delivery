"""
Integrate stabilizer-equivalent decoding success over calibration gain and sum the unconditional fault probabilities.

One fault support can succeed over disconnected gain intervals. Degenerate data corrections must be grouped by stabilizer equivalence, while actual noise probabilities stay fixed.

Returns
-------
tuple: (counts, contributions), two floating NumPy arrays of shape (max_faults + 1,), indexed by fault count; counts contains gain-averaged numbers of failing configurations, and contributions contains their unconditional gain-averaged failure probabilities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def truncated_failure_probability(
    d_matrix: "np.ndarray",
    hz: "np.ndarray",
    probabilities: "np.ndarray",
    decode_fn: "Callable[[np.ndarray], tuple]",
    max_faults: int,
    gain_interval: tuple,
) -> tuple:
    r"""Return gain-averaged failing counts and probability contributions by fault order.

    Faults independently occupy the columns of $D$ with the supplied
    probabilities. The first ``hz.shape[1]`` columns are data faults. For each
    support of size at most ``max_faults``, including the empty support, use
    its binary syndrome to obtain ``(breakpoints, corrections)`` from
    ``decode_fn``. The profile covers ``gain_interval`` in the rational format
    of ``hierarchical_decode``. The gain is uniform on this interval and
    independent of the faults.

    On a profile interval, failure means that the actual data error $\oplus$
    the correction's data part is outside the $\mathrm{GF}(2)$ row space of
    ``hz``. Let $\lambda(S)$ be the total length of failing gain intervals
    divided by $\mathrm{hi} - \mathrm{lo}$ for support $S$. Return arrays
    ``counts`` and ``contributions``, each of length ``max_faults`` $+\,1$,
    where ``counts[k]`` sums $\lambda(S)$ over $k$-fault supports and
    ``contributions[k]`` sums $\lambda(S)$ times the unconditional
    probability of each support, including absence of all other faults.
    Counts can be nonintegral. Do not condition on the retained fault order.

    Parameters
    ----------
    d_matrix : np.ndarray
        Binary array $D$ of shape $(m, n)$.
    hz : np.ndarray
        Binary stabilizer generators with $1 \le n_{\mathrm{data}} \le n$ columns.
    probabilities : np.ndarray
        Length-$n$ probabilities in $[0, 1]$, including deterministic faults.
    decode_fn : callable
        Pure deterministic syndrome-to-profile mapping; corrections have
        $n$ entries and satisfy $D c = s \pmod 2$. Profiles may be cached.
    max_faults : int
        Integer in $[0, n]$.
    gain_interval : tuple
        Two rational endpoints, ``((lo_num, lo_den), (hi_num, hi_den))``,
        $0 < \mathrm{lo} < \mathrm{hi}$.

    Returns
    -------
    tuple
        Two floating arrays ``(counts, contributions)``, ordered by fault count.

    Raises
    ------
    ValueError
        If inputs violate the stated domains or a returned profile has
        invalid rational endpoints, does not cover the requested interval,
        has unordered boundaries or invalid corrections, or has a wrong syndrome.

    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction

def _oracle_truncated_failure_probability(
    d_matrix: "np.ndarray",
    hz: "np.ndarray",
    probabilities: "np.ndarray",
    decode_fn: "Callable[[np.ndarray], tuple]",
    max_faults: int,
    gain_interval: tuple,
) -> tuple:
    import itertools
    import math
    d=np.asarray(d_matrix);h=np.asarray(hz)
    if d.ndim!=2 or min(d.shape)<1 or not np.all((d==0)|(d==1)):
        raise ValueError("invalid fault matrix")
    d=d.astype(np.int64);m,n=d.shape
    if h.ndim!=2 or not 1<=h.shape[1]<=n or not np.all((h==0)|(h==1)):
        raise ValueError("invalid stabilizer matrix")
    nd=h.shape[1];p=np.asarray(probabilities,dtype=float)
    if p.shape!=(n,) or not np.all((p>=0)&(p<=1)):
        raise ValueError("invalid probabilities")
    if isinstance(max_faults,(bool,np.bool_)) or not isinstance(max_faults,(int,np.integer)) or not 0<=max_faults<=n or not callable(decode_fn):
        raise ValueError("invalid truncation or decoder")
    lo,hi=_gain_bounds(gain_interval)
    basis={}
    def mask(v):return sum(int(bit)<<i for i,bit in enumerate(v))
    for row in h:
        v=mask(row)
        while v:
            pivot=v.bit_length()-1
            if pivot in basis:v^=basis[pivot]
            else:basis[pivot]=v;break
    def remainder(v):
        for pivot in sorted(basis,reverse=True):
            if v>>pivot&1:v^=basis[pivot]
        return v
    cache={};counts=[[] for _ in range(max_faults+1)];terms=[[] for _ in counts]
    for order in range(max_faults+1):
        for support in itertools.combinations(range(n),order):
            e=np.zeros(n,dtype=np.int64);e[list(support)]=1;s=(d@e)%2;key=tuple(s)
            if key not in cache:
                try:
                    raw,corrections=decode_fn(s.copy())
                    if any(len(pair)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in pair) or pair[1]<=0 for pair in raw):raise ValueError
                    edges=[Fraction(int(a),int(b)) for a,b in raw]
                    corrections=np.asarray(corrections)
                    if len(edges)<2 or edges[0]!=lo or edges[-1]!=hi or any(a>=b for a,b in zip(edges,edges[1:])):raise ValueError
                    if corrections.shape!=(len(edges)-1,n) or not np.all((corrections==0)|(corrections==1)):raise ValueError
                    corrections=corrections.astype(np.int64)
                    if not np.all((corrections@d.T)%2==s):raise ValueError
                except (TypeError,ValueError,ZeroDivisionError):
                    raise ValueError("invalid decoder profile") from None
                cache[key]=[(b-a,remainder(mask(c[:nd]))) for a,b,c in zip(edges,edges[1:],corrections)]
            residual_class=remainder(mask(e[:nd]))
            fraction=sum((length for length,cls in cache[key] if cls!=residual_class),Fraction(0))/(hi-lo)
            mass=float(np.prod(np.where(e,p,1-p)))
            counts[order].append(float(fraction));terms[order].append(float(fraction)*mass)
    return np.array([math.fsum(x) for x in counts]),np.array([math.fsum(x) for x in terms])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """Return explicit differential cases for Studio contract inspection."""
    return [{'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),decoder(),3,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),decoder(),3,((1,2),(3,2)))',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),decoder(),0,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),decoder(),0,((1,2),(3,2)))',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),constant_decoder(),4,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),HZ.copy(),P.copy(),constant_decoder(),4,((1,2),(3,2)))',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),HZ.copy(),np.array([1.,0.,.2,0.]),decoder(),2,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),HZ.copy(),np.array([1.,0.,.2,0.]),decoder(),2,((1,2),(3,2)))',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),np.vstack([HZ,HZ,np.zeros_like(HZ)]),P.copy(),decoder(),2,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),np.vstack([HZ,HZ,np.zeros_like(HZ)]),P.copy(),decoder(),2,((1,2),(3,2)))',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               'D=np.array([[1,1,0,1],[0,0,1,0]])\n'
               'HZ=np.array([[1,1,0]])\n'
               'P=np.array([.08,.11,.06,.03])\n'
               'def decoder():\n'
               '    def f(s):\n'
               '        a=np.array([0,0,s[1],s[0]],dtype=int)\n'
               '        b=a.copy();b[0]^=1;b[3]^=1\n'
               '        return ((1,2),(4,5),(6,5),(3,2)),np.array([a,b,a])\n'
               '    return f\n'
               'def constant_decoder():\n'
               '    return lambda s: (((1,2),(3,2)),np.array([[0,0,s[1],s[0]]],dtype=int))\n',
      'call': 'truncated_failure_probability(D.copy(),np.empty((0,3),dtype=int),P.copy(),decoder(),2,((1,2),(3,2)))',
      'gold_call': '_oracle_truncated_failure_probability(D.copy(),np.empty((0,3),dtype=int),P.copy(),decoder(),2,((1,2),(3,2)))',
      'tol': 1e-11}]
