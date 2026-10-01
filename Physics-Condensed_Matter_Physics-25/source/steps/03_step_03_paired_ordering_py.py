"""
Construct a matching-preserving, fill-aware initial permutation.

Main paper Section III.B pairs vertices by a maximum-cardinality matching of the nonzero graph, then reorders the compressed pair graph without splitting the pairs. This small-system adaptation uses exact minimum-current-degree elimination with fill instead of METIS nested dissection. Any maximum-cardinality matching is accepted; sort each matched pair internally, pair unmatched vertices in ascending order, and break equal quotient degrees lexicographically by the original sorted vertex pair.

Returns
-------
return result  # integer ndarray (n,), a permutation. Consecutive entries are ascending vertex pairs. Pair order is minimum-current-degree elimination on the quotient graph, with clique fill at each elimination and lexicographic pair ties. Any maximum-cardinality matching is valid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def paired_ordering(skew: ArrayLike) -> np.ndarray:
    'Construct a matching-preserving, fill-aware initial permutation.\n\nParameters\n----------\nskew : real (n,n), finite skew-symmetric matrix of positive even size; graph edges mean exact nonzero entries.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12; the matrix has positive even size.\n\nReturns\n-------\ninteger ndarray (n,), a permutation. Consecutive entries are ascending vertex pairs. Pair order is minimum-current-degree elimination on the quotient graph, with clique fill at each elimination and lexicographic pair ties. Any maximum-cardinality matching is valid.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_paired_ordering(skew: ArrayLike) -> np.ndarray:
    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    a=_checked_skew(skew)
    # Maximum-cardinality graph matching by Edmonds alternating forests.
    n=len(a);graph=[list(np.flatnonzero(a[i]!=0)) for i in range(n)]
    match=[-1]*n
    for root in range(n):
        if match[root]!=-1:continue
        parent=[-1]*n;base=list(range(n));used=[False]*n;queue=[root];used[root]=True;found=-1
        def _lca(v,w):
            seen=[False]*n
            while True:
                v=base[v];seen[v]=True
                if match[v]==-1:break
                v=parent[match[v]]
            while not seen[base[w]]:w=parent[match[base[w]]]
            return base[w]
        def _mark(v,b,child,blossom):
            while base[v]!=b:
                blossom[base[v]]=blossom[base[match[v]]]=True
                parent[v]=child;child=match[v];v=parent[match[v]]
        for v in queue:
            for w in graph[v]:
                if base[v]==base[w] or match[v]==w:continue
                if w==root or (match[w]!=-1 and parent[match[w]]!=-1):
                    b=_lca(v,w);blossom=[False]*n
                    _mark(v,b,w,blossom);_mark(w,b,v,blossom)
                    for i in range(n):
                        if blossom[base[i]]:
                            base[i]=b
                            if not used[i]:used[i]=True;queue.append(i)
                elif parent[w]==-1:
                    parent[w]=v
                    if match[w]==-1:found=w;break
                    w=match[w];used[w]=True;queue.append(w)
            if found!=-1:break
        while found!=-1:
            v=parent[found];nxt=match[v] if v!=-1 else -1
            match[found]=v
            if v!=-1:match[v]=found
            found=nxt
    pairs=[(i,match[i]) for i in range(n) if match[i]>i]
    free=[i for i in range(n) if match[i]==-1]
    pairs+=list(zip(free[::2],free[1::2]));pairs=sorted(pairs)
    m=len(pairs);adj=[set() for _ in range(m)]
    for i in range(m):
        for j in range(i+1,m):
            if np.any(a[np.ix_(pairs[i],pairs[j])]!=0):adj[i].add(j);adj[j].add(i)
    remaining=set(range(m));order=[]
    while remaining:
        k=min(remaining,key=lambda j:(len(adj[j]&remaining),pairs[j]))
        nbr=adj[k]&remaining
        for j in nbr:adj[j].update(nbr-{j})
        remaining.remove(k);order.extend(pairs[k])
    return np.asarray(order,int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 0.0], [0.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 0.0, -5.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, 5.0, 0.0], [-3.0, 0.0, 0.0, 3.0], [-5.0, 0.0, 0.0, 0.0], [0.0, -3.0, 0.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, -3.0, 2.0, 0.0, 0.0], [-3.0, 0.0, 1.0, 0.0, 5.0, 0.0], [3.0, -1.0, 0.0, 0.0, 0.0, 3.0], [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -5.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, 0.0, 0.0, -5.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 4.0, 0.0, 3.0, 0.0, 0.0], [0.0, 0.0, -4.0, 0.0, 2.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, 5.0, 2.0, 4.0, 1.0], [-3.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-5.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-4.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 3.0, 5.0], [0.0, 0.0, 0.0, 0.0, 5.0, 2.0, 4.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 5.0, 2.0], [0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 1.0, 3.0], [-4.0, -5.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0, 0.0], [-3.0, -4.0, -5.0, -1.0, 0.0, 0.0, 0.0, 0.0], [-5.0, -1.0, -2.0, -3.0, 0.0, 0.0, 0.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}, {'setup': 'import numpy as np\ndef _ordering_metrics(s, output):\n    p=np.asarray(output);n=len(s)\n    if p.shape!=(n,) or not np.all(np.isfinite(p)) or not np.all(p==np.rint(p)):\n        return np.array([0.,-1.,0.])\n    p=p.astype(int)\n    if not np.array_equal(np.sort(p),np.arange(n)):\n        return np.array([0.,-1.,0.])\n    pairs=[tuple(p[i:i+2]) for i in range(0,n,2)]\n    if any(a>=b for a,b in pairs):return np.array([0.,-1.,0.])\n    ordered=sorted(pairs);m=len(ordered);adj=[set() for _ in range(m)]\n    for i in range(m):\n        for j in range(i+1,m):\n            if np.any(s[np.ix_(ordered[i],ordered[j])]!=0):adj[i].add(j);adj[j].add(i)\n    live=set(range(m));canonical=[]\n    while live:\n        v=min(live,key=lambda i:(len(adj[i]&live),ordered[i]));nbr=adj[v]&live\n        for i in nbr:adj[i].update(nbr-{i})\n        live.remove(v);canonical.append(ordered[v])\n    return np.array([1.,sum(s[a,b]!=0 for a,b in pairs),float(pairs==canonical)])\n\ns=np.array([[0.0, 3.0, 5.0, 0.0, 4.0, 1.0, 0.0, 5.0, 2.0, 0.0], [-3.0, 0.0, 0.0, 3.0, 5.0, 0.0, 4.0, 1.0, 0.0, 5.0], [-5.0, 0.0, 0.0, 4.0, 0.0, 3.0, 5.0, 0.0, 4.0, 1.0], [0.0, -3.0, -4.0, 0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 0.0], [-4.0, -5.0, 0.0, -2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 3.0], [-1.0, 0.0, -3.0, -4.0, 0.0, 0.0, 3.0, 0.0, 2.0, 4.0], [0.0, -4.0, -5.0, 0.0, -2.0, -3.0, 0.0, 1.0, 3.0, 0.0], [-5.0, -1.0, 0.0, -3.0, -4.0, 0.0, -1.0, 0.0, 0.0, 1.0], [-2.0, 0.0, -4.0, -5.0, 0.0, -2.0, -3.0, 0.0, 0.0, 2.0], [0.0, -5.0, -1.0, 0.0, -3.0, -4.0, 0.0, -1.0, -2.0, 0.0]],dtype=float)', 'call': '_ordering_metrics(s,paired_ordering(s))', 'gold_call': '_ordering_metrics(s,_oracle_paired_ordering(s))'}]
