"""
Fit the response-adjusted coherence curve to the constrained decay dictionary.

The source method uses an oscillatory localization profile. The benchmark prescribes the frequency, linearized interaction-uncertainty target, amplitude cone and finite rate dictionaries as its fit configuration.

Returns
-------
return fit
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_decay_dictionary(distances: 'np.ndarray | list | tuple', gamma: 'np.ndarray | list | tuple', frequency: float, rates: 'np.ndarray | list | tuple', oscillation_rates: 'np.ndarray | list | tuple', radius: float) -> 'np.ndarray':
    """Parameters
    ----------
    distances : array_like
        Finite real vector of at least two nonnegative separations.
    gamma : array_like, shape (3,len(distances))
        Finite real rows gamma, gamma'_+, gamma'_-, with base gamma>=0. The fitted target is y=max(0,gamma+radius*min(0,gamma'_+,gamma'_-)) entrywise.
    frequency : float
        Finite cycles per spacing in [0,1].
    rates, oscillation_rates : array_like
        Nonempty finite positive dictionaries with at least one pair k_o>=k. Order and repetitions have no effect.
    radius : float
        Finite nonnegative linearization radius.
    
    Returns
    -------
    fit : real ndarray, shape (5,)
        [xi,xi_O,A,B,r], xi=1/k, xi_O=1/k_o. Globally minimize r=||y-A exp(-k*x)-B cos(2*pi*frequency*x) exp(-k_o*x)||_2/||y||_2 subject to k_o>=k and A>=|B|. Rate pairs within 1e-12 of the minimum r tie by smaller k then smaller k_o. Minimum A^2+B^2 resolves nonunique fixed-rate amplitudes. If y is identically zero, r=0 and the same ties apply. Invalid domains raise ValueError."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fit_decay_dictionary(distances: 'np.ndarray | list | tuple', gamma: 'np.ndarray | list | tuple', frequency: float, rates: 'np.ndarray | list | tuple', oscillation_rates: 'np.ndarray | list | tuple', radius: float) -> 'np.ndarray':
    x = np.asarray(distances, dtype=float)
    g=np.asarray(gamma,dtype=float)
    if g.shape!=(3,len(x)) or not np.isfinite(g).all() or np.any(g[0]<0) or not np.isfinite(radius) or radius<0:
        raise ValueError('finite curve jet with nonnegative base and radius required')
    y=np.maximum(0.,g[0]+radius*np.minimum(0.,np.minimum(g[1],g[2])))
    ks = np.asarray(rates, dtype=float)
    os = np.asarray(oscillation_rates, dtype=float)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
        raise ValueError('distances and nonzero nonnegative gamma must be matching finite vectors')
    if not np.isfinite(frequency) or not 0 <= frequency <= 1:
        raise ValueError('frequency must lie in [0, 1]')
    if any(v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() or np.any(v <= 0) for v in (ks, os)) or np.max(os) < np.min(ks):
        raise ValueError('positive finite dictionaries must contain an admissible rate pair')
    scale = np.max(y)
    if scale==0:
        k=float(np.min(ks));ko=float(np.min(os[os>=k]))
        return np.array([1/k,1/ko,0.,0.,0.])
    normalized = y / scale
    records = []
    for k in sorted(set(ks.tolist())):
        for ko in sorted(set(os.tolist())):
            if ko < k:
                continue
            u = np.exp(-k * x)
            v = np.cos(2 * np.pi * frequency * x) * np.exp(-ko * x)
            D = np.column_stack((u + v, u - v))
            # p=(A+B)/2, q=(A-B)/2 makes A>=|B| equivalent to p,q>=0.
            unconstrained = np.linalg.lstsq(D, normalized, rcond=None)[0]
            candidates = [unconstrained] if np.all(unconstrained >= 0) else []
            for j in range(2):
                z = np.zeros(2)
                norm = D[:, j] @ D[:, j]
                z[j] = max(0., float(D[:, j] @ normalized / norm)) if norm > 0 else 0.
                candidates.append(z)
            candidates.append(np.zeros(2))
            # A feasible least-squares solution is already globally optimal;
            # retain its minimum-norm choice when the design is rank deficient.
            p = unconstrained if np.all(unconstrained >= 0) else min(candidates, key=lambda z: (float(np.sum((D @ z - normalized) ** 2)), float(z @ z)))
            residual = float(np.linalg.norm(D @ p - normalized) / np.linalg.norm(normalized))
            records.append((residual, k, ko, (p[0] + p[1]) * scale, (p[0] - p[1]) * scale))
    minimum = min(r[0] for r in records)
    best = min((r for r in records if r[0] <= minimum + 1e-12), key=lambda r: (r[1], r[2]))
    residual, k, ko, amplitude, oscillation_amplitude = best
    return np.array([1 / k, 1 / ko, amplitude, oscillation_amplitude, residual], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n# pure monotone envelope\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=2.0*np.exp(-0.2*x)+0.0*np.cos(2*np.pi*0.5*x)*np.exp(-0.4*x)\nargs=(x,y,0.5,[0.1, 0.2, 0.3],[0.2, 0.4, 0.8])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# positive two-kF oscillation\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=1.0*np.exp(-0.15*x)+0.7*np.cos(2*np.pi*0.5*x)*np.exp(-0.6*x)\nargs=(x,y,0.5,[0.1, 0.15, 0.3],[0.3, 0.6, 1.0])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# negative two-kF oscillation\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=1.0*np.exp(-0.3*x)+-0.65*np.cos(2*np.pi*0.5*x)*np.exp(-0.6*x)\nargs=(x,y,0.5,[0.1, 0.3, 0.5],[0.3, 0.6, 1.0])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# positive cone face\nx=np.array([1, 2, 3, 4, 5],dtype=float)\ny=1.0*np.exp(-0.2*x)+1.0*np.cos(2*np.pi*0.5*x)*np.exp(-0.4*x)\nargs=(x,y,0.5,[0.1, 0.2, 0.5],[0.2, 0.4, 0.8])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# negative cone face\nx=np.array([1, 2, 3, 4, 5],dtype=float)\ny=1.0*np.exp(-0.2*x)+-1.0*np.cos(2*np.pi*0.5*x)*np.exp(-0.4*x)\nargs=(x,y,0.5,[0.1, 0.2, 0.5],[0.2, 0.4, 0.8])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# quarter-filled oscillation\nx=np.array([1, 2, 3, 4, 5, 6],dtype=float)\ny=2.0*np.exp(-0.2*x)+0.9*np.cos(2*np.pi*0.25*x)*np.exp(-0.4*x)\nargs=(x,y,0.25,[0.1, 0.2, 0.3],[0.2, 0.4, 0.6])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# zero filling rank deficiency\nx=np.array([1, 2, 3, 4],dtype=float)\ny=1.0*np.exp(-0.2*x)+0.5*np.cos(2*np.pi*0.0*x)*np.exp(-0.2*x)\nargs=(x,y,0.0,[0.2],[0.2])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# irregular distance sampling\nx=np.array([0.5, 1.2, 2.7, 4.1, 7.0],dtype=float)\ny=1.3*np.exp(-0.15*x)+-0.4*np.cos(2*np.pi*0.37*x)*np.exp(-0.3*x)\nargs=(x,y,0.37,[0.1, 0.15, 0.2],[0.15, 0.3, 0.5])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# off-dictionary physical decay\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=1.0*np.exp(-0.173*x)+0.6*np.cos(2*np.pi*0.5*x)*np.exp(-0.47*x)\nargs=(x,y,0.5,[0.1, 0.15, 0.2],[0.2, 0.4, 0.6])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# widely separated decay scales\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=0.4*np.exp(-0.025*x)+0.35*np.cos(2*np.pi*0.5*x)*np.exp(-2.0*x)\nargs=(x,y,0.5,[0.025, 0.1, 0.3],[0.1, 0.4, 2.0])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# oscillatory-rate ordering constraint\nx=np.array([1, 2, 3, 4, 5],dtype=float)\ny=1.0*np.exp(-0.4*x)+0.1*np.cos(2*np.pi*0.25*x)*np.exp(-0.1*x)\nargs=(x,y,0.25,[0.1, 0.2, 0.4],[0.1, 0.2, 0.4])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# small amplitude scale invariance\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=1e-90*np.exp(-0.2*x)+-4e-91*np.cos(2*np.pi*0.5*x)*np.exp(-0.6*x)\nargs=(x,y,0.5,[0.1, 0.2, 0.3],[0.2, 0.6, 1.0])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# even-only rank deficient sampling\nx=np.array([2, 4, 6, 8],dtype=float)\ny=1.0*np.exp(-0.2*x)+-0.3*np.cos(2*np.pi*0.5*x)*np.exp(-0.2*x)\nargs=(x,y,0.5,[0.2],[0.2])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# unordered repeated dictionary entries\nx=np.array([2, 3, 4, 5, 6],dtype=float)\ny=1.0*np.exp(-0.2*x)+0.2*np.cos(2*np.pi*0.5*x)*np.exp(-0.4*x)\nargs=(x,y,0.5,[0.3, 0.2, 0.1, 0.2],[0.8, 0.4, 0.2, 0.4])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# zero entries at alternating separations\nargs=([1,2,3,4,5], [0.,.5,0.,.2,0.], .5, [.1,.3,.5], [.1,.3,.5,1.])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\n# cone projection of noisy nonnegative data\nargs=([1,2,3,4,5,6], [.8,.01,.5,.01,.3,.01], .5, [.1,.2,.4], [.2,.4,.8])\nargs=tuple(np.asarray(value) if i in [0, 1, 3, 4] else value for i,value in enumerate(args))\nfrom copy import deepcopy\nx,y,f,k,o=args;y=np.asarray(y);args=(x,np.stack((y,np.zeros_like(y),np.zeros_like(y))),f,k,o,.1)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# finite response changes envelope dictionary winner\nargs=([1.,2.,3.,4.],[[0.9, 0.6, 0.4, 0.3], [0.2, 0.3, 0.1, 0.2], [-0.2, -0.3, -0.1, -0.2]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.5)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# both directions positive leave robust base\nargs=([1.,2.,3.,4.],[[0.9, 0.6, 0.4, 0.3], [0.2, 0.3, 0.1, 0.2], [0.1, 0.1, 0.2, 0.2]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.5)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# directional branch changes with separation\nargs=([1.,2.,3.,4.],[[0.9, 0.6, 0.4, 0.3], [-0.2, 0.3, -0.1, 0.2], [0.1, -0.2, 0.2, -0.3]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.7)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# robust linearization reaches zero at selected separations\nargs=([1.,2.,3.,4.],[[0.9, 0.1, 0.4, 0.03], [-0.2, -0.3, -0.1, -0.2], [0.1, 0.1, 0.2, 0.2]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.8)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# zero radius recovers physical curve\nargs=([1.,2.,3.,4.],[[0.9, 0.6, 0.4, 0.3], [-2.0, -3.0, -1.0, -2.0], [0.1, 0.1, 0.2, 0.2]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.0)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# large asymmetric response stays inside positive cone\nargs=([1.,2.,3.,4.],[[0.8, 0.01, 0.5, 0.01], [-0.1, -0.01, -0.2, -0.02], [0.1, 0.03, 0.2, 0.05]],.3333333333333333,[.1,.2,.4],[.2,.4,.8],0.5)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}, {'setup': 'import numpy as np\nfrom copy import deepcopy\n# zero robust coherence selects declared rate tie\nargs=([1.,2.,3.],[[.1,.2,.3],[-1.,-1.,-1.],[1.,1.,1.]],.3,[.2,.1],[.4,.2],1.)', 'call': 'fit_decay_dictionary(*deepcopy(args))', 'gold_call': '_oracle_fit_decay_dictionary(*deepcopy(args))'}]
