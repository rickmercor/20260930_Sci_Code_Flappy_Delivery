"""
Find the common-order worst-scenario finite-block rate and its comparators.

Section V compares the direct sandwiched estimate with the continuity expansion and AEP. The common order across a finite channel family is a constructed extension. R_inf includes privacy amplification and leakage; R_down is evaluated at the chosen active scenario and t; B and AEP are robust over the entire scenario family.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def robust_rate(states: np.ndarray, channels: np.ndarray, statistics: np.ndarray, n: int, epsilon: float, epsilon_prime: float) -> np.ndarray:
    """Find the common-order worst-scenario finite-block rate and its comparators.

    states : ndarray, shape (S,N,N), complex
        The S conditional states rho_(E|Y=0), in scenario order, N=2 or 4.
    channels : ndarray, shape (S,N,N), float
        Corresponding optical transition matrices P[y,x].
    statistics : ndarray, shape (S,3), float
        Each row [H(Y|E), V(Y|E), H_2_down(Y|E)] from entropy_statistics.
    n : int
        Positive total block size in transmitted signals.
    epsilon, epsilon_prime : float
        Smoothing and extraction parameters, each in (0,1).
    Returns
    -------
    ndarray, shape (8,), float
        [R_star, t_star, active_scenario, h_star, R_down, R_B, R_AEP, R_inf].
        All rates are signed bits per signal; h_star is bits; t_star in [0,0.98].
        R_star maximizes the minimum scenario direct rate using one common t.
        At t=0 use min_entropy_optimum. Otherwise use sandwiched_optimum.
        Choose the first scenario within 1e-7 of the minimum; choose the least t
        on a maximizing interval. R_B optimizes one common 1<a<2 independently;
        R_AEP and R_inf are scenario minima at the same modulation setting.
        The global background defines all rate and comparator equations.
        Rate tolerance 5e-7, t tolerance 2e-5, comparison tolerances 5e-5.
    Raises
    ------
    ValueError
        If arrays have incompatible shapes, a state is invalid, or block/security
        parameters are outside their domains.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def _sand_loss(state, sigma, reciprocal_order, gradient=False):
    t=reciprocal_order;d=np.exp((t-1)*np.log(sigma)/2)
    B=d[:,None]*state*d[None,:]
    e,u=np.linalg.eigh(B);e=np.maximum(e,1e-300)
    logs=np.log(e)/t;L=logsumexp(logs)
    value=t/(1-t)*L/np.log(2)
    if not gradient:return value
    diagonal=np.abs(u)**2@np.exp(logs-L)
    return value,-diagonal/sigma/np.log(2)

def _oracle_robust_rate(states: np.ndarray, channels: np.ndarray, statistics: np.ndarray, n: int, epsilon: float, epsilon_prime: float) -> np.ndarray:
    rs=np.asarray(states,complex);ps=np.asarray(channels,float);st=np.asarray(statistics,float)
    if rs.ndim!=3 or rs.shape[1]!=rs.shape[2] or rs.shape[1] not in (2,4) or ps.shape!=rs.shape or st.shape!=(len(rs),3) or len(rs)==0 or not np.isfinite(ps).all() or not np.isfinite(st).all() or not np.isfinite([n,epsilon,epsilon_prime]).all() or n<1 or n!=int(n) or not 0<epsilon<1 or not 0<epsilon_prime<1:
        raise ValueError('incompatible arrays or invalid block/security parameters')
    if np.any(ps < 0) or not np.allclose(ps.sum(axis=1), 1, rtol=0, atol=1e-10) or not np.allclose(ps.sum(axis=2), 1, rtol=0, atol=1e-10):
        raise ValueError('channels must be nonnegative doubly stochastic probability matrices')
    N=rs.shape[1];S=len(rs)
    for r in rs:_checked_state(r)
    leak=np.array([-np.sum(p[p>0]*np.log2(p[p>0]))/N for p in ps])
    g=np.log2(1+np.sqrt((1-epsilon)*(1+epsilon)))-2*np.log2(epsilon)
    extract=(1+2*np.log2(epsilon_prime))/n
    mins=np.array([_oracle_min_entropy_optimum(r)[0] for r in rs])
    cache={}
    def _values(t):
        key=float(t)
        if key not in cache:
            h=mins if key==0 else np.array([_oracle_sandwiched_optimum(r,key)[0] for r in rs])
            cache[key]=(h,h+extract-g*key/(n*(1-key))-leak)
        return cache[key]
    grid=np.r_[0,np.linspace(.02,.98,33)]
    vals=np.array([np.min(_values(t)[1]) for t in grid])
    candidates=[(vals[i],grid[i]) for i in [0,len(grid)-1]]
    for i in range(1,len(grid)-1):
        if i==1 or (vals[i]>=vals[i-1] and vals[i]>=vals[i+1]):
            fit=minimize_scalar(lambda t:-min(_values(t)[1]),bounds=(max(1e-7,grid[i-1]),grid[i+1]),method='bounded',options={'xatol':1e-9})
            candidates.append((-fit.fun,fit.x))
    value,t=max(candidates,key=lambda z:(z[0],-z[1]))
    # Refine scenario intersections around the selected point.
    if t>0:
        for i in range(S):
            for j in range(i):
                a=max(1e-6,t-.002);b=min(.98,t+.002)
                _fun=lambda u:_values(u)[1][i]-_values(u)[1][j]
                if _fun(a)*_fun(b)<0:
                    root=brentq(_fun,a,b,xtol=1e-11);v=min(_values(root)[1])
                    if v>value-1e-11 and abs(root-t)<1e-5:value,t=v,root
    H,R=_values(t);j=int(np.flatnonzero(R<=np.min(R)+1e-7)[0]);r=rs[j];h,V,h2=st[j]
    active=r.diagonal().real>1e-15;rr=r[np.ix_(active,active)];w=rr.diagonal().real;w=w/w.sum();rr=rr/np.trace(rr).real
    if t==0:
        D=np.diag(w**-.5);down=np.log2(N)-np.log2(np.linalg.eigvalsh(D@rr@D).max())
    else:down=np.log2(N)-_sand_loss(rr,w,t)
    Rdown=down+extract-g*t/(n*(1-t))-leak[j]
    delta=4*np.log2(2+np.sqrt(N))*np.sqrt(np.log2(2)-2*np.log2(epsilon))
    RAEP=float(np.min(st[:,0]+extract-delta/np.sqrt(n)-leak))
    # The continuity expansion is a scientifically distinct comparator, optimized on (1,2).
    def _continuity(a,k):
        rk=rs[k];wk=rk.diagonal().real;ac=wk>1e-15;rk=rk[np.ix_(ac,ac)];wk=wk[ac];rk/=wk.sum();wk/=wk.sum()
        e,u=np.linalg.eigh(rk);e=np.maximum(e,0)
        trace=np.sum((np.abs(u)**2@e**a)*wk**(1-a))
        hp=np.log2(N)+np.log2(trace)/(1-a)
        hk,vk,h2k=st[k]
        logK=(a-1)*(-hp+hk)*np.log(2)+3*np.log(np.log(2**(-h2k+hk)+np.exp(2)))-np.log(6*np.log(2))-3*np.log(2-a)
        return hk-(a-1)*np.log(2)*vk/2-(a-1)**2*np.exp(logK)+extract-g/(n*(a-1))-leak[k]
    fit=minimize_scalar(lambda a:-min(_continuity(a,k) for k in range(S)),bounds=(1.00001,1.99),method='bounded',options={'xatol':2e-9})
    RB=-fit.fun
    return np.array([np.min(R),t,j,H[j],Rdown,RB,RAEP,np.min(mins+extract-leak)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [
        {'setup': 'import numpy as np\nN=2; A=1.17; sc=np.array([[0.84, 0.03]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),180,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),180,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=4; A=1.27; sc=np.array([[0.88, 0.01], [0.896, 0.73]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),420,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),420,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=2; A=0.7; sc=np.array([[1.0, 0.2]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),1300,1e-07,1e-09)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),1300,1e-07,1e-09)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=4; A=1.61; sc=np.array([[0.79, -0.28]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),710,1e-06,1e-09)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),710,1e-06,1e-09)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=4; A=1.31; sc=np.array([[0.87, 0.1], [0.91, 0.51]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),17000,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),17000,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=2; A=0.86; sc=np.array([[0.91, 1.5707963267948966]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),240,1e-09,1e-07)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),240,1e-09,1e-07)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=4; A=1.19; sc=np.array([[0.9, 0.32], [0.9, -0.32]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),870,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),870,1e-08,1e-08)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
        {'setup': 'import numpy as np\nN=4; A=1.28; sc=np.array([[0.93, 0.41], [0.885, 0.03], [0.9, 0.69]],float)\nps=np.array([_oracle_psk_channel(N,A,eta,phase) for eta,phase in sc])\nrs=np.array([_oracle_reverse_states(P,_oracle_cyclic_weights(N,(1-eta)*A*A))[0]\n             for P,(eta,phase) in zip(ps,sc)])\nss=np.array([_oracle_entropy_statistics(r) for r in rs])\n', 'call': '(robust_rate(rs.copy(),ps.copy(),ss.copy(),550,1e-05,1e-10)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'gold_call': '(_oracle_robust_rate(rs.copy(),ps.copy(),ss.copy(),550,1e-05,1e-10)) * np.array([1.,.025,1.,.01,.01,.01,.01,.01])', 'tol': 5e-07},
    ]
