#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_smoothed_payoff(K: float, S_max: float, epsilon: float, M: int) -> "np.ndarray":
    import numpy as np
    if not (isinstance(K, (int, float)) and float(K) > 0.0): raise ValueError("K must be positive")
    if not (isinstance(S_max, (int, float)) and float(S_max) > 0.0): raise ValueError("S_max must be positive")
    if not (isinstance(epsilon, (int, float)) and float(epsilon) > 0.0): raise ValueError("epsilon must be positive")
    if not (isinstance(M, (int, np.integer)) and int(M) >= 2): raise ValueError("M must be an integer >= 2")
    K=float(K); S_max=float(S_max); epsilon=float(epsilon); M=int(M)
    s_grid=np.arange(M+1,dtype=np.float64)*(S_max/M)
    x=s_grid-K
    e1=35.0*epsilon/256.0; e2=0.5; e3=35.0/(64.0*epsilon); e5=-35.0/(128.0*epsilon**3); e7=7.0/(64.0*epsilon**5); e9=5.0/(256.0*epsilon**7)
    polynomial=e1+e2*x+e3*x**2+e5*x**4+e7*x**6+e9*x**8
    payoff=np.where(x<=-epsilon,0.0,np.where(x>=epsilon,x,polynomial))
    return np.column_stack((s_grid,payoff.astype(np.float64)))

def compute_nsfd_coefficients(r: float, sigma: float, T: float, S_max: float, M: int, N: int) -> "np.ndarray":
    import numpy as np
    if not (isinstance(r,(int,float)) and float(r)>0): raise ValueError("r must be positive")
    if not (isinstance(sigma,(int,float)) and float(sigma)>0): raise ValueError("sigma must be positive")
    if not (isinstance(T,(int,float)) and float(T)>0): raise ValueError("T must be positive")
    if not (isinstance(S_max,(int,float)) and float(S_max)>0): raise ValueError("S_max must be positive")
    if not (isinstance(M,(int,np.integer)) and int(M)>=2): raise ValueError("M must be an integer >= 2")
    if not (isinstance(N,(int,np.integer)) and int(N)>=1): raise ValueError("N must be an integer >= 1")
    r=float(r); sigma=float(sigma); T=float(T); M=int(M); N=int(N)
    D=0.5*sigma**2; alpha=r/D; delta_t=T/N; psi_1=float(np.exp(r*delta_t)-1.0)
    if not np.isfinite(psi_1) or psi_1<=0: raise ValueError("invalid temporal denominator")
    B_left=np.empty(M-1,dtype=np.float64); B_center=np.empty(M-1,dtype=np.float64); B_right=np.empty(M-1,dtype=np.float64)
    for idx,m in enumerate(range(1,M)):
        A1=(m+1.0)/(m**alpha)-m/((m+1.0)**alpha)
        A2=(m+2.0)/(m**alpha)-m/((m+2.0)**alpha)
        A3=(m+2.0)/((m+1.0)**alpha)-(m+1.0)/((m+2.0)**alpha)
        q=A2-A1-A3
        if q<=0 or not np.isfinite(q): raise ValueError("reference coefficient denominator is invalid")
        B_left[idx]=(A2-A1)/q-1.0; B_center[idx]=-A2/q-1.0/psi_1; B_right[idx]=A1/q
    packed=np.empty((M-1,4),dtype=np.float64)
    packed[:,0]=B_left; packed[:,1]=B_center; packed[:,2]=B_right; packed[:,3]=psi_1
    return packed

def assemble_nsfd_timestep(previous: "np.ndarray", coefficients: "np.ndarray", S_max: float, K: float, r: float, t: float, M: int) -> "np.ndarray":
    import numpy as np
    previous=np.asarray(previous,dtype=np.float64); coefficients=np.asarray(coefficients,dtype=np.float64)
    if not (isinstance(M,(int,np.integer)) and int(M)>=2): raise ValueError("M must be an integer >= 2")
    M=int(M)
    if previous.shape!=(M+1,): raise ValueError("previous must have shape (M+1,)")
    if coefficients.shape!=(M-1,4): raise ValueError("coefficients must have shape (M-1,4)")
    B_left=coefficients[:,0]; B_center=coefficients[:,1]; B_right=coefficients[:,2]; psi_1=float(coefficients[0,3])
    if not (np.isfinite(psi_1) and psi_1>0) or not np.allclose(coefficients[:,3],psi_1,rtol=0.0,atol=0.0): raise ValueError("invalid temporal coefficient column")
    if S_max<=0 or K<=0 or r<=0 or t<=0: raise ValueError("physical inputs must be positive")
    g=float(S_max-K*np.exp(-r*t))
    lower=B_left[1:].copy(); diagonal=B_center.copy(); upper=B_right[:-1].copy(); rhs=-previous[1:M]/psi_1; rhs[-1]-=B_right[-1]*g
    packed=np.zeros((M-1,4),dtype=np.float64)
    packed[:,0]=np.r_[lower,0.0]; packed[:,1]=diagonal; packed[:,2]=np.r_[upper,0.0]; packed[:,3]=rhs
    return packed

def solve_nsfd_surface(initial: "np.ndarray", coefficients: "np.ndarray", r: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    import numpy as np
    from scipy.linalg import solve_banded
    initial=np.asarray(initial,dtype=np.float64); coefficients=np.asarray(coefficients,dtype=np.float64)
    if not (isinstance(M,(int,np.integer)) and int(M)>=2): raise ValueError("M must be an integer >= 2")
    if not (isinstance(N,(int,np.integer)) and int(N)>=1): raise ValueError("N must be an integer >= 1")
    M=int(M); N=int(N)
    if initial.shape!=(M+1,2): raise ValueError("initial must have shape (M+1,2)")
    if coefficients.shape!=(M-1,4): raise ValueError("coefficients must have shape (M-1,4)")
    if not (r>0 and T>0 and K>0 and S_max>K): raise ValueError("invalid physical or domain parameters")
    dt=T/N
    solution=np.empty((N+1,M+1),dtype=np.float64); solution[0]=initial[:,1]
    for k in range(1,N+1):
        t=k*dt
        system=assemble_nsfd_timestep(solution[k-1],coefficients,S_max,K,r,t,M)
        n=M-1; banded=np.zeros((3,n),dtype=np.float64); banded[0,1:]=system[:-1,2]; banded[1,:]=system[:,1]; banded[2,:-1]=system[:-1,0]
        interior=solve_banded((1,1),banded,system[:,3])
        solution[k,0]=0.0; solution[k,M]=S_max-K*np.exp(-r*t); solution[k,1:M]=interior
    return solution

def compute_black_scholes_reference(r: float, sigma: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    import numpy as np
    from scipy.special import ndtr
    if not (r>0 and sigma>0 and T>0 and K>0 and S_max>0): raise ValueError("physical parameters must be positive")
    if not (isinstance(M,(int,np.integer)) and int(M)>=1): raise ValueError("M must be an integer >= 1")
    if not (isinstance(N,(int,np.integer)) and int(N)>=1): raise ValueError("N must be an integer >= 1")
    M=int(M); N=int(N)
    s_grid=np.arange(M+1,dtype=np.float64)*(S_max/M); times=np.arange(N+1,dtype=np.float64)*(T/N); reference=np.empty((N+1,M+1),dtype=np.float64)
    for k,tau in enumerate(times):
        for m,s in enumerate(s_grid):
            if tau<=1e-15: reference[k,m]=max(float(s-K),0.0)
            elif s<=0: reference[k,m]=0.0
            else:
                root=np.sqrt(tau); d1=(np.log(s/K)+(r+0.5*sigma**2)*tau)/(sigma*root); d2=d1-sigma*root; reference[k,m]=s*ndtr(d1)-K*np.exp(-r*tau)*ndtr(d2)
    return reference

def compute_max_nodal_error(numerical: "np.ndarray", reference: "np.ndarray") -> float:
    import numpy as np
    numerical=np.asarray(numerical,dtype=np.float64); reference=np.asarray(reference,dtype=np.float64)
    if numerical.ndim!=2 or reference.ndim!=2: raise ValueError("both surfaces must be 2D")
    if numerical.shape!=reference.shape or numerical.size==0: raise ValueError("surfaces must have the same non-empty shape")
    if not (np.all(np.isfinite(numerical)) and np.all(np.isfinite(reference))): raise ValueError("surfaces must be finite")
    return float(np.max(np.abs(numerical-reference)))

def run_nsfd_benchmark(r: float, sigma: float, T: float, K: float, S_max: float, epsilon: float, M: int, N: int) -> float:
    import numpy as np
    initial_table=build_smoothed_payoff(K,S_max,epsilon,M)
    coefficients=compute_nsfd_coefficients(r,sigma,T,S_max,M,N)
    numerical=solve_nsfd_surface(initial_table,coefficients,r,T,K,S_max,M,N)
    reference=compute_black_scholes_reference(r,sigma,T,K,S_max,M,N)
    error=compute_max_nodal_error(numerical,reference)
    if initial_table.shape!=(M+1,2): raise ValueError("upstream payoff stage returned an invalid shape")
    return float(error)
SCICODE_GOLD_EOF
