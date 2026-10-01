"""
Construct the finite-depth nonorthogonal density action and determine its chemical potential from the particle count of the spatially extracted density.

Finite-depth color reconstruction changes the spectral particle weights. At low temperature, the chemical potential in a gap is fixed by balancing thermally activated electrons and holes, even when the separately rounded occupations appear to be exactly zero or one. Unequal weights can shift that balance away from the middle of the gap; a stable logistic alone does not make the count residual reliable.

Returns
-------
tuple (float mu, np.ndarray rho of shape (n,n), np.ndarray X of shape (n,m)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def solve_fixed_population_density(overlap: np.ndarray, overlap_map: np.ndarray, transformed_hamiltonian: np.ndarray, probes: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, beta: float, electron_count: float, depth: int, keep_radius: float) -> tuple[float,np.ndarray,np.ndarray]:
    """Solve the fixed-population compressed density problem.

    Parameters
    ----------
    overlap : np.ndarray
        Finite real symmetric positive-definite S of shape (n,n).
    overlap_map : np.ndarray
        Finite real symmetric reconstructed inverse-square-root map M, (n,n).
    transformed_hamiltonian : np.ndarray
        Finite real symmetric, already extracted orthogonal Hamiltonian K, (n,n).
    probes : np.ndarray
        Unnormalized signed probe block R, (n,m), matching colors and signs exactly.
    distances : np.ndarray
        Finite nonnegative symmetric (n,n) distance matrix, zero diagonal.
    colors : np.ndarray
        Contiguous nonnegative integer-valued labels (n,), covering all m columns.
    signs : np.ndarray
        Real (n,) values +/-1; integer or float dtype accepted.
    beta : float
        Positive finite inverse temperature in eV**(-1). The finite-temperature
        root is required even in a spectral gap: evaluate occupations without
        overflow and the count residual without losing small electron/hole
        contributions to saturation, cancellation, or underflow.
    electron_count : float
        Requested one-spin particle count strictly between zero and the total
        spectral weight, with a strictly bracketing chemical-potential interval.
    depth : int
        Positive depth for the canonical block-Krylov space starting at M R.
    keep_radius : float
        Finite nonnegative inclusive density extraction radius; no magnitude
        threshold is applied to density or to its spectral components.

    Returns
    -------
    chemical_potential : float
        Mu in eV solving the finite-temperature Tr(S rho(mu))=electron_count
        relation to absolute error 1e-10. An edge of a rounded count plateau
        is not an alternative root; no zero-temperature midpoint convention
        or population-residual-only stopping rule replaces this requirement.
    density : np.ndarray
        Symmetric extracted density rho of shape (n,n).
    density_action : np.ndarray
        Unextracted original-coordinate density action X of shape (n,m).

    Raises
    ------
    ValueError
        If shapes, finite-real data, symmetry, positive definiteness of S,
        probe consistency, distance/color/sign constraints, depth, beta,
        radius, spectral weights, or the root bracket are invalid. Spectral
        weights must be nonnegative, have positive total, and bracket the count.

    Notes
    -----
    Use canonical_block_krylov_basis for the depth-limited subspace of K
    starting at M R, and extract_sparse_operator for spatial reconstruction.
    Both sides of the particle constraint refer to the same extracted
    original-coordinate density. Spectral particle weights are the
    overlap-traces of its extracted spectral components, not unit mode
    multiplicities or Euclidean norms. Negative weights are outside the
    monotone-root domain; zero weights are permitted.
    Use [min(eps)-max(2,50/beta), max(eps)+max(2,50/beta)] and 64 bisections,
    moving the lower bound when the mathematical count is below target and
    the upper bound otherwise, then take the midpoint. An equivalent root
    solve agreeing within 1e-10 is valid. Test the strict bracket and each
    comparison with a cancellation-resistant finite-temperature residual;
    rounding a sum of saturated occupations before subtracting the target
    does not define the comparison. Occupations in the returned arrays may
    round to 0 or 1, but their tiny tails still determine the root.
    Build the unextracted action before symmetric spatial extraction. Do not
    apply a magnitude threshold to density or spectral components. Matrix
    symmetry tolerance is absolute 1e-12. Do not mutate the inputs.
    """
    return 0.0,np.zeros(np.asarray(overlap).shape),np.zeros(np.asarray(probes).shape)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_fixed_population_density(overlap: np.ndarray, overlap_map: np.ndarray, transformed_hamiltonian: np.ndarray, probes: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, beta: float, electron_count: float, depth: int, keep_radius: float) -> tuple[float, np.ndarray, np.ndarray]:
    import numpy as np
    from numbers import Real, Integral
    import math
    if any(np.iscomplexobj(a) for a in (overlap,overlap_map,transformed_hamiltonian,probes)):
        raise ValueError("matrices must be real")
    s=np.asarray(overlap,dtype=float);m=np.asarray(overlap_map,dtype=float)
    k=np.asarray(transformed_hamiltonian,dtype=float);r=np.asarray(probes,dtype=float)
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or s.shape[0]<1:
        raise ValueError("overlap must be square")
    n=s.shape[0]
    for a in (s,m,k):
        if a.shape!=(n,n) or not np.all(np.isfinite(a)) or not np.allclose(a,a.T,rtol=0,atol=1e-12):
            raise ValueError("matrices must be finite symmetric and shape-compatible")
    if np.linalg.eigvalsh(s)[0]<=0:
        raise ValueError("overlap must be positive definite")
    if r.ndim!=2 or r.shape[0]!=n or r.shape[1]<1 or not np.all(np.isfinite(r)):
        raise ValueError("probes must be a finite block")
    # Validate the extraction contract before interpreting the labels.
    _oracle_extract_sparse_operator(np.zeros_like(r),distances,colors,signs,keep_radius,0.0)
    c=np.asarray(colors,dtype=int);sig=np.asarray(signs,dtype=float)
    expected=np.zeros_like(r);expected[np.arange(n),c]=sig
    if not np.array_equal(r,expected):
        raise ValueError("probes must match colors and supplied signs without normalization")
    if isinstance(beta,(bool,np.bool_)) or not isinstance(beta,Real) or not np.isfinite(beta) or beta<=0:
        raise ValueError("beta must be positive finite scalar")
    if isinstance(electron_count,(bool,np.bool_)) or not isinstance(electron_count,Real) or not np.isfinite(electron_count):
        raise ValueError("electron_count must be finite real scalar")
    if isinstance(depth,(bool,np.bool_)) or not isinstance(depth,Integral) or depth<1:
        raise ValueError("depth must be positive integer")
    b=m@r
    q=_oracle_canonical_block_krylov_basis(k,b,depth)
    t=q.T@k@q;t=0.5*(t+t.T)
    e,u=np.linalg.eigh(t)
    left=m@q@u;right=u.T@(q.T@b)
    weights=np.empty(e.size)
    for j in range(e.size):
        rank_one=left[:,j,None]*right[j,None,:]
        component=_oracle_extract_sparse_operator(rank_one,distances,c,sig,keep_radius,0.0)
        weights[j]=np.sum(s*component.T)
    total=math.fsum(float(w) for w in weights)
    if not np.all(np.isfinite(weights)) or np.any(weights<0) or total<=0 or not 0<electron_count<total:
        raise ValueError("spectral weights and electron count do not define a monotone root")
    pad=max(2.0,50.0/float(beta));low=float(e[0]-pad);high=float(e[-1]+pad)
    if not np.isfinite(low) or not np.isfinite(high) or low>=high:
        raise ValueError("chemical-potential bracket must be finite and ordered")

    def _residual_sign(mu):
        # Split off exactly occupied reference levels, retaining hole tails.
        # fsum avoids losing a small noninteger filling in that baseline.
        below=e<mu
        baseline=math.fsum([float(w) for w in weights[below]]+[-float(electron_count)])
        positive=[];negative=[]
        if baseline>0:positive.append(math.log(baseline))
        elif baseline<0:negative.append(math.log(-baseline))
        for energy,weight,is_below in zip(e,weights,below):
            if weight==0:continue
            t=abs(float(beta)*(float(energy)-float(mu)))
            log_tail=math.log(float(weight))-t-math.log1p(math.exp(-t))
            if is_below:negative.append(log_tail)
            else:positive.append(log_tail)
        # Compare positive and negative contributions without materializing
        # exponentially small carrier populations.
        logs=[]
        for terms in (positive,negative):
            if not terms:logs.append(-math.inf);continue
            peak=max(terms)
            if peak==-math.inf:logs.append(peak);continue
            logs.append(peak+math.log(math.fsum(math.exp(v-peak) for v in terms)))
        return (logs[0]>logs[1])-(logs[0]<logs[1])

    if not _residual_sign(low)<0 or not _residual_sign(high)>0:
        raise ValueError("chemical-potential interval does not strictly bracket the population")
    for _ in range(64):
        mu=low+(high-low)/2
        if _residual_sign(mu)<0:low=mu
        else:high=mu
    mu=low+(high-low)/2
    scaled=float(beta)*(e-mu);p=np.exp(-np.abs(scaled))
    occupation=np.where(scaled>=0,p/(1+p),1/(1+p))
    x=(left*occupation[None,:])@right
    rho=_oracle_extract_sparse_operator(x,distances,c,sig,keep_radius,0.0)
    return float(mu),rho,x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Original regressions and independent scientific/stability regimes."""
    return [{'setup': 'import numpy as np\n'
           'S=np.array([[1.2,.15],[.15,1.]]);M=np.array([[.93,-.04],[-.04,1.05]]);K=np.array([[-.4,.22],[.22,.3]]);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);sig=np.array([1.,-1.]);R=np.diag(sig)\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n',
  'call': '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,12.0,0.9,1,1.0))',
  'gold_call': '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,12.0,0.9,1,1.0))'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(1);M=np.array([[2.]]);K=np.array([[.7]]);D=np.zeros((1,1));c=np.array([0]);sig=np.array([1.]);R=np.ones((1,1))\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n',
  'call': '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,1.0,1,0.0))',
  'gold_call': '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,1.0,1,0.0))'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(3);M=np.eye(3);K=np.diag([-2.,0.,2.]);D=np.ones((3,3))-np.eye(3);c=np.arange(3);sig=np.ones(3);R=np.eye(3)\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _stable_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise"):\n'
           '        return fun()\n',
  'call': '_stable_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,1000.0,1.25,1,1.0)))',
  'gold_call': '_stable_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,1000.0,1.25,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.,1.1,1.2,1.3]);M=np.diag([1.,.95,.9,.85]);K=np.array([[-.5,.1,0.,0.],[.1,-.2,.12,0.],[0.,.12,.2,.08],[0.,0.,.08,.5]]);D=np.abs(np.arange(4)[:,None]-np.arange(4)[None,:]).astype(float);c=np.array([0,1,0,1]);sig=np.array([1.,-1.,-1.,1.]);R=np.eye(2)[c]*sig[:,None]\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n',
  'call': '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,8.0,1.7,1,1.0))',
  'gold_call': '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,8.0,1.7,1,1.0))'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(2);M=np.eye(2);K=np.diag([-1.,1.]);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);sig=np.ones(2);R=np.eye(2)\n'
           'def run_model():\n'
           '    try:\n'
           '        solve_fixed_population_density(S,M,K,R,D,c,sig,5.0,2.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,5.0,2.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'S=np.eye(2);M=np.eye(2);K=np.diag([-1.,1.]);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);sig=np.ones(2);R=np.eye(2)\n'
           'def run_model():\n'
           '    try:\n'
           '        solve_fixed_population_density(S,M,K,R,D,c,sig,0.0,1.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,0.0,1.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.0, 4.0]);M=np.eye(2);K=np.diag([-1.0, 2.0])\n'
           'c=np.arange(2);sig=np.ones(2);R=np.eye(2);D=np.ones((2,2))-np.eye(2)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,40.0,1.0,1,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,40.0,1.0,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.0, 4.0]);M=np.eye(2);K=np.diag([-2.0, 2.0])\n'
           'c=np.arange(2);sig=np.ones(2);R=np.eye(2);D=np.ones((2,2))-np.eye(2)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,1000.0,1.0,1,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,1000.0,1.0,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([0.5, 1.5, 3.0, 1.0]);M=np.eye(4);K=np.diag([-2.0, -0.8, 1.1, 1.8])\n'
           'c=np.arange(4);sig=np.ones(4);R=np.eye(4);D=np.ones((4,4))-np.eye(4)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,80.0,2.0,1,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,80.0,2.0,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.0, 1.0]);M=np.eye(2);K=np.diag([-1.0, 1.0])\n'
           'c=np.arange(2);sig=np.ones(2);R=np.eye(2);D=np.ones((2,2))-np.eye(2)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          'np.stack([_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,160.0,1.0-2.0**(-40),1,1.0)),_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,160.0,1.0+2.0**(-40),1,1.0))]))',
  'gold_call': '_finite_call(lambda: '
               'np.stack([_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,160.0,1.0-2.0**(-40),1,1.0)),_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,160.0,1.0+2.0**(-40),1,1.0))]))'},
 {'setup': 'import numpy as np\n'
           'n=6;i=np.arange(n);D=np.minimum(abs(i[:,None]-i[None,:]),n-abs(i[:,None]-i[None,:])).astype(float)\n'
           'S=np.eye(n)+.18*np.exp(-D/1.7)\n'
           'M=np.diag([1.02,.96,.90,1.05,.98,1.01])+.025*(D==1)+.007*(D==2)\n'
           'K=np.diag([-.7,-.25,.1,.55,-.4,.35])-.16*(D==1)+.03*(D==2)\n'
           'c=np.array([0,1,0,1,0,1]);sig=np.array([1.,-1.,-1.,1.,1.,-1.]);R=np.eye(2)[c]*sig[:,None]\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,1,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'n=6;i=np.arange(n);D=np.minimum(abs(i[:,None]-i[None,:]),n-abs(i[:,None]-i[None,:])).astype(float)\n'
           'S=np.eye(n)+.18*np.exp(-D/1.7)\n'
           'M=np.diag([1.02,.96,.90,1.05,.98,1.01])+.025*(D==1)+.007*(D==2)\n'
           'K=np.diag([-.7,-.25,.1,.55,-.4,.35])-.16*(D==1)+.03*(D==2)\n'
           'c=np.array([0,1,0,1,0,1]);sig=np.array([1.,-1.,-1.,1.,1.,-1.]);R=np.eye(2)[c]*sig[:,None]\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,2,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,2,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([1.0, 4.0]);M=np.eye(2);K=np.diag([12.0, 15.0])\n'
           'c=np.arange(2);sig=np.ones(2);R=np.eye(2);D=np.ones((2,2))-np.eye(2)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          '_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,40.0,1.0,1,1.0)))',
  'gold_call': '_finite_call(lambda: '
               '_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,40.0,1.0,1,1.0)))'},
 {'setup': 'import numpy as np\n'
           'S=np.diag([4.0]);M=np.eye(1);K=np.diag([0.25])\n'
           'c=np.arange(1);sig=np.ones(1);R=np.eye(1);D=np.ones((1,1))-np.eye(1)\n'
           '\n'
           'def _pack_result(value):\n'
           '    mu,rho,action=value\n'
           '    return '
           'np.concatenate((np.array([mu,*rho.shape,*action.shape],dtype=float),rho.ravel(),action.ravel()))\n'
           '\n'
           'def _finite_call(fun):\n'
           '    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):\n'
           '        result=fun()\n'
           '    if not np.all(np.isfinite(result)):\n'
           '        raise ValueError("nonfinite output")\n'
           '    return result\n',
  'call': '_finite_call(lambda: '
          'np.stack([_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,2.0**(-44),1,0.0)),_pack_result(solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,4.0-2.0**(-40),1,0.0))]))',
  'gold_call': '_finite_call(lambda: '
               'np.stack([_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,2.0**(-44),1,0.0)),_pack_result(_oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,4.0,4.0-2.0**(-40),1,0.0))]))'},
 {'setup': 'import numpy as np\n'
           'S=.1*np.eye(3)+.9*np.ones((3,3));M=np.eye(3)\n'
           'D=abs(np.arange(3)[:,None]-np.arange(3)[None,:]).astype(float)\n'
           'K=S*(D<=1.);c=np.arange(3);sig=np.ones(3);R=np.eye(3)\n'
           '\n'
           'def run_model():\n'
           '    try:\n'
           '        solve_fixed_population_density(S,M,K,R,D,c,sig,5.0,1.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,5.0,1.0,1,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'},
 {'setup': 'import numpy as np\n'
           'n=6;i=np.arange(n);D=np.minimum(abs(i[:,None]-i[None,:]),n-abs(i[:,None]-i[None,:])).astype(float)\n'
           'S=np.eye(n)+.18*np.exp(-D/1.7)\n'
           'M=np.diag([1.02,.96,.90,1.05,.98,1.01])+.025*(D==1)+.007*(D==2)\n'
           'K=np.diag([-.7,-.25,.1,.55,-.4,.35])-.16*(D==1)+.03*(D==2)\n'
           'c=np.array([0,1,0,1,0,1]);sig=np.array([1.,-1.,-1.,1.,1.,-1.]);R=np.eye(2)[c]*sig[:,None]\n'
           'R=R/np.sqrt(3.)\n'
           '\n'
           'def run_model():\n'
           '    try:\n'
           '        solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,2,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def run_gold():\n'
           '    try:\n'
           '        _oracle_solve_fixed_population_density(S,M,K,R,D,c,sig,24.0,2.25,2,1.0)\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': 'run_model()',
  'gold_call': 'run_gold()'}]
