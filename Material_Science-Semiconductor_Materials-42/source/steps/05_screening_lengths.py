"""
Extract the anisotropic microscopic screening lengths from the dielectric Schur complement (SI S.12) for the origin-cell treatment (SI S.20–S.21); the finite sampling rule is stipulated.

Extract the anisotropic microscopic screening lengths from the dielectric Schur complement (SI S.12) for the origin-cell treatment (SI S.20–S.21); the finite sampling rule is stipulated.

The following finite-grid origin prescription fixes the benchmark and extends the paper's first-order circular-cell rule to its pair-averaged Q2D potential. Let \(\delta=0.002/a\) and \(s_\alpha(t)=[\varepsilon_M(t\hat{\boldsymbol\alpha})-1]/t\); use \(r_\alpha=2s_\alpha(\delta/2)-s_\alpha(\delta)\) for \(\alpha=x,y\). With \(q_0=0.35(2\pi/an)\), replace \(\bar W_{00}(0)\) by \(c[2/q_0-(r_x+r_y)/2-d/3]\), set the origin wings to zero, and retain the origin body. This is the stipulated first-order cell prescription at the fixed \(\delta,q_0\); the synthetic targets refer to that prescription.
Here delta is the supplied sampling momentum. The inverse-head reciprocal is the dielectric Schur complement, including the finite-G body.

Returns
-------
return result  # real ndarray, shape (2,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def screening_lengths(dielectric_samples: np.ndarray, delta: float) -> np.ndarray:
    """Parameters
    ----------
    dielectric_samples : finite positive Hermitian complex ndarray, shape (4,G,G)
        Dimensionless symmetric epsilon_bar at (delta,0), (delta/2,0),
        (0,delta), (0,delta/2), in exactly that order; G>=1 and head index 0.
    delta : finite positive float
        Sampling momentum in inverse angstroms.
    Returns
    -------
    real ndarray, shape (2,)
        Ordered [r_x,r_y] in angstroms, using r_alpha=2*s_alpha(delta/2)
        -s_alpha(delta), with s_alpha(t)=(epsilon_M(t)-1)/t. The macroscopic
        epsilon_M is the full inverse-head reciprocal, equivalently the Schur
        complement of the microscopic dielectric body; include local fields.
    Raises
    ------
    ValueError : malformed/nonfinite/non-positive-Hermitian samples or invalid delta."""
    return np.zeros((2,))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_screening_lengths(dielectric_samples, delta):
    e = np.asarray(dielectric_samples, dtype=complex)
    if e.ndim != 3 or e.shape[0] != 4 or e.shape[1] != e.shape[2] or not e.shape[1] or not np.isfinite(e).all() or not np.isfinite(delta) or delta <= 0:
        raise ValueError('four finite square dielectric samples and positive delta required')
    if not np.allclose(e,e.conj().swapaxes(-1,-2),atol=1e-10) or np.any(np.linalg.eigvalsh(e) <= 0):
        raise ValueError('positive Hermitian dielectric samples required')
    macro = e[:,0,0].real.copy()
    if e.shape[1] > 1:
        body_solution = np.linalg.solve(e[:,1:,1:],e[:,1:,0,None])[:,:,0]
        macro -= np.einsum('qg,qg->q',e[:,0,1:],body_solution).real
    r = (macro-1)/np.array([delta,delta/2,delta,delta/2])
    return np.array([2*r[1]-r[0],2*r[3]-r[2]],dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ndelta=0.001; t=np.array([delta,delta/2,delta,delta/2]); rr=np.repeat([2.0, 2.0],2); ss=np.repeat([0.0, 0.0],2); tt=np.repeat([0.0, 0.0],2); h=1/(1+rr*t+ss*t*t+tt*t*t*t)\neps=np.tile(np.array([[1.,.2j,.3],[-.2j,2.,.1],[.3,.1,1.5]],complex),(4,1,1))\nfor i in range(4): eps[i,0,0]=1/h[i]+eps[i,0,1:]@np.linalg.solve(eps[i,1:,1:],eps[i,1:,0])\n', 'call': 'screening_lengths(eps,delta)', 'gold_call': '_oracle_screening_lengths(eps,delta)'}, {'setup': 'import numpy as np\ndelta=0.003; t=np.array([delta,delta/2,delta,delta/2]); rr=np.repeat([0.13, 4.7],2); ss=np.repeat([0.0, 0.0],2); tt=np.repeat([0.0, 0.0],2); h=1/(1+rr*t+ss*t*t+tt*t*t*t)\neps=np.tile(np.array([[1.,.2j,.3],[-.2j,2.,.1],[.3,.1,1.5]],complex),(4,1,1))\nfor i in range(4): eps[i,0,0]=1/h[i]+eps[i,0,1:]@np.linalg.solve(eps[i,1:,1:],eps[i,1:,0])\n', 'call': 'screening_lengths(eps,delta)', 'gold_call': '_oracle_screening_lengths(eps,delta)'}, {'setup': 'import numpy as np\ndelta=0.002; t=np.array([delta,delta/2,delta,delta/2]); rr=np.repeat([0.3, 0.8],2); ss=np.repeat([2.0, -1.0],2); tt=np.repeat([0.0, 0.0],2); h=1/(1+rr*t+ss*t*t+tt*t*t*t)\neps=np.tile(np.array([[1.,.2j,.3],[-.2j,2.,.1],[.3,.1,1.5]],complex),(4,1,1))\nfor i in range(4): eps[i,0,0]=1/h[i]+eps[i,0,1:]@np.linalg.solve(eps[i,1:,1:],eps[i,1:,0])\n', 'call': 'screening_lengths(eps,delta)', 'gold_call': '_oracle_screening_lengths(eps,delta)'}, {'setup': 'import numpy as np\ndelta=0.02; t=np.array([delta,delta/2,delta,delta/2]); rr=np.repeat([0.7, 1.3],2); ss=np.repeat([2.0, 3.0],2); tt=np.repeat([7.0, -4.0],2); h=1/(1+rr*t+ss*t*t+tt*t*t*t)\neps=np.tile(np.array([[1.,.2j,.3],[-.2j,2.,.1],[.3,.1,1.5]],complex),(4,1,1))\nfor i in range(4): eps[i,0,0]=1/h[i]+eps[i,0,1:]@np.linalg.solve(eps[i,1:,1:],eps[i,1:,0])\n', 'call': 'screening_lengths(eps,delta)', 'gold_call': '_oracle_screening_lengths(eps,delta)'}, {'setup': 'import numpy as np\ndelta=0.001; t=np.array([delta,delta/2,delta,delta/2]); rr=np.repeat([0.0, 0.0],2); ss=np.repeat([0.0, 0.0],2); tt=np.repeat([0.0, 0.0],2); h=1/(1+rr*t+ss*t*t+tt*t*t*t)\neps=np.tile(np.array([[1.,.2j,.3],[-.2j,2.,.1],[.3,.1,1.5]],complex),(4,1,1))\nfor i in range(4): eps[i,0,0]=1/h[i]+eps[i,0,1:]@np.linalg.solve(eps[i,1:,1:],eps[i,1:,0])\n', 'call': 'screening_lengths(eps,delta)', 'gold_call': '_oracle_screening_lengths(eps,delta)'}, {'setup': 'import numpy as np\n\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: screening_lengths(np.ones(3),.001))', 'gold_call': '_exception_code(lambda: _oracle_screening_lengths(np.ones(3),.001))'}]
