"""
Reconvolution validity of a regularized formation history

A regularized inverse is biased by construction, so a reconstructed formation history is trustworthy only where pushing it back through the forward model reproduces the merger history it was built from. Convolve the reconstruction with the discrete delay kernel $$p\,\Delta t$$ on a periodic vector of length $$2N$$, retain the first $$N$$ samples, and compare with the supplied merger history. The diagnostic is the largest fractional discrepancy over a supplied boolean window, $$\mathrm{err}=\max_{\rm window}|\mathcal{R}_{\rm reconv}-\mathcal{R}_{\rm merge}|/|\mathcal{R}_{\rm merge}|$$. The window selects the well-measured samples, where the merger history is bounded away from zero. A large diagnostic means the sampling does not resolve the delay kernel, so a physicality statement read off that reconstruction describes the discretization rather than the delay model.

Returns
-------
float, the maximum fractional reconvolution error over the window as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconvolution_error(r_form: np.ndarray, p_tau: np.ndarray,
                        r_merge: np.ndarray, dt: float,
                        window_mask: np.ndarray) -> float:
    r'''Measure how well a reconstruction reproduces its own merger history.

    Parameters
    ----------
    r_form : np.ndarray
        Finite real one-dimensional reconstructed formation history, N >= 2 samples, in Gpc^-3 yr^-1. Signed entries are allowed, because a regularized inverse may undershoot.
    p_tau : np.ndarray
        Finite real one-dimensional delay density on the same uniform grid, shape (N,), in Gyr^-1. The supplied sampled amplitudes are used as given.
    r_merge : np.ndarray
        Finite real one-dimensional merger history on the same grid, shape (N,), in Gpc^-3 yr^-1, nonzero on every selected sample.
    dt : float
        Finite positive sample spacing in Gyr, supplying the kernel quadrature weight.
    window_mask : np.ndarray
        Boolean shape (N,) comparison window selecting at least one sample.

    Returns
    -------
    err : float
        Finite native Python float, the maximum over the selected samples of the absolute reconvolution residual divided by the absolute merger history. Zero-padding to length 2N and retention of the first N samples make the comparison acyclic on the supplied grid.

    Raises
    ------
    ValueError
        If the finite real input, array-shape, boolean-mask, spacing or nonzero-denominator contracts fail, if the mask selects no sample, or if the diagnostic is nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconvolution_error(r_form: np.ndarray, p_tau: np.ndarray,
                                r_merge: np.ndarray, dt: float,
                                window_mask: np.ndarray) -> float:
    values = (r_form, p_tau, r_merge)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all histories and densities must be real")
    try:
        formation, density, merger = [np.asarray(v, dtype=float) for v in values]
        if not np.isscalar(dt):
            raise ValueError("the sample spacing must be a scalar")
        spacing = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real arrays and a scalar spacing required") from exc
    if formation.ndim != 1 or formation.size < 2:
        raise ValueError("r_form must be one-dimensional with at least two samples")
    n = formation.size
    mask = np.asarray(window_mask)
    if density.shape != (n,) or merger.shape != (n,) or mask.shape != (n,):
        raise ValueError("densities, histories and the mask must match the time grid")
    if mask.dtype != np.bool_:
        raise ValueError("window_mask must be a boolean array")
    if not all(np.all(np.isfinite(v)) for v in (formation, density, merger)):
        raise ValueError("all arrays must be finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("finite positive spacing required")
    if not np.any(mask):
        raise ValueError("window_mask must select at least one sample")
    if np.any(merger[mask] == 0.0):
        raise ValueError("r_merge must be nonzero on every selected sample")
    reconvolved = np.fft.irfft(np.fft.rfft(formation, n=2*n)
                               * np.fft.rfft(density * spacing, n=2*n), n=2*n)[:n]
    error = float(np.max(np.abs(reconvolved[mask] - merger[mask]) / np.abs(merger[mask])))
    if not np.isfinite(error):
        raise ValueError("the reconvolution diagnostic must be finite")
    return error

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Explicit cases keep static Studio validation aligned with execution."""
    return [{'setup': 'import numpy as np\n'
               'n=512\n'
               't=np.linspace(0.0,13.797,n)\n'
               'dt=t[1]-t[0]\n'
               'r_form=8.0*np.exp(-.5*((t-5.0)/1.6)**2)+1.5\n'
               'p=np.zeros(n)\n'
               'inside=t>=.01\n'
               'p[inside]=t[inside]**-1.3\n'
               'p/=(13.797**-.3-.01**-.3)/-.3\n'
               'r_merge=np.fft.irfft(np.fft.rfft(r_form,n=2*n)*np.fft.rfft(p*dt,n=2*n),n=2*n)[:n]\n'
               'mask=(t>2.0)&(t<9.0)\n',
      'call': 'reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'gold_call': '_oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'n=512\n'
               't=np.linspace(0.0,13.797,n)\n'
               'dt=t[1]-t[0]\n'
               'r_merge=25.0*np.exp(-.5*((t-6.0)/3.0)**2)+5.0\n'
               'p=np.zeros(n)\n'
               'inside=t>=.01\n'
               'p[inside]=t[inside]**-1.3\n'
               'p/=(13.797**-.3-.01**-.3)/-.3\n'
               'k=np.fft.rfft(p*dt,n=2*n)\n'
               'r_form=np.fft.irfft(np.fft.rfft(r_merge,n=2*n)*k.conj()/(np.abs(k)**2+1e-3*np.max(np.abs(k)**2)),n=2*n)[:n]\n'
               'mask=(t>2.0)&(t<9.0)\n',
      'call': 'reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'gold_call': '_oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'n=32\n'
               't=np.linspace(0.0,13.797,n)\n'
               'dt=t[1]-t[0]\n'
               'r_merge=25.0*np.exp(-.5*((t-6.0)/3.0)**2)+5.0\n'
               'p=np.zeros(n)\n'
               'inside=t>=.01\n'
               'p[inside]=t[inside]**-1.3\n'
               'p/=(13.797**-.3-.01**-.3)/-.3\n'
               'k=np.fft.rfft(p*dt,n=2*n)\n'
               'r_form=np.fft.irfft(np.fft.rfft(r_merge,n=2*n)*k.conj()/(np.abs(k)**2+1e-3*np.max(np.abs(k)**2)),n=2*n)[:n]\n'
               'mask=(t>2.0)&(t<9.0)\n',
      'call': 'reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'gold_call': '_oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'n=64\n'
               't=np.linspace(0.0,13.797,n)\n'
               'dt=t[1]-t[0]\n'
               'r_form=np.ones(n)\n'
               'p=np.zeros(n)\n'
               'inside=t>=.01\n'
               'p[inside]=t[inside]**-1.3\n'
               'p/=(13.797**-.3-.01**-.3)/-.3\n'
               'r_merge=np.full(n,2.0)\n'
               'r_merge[40]=-3.0\n'
               'mask=np.zeros(n,dtype=bool)\n'
               'mask[40]=True\n',
      'call': 'reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'gold_call': '_oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),dt,mask.copy())',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'n=64\n'
               'r_form=np.ones(n)\n'
               'p=np.ones(n)\n'
               'r_merge=np.ones(n)\n'
               'mask=np.zeros(n,dtype=bool)\n'
               'def run_model():\n'
               '    try:\n'
               '        reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),0.1,mask.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),0.1,mask.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'n=64\n'
               'r_form=np.ones(n)\n'
               'p=np.ones(n)\n'
               'r_merge=np.ones(n)\n'
               'r_merge[10]=0.0\n'
               'mask=np.zeros(n,dtype=bool)\n'
               'mask[10]=True\n'
               'def run_model():\n'
               '    try:\n'
               '        reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),0.1,mask.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_reconvolution_error(r_form.copy(),p.copy(),r_merge.copy(),0.1,mask.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
