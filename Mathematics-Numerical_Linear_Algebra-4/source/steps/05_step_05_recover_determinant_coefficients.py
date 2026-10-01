"""
Recover a determinant polynomial from confluent Fourier data and certify it off grid.

The polynomial $D(x)=\sum_{n=0}^{K-1}c_nx^n$ is sampled through ordinary

derivatives $D^{(r)}(x_j)$ on a Fourier grid of $S$ nodes. Derivative data

separate coefficients whose degrees coincide modulo $S$. Recovering these

bands is a Hermite interpolation problem, with the degree bound preserved.

Returns
-------
A real NumPy array of shape $(K,)$ containing all determinant coefficients in ascending degree order, including padding.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def recover_determinant_coefficients(
    determinant_samples: np.ndarray, coefficient_tensor: np.ndarray,
    validation_point: float = 0.37, validation_tolerance: float = 1e-9,
) -> np.ndarray:
    r"""Recover the $K$ ascending real coefficients of $D(x)=\det M(x)$.
    
    Parameters
    ----------
    determinant_samples : np.ndarray
        Finite real or complex data of shape $(S,)$ for values or $(J,S)$
        for ordinary derivatives $D^{(r)}(x_j)$, with $1\le J\le4$,
        $S\ge2$, $0\le r<J$, and $x_j=\exp(-2\pi i j/S)$.
        Value-only input means $J=1$. Require $JS\ge K$.
    coefficient_tensor : np.ndarray
        Finite real or complex array of shape $(K,N,N)$, with $K\ge1$,
        representing $M(x)$ in ascending degree order. The determinant
        $D(x)=\det M(x)$ is promised to have degree at most $K-1$ and
        numerically real coefficients.
    validation_point : float, optional
        Finite real off-grid point $x_*$, default $0.37$. Its distance
        from every sampling node must exceed $10^{-12}$.
    validation_tolerance : float, optional
        Finite positive tolerance $\tau$, default $10^{-9}$, for the
        coefficient-reality, reconstructed-sample, and off-grid checks.
    
    Returns
    -------
    coefficients : np.ndarray
        Real vector of shape $(K,)$ containing $c_0,\ldots,c_{K-1}$
        in ascending degree order. Retain trailing padding; the coefficients
        must pass all three consistency checks described below.
    
    Raises
    ------
    ValueError
        If input shapes, channel count, sample count, or tensor dimensions
        are invalid; $JS<K$; data or settings are nonfinite; $\tau\le0$;
        the validation point is within $10^{-12}$ of a grid node; or any
        coefficient-reality, reconstructed-sample, or off-grid check fails.
    
    Notes
    -----
    Within each Fourier residue class, separate aliased coefficients using all
    available derivative channels. Let $V$ be that class's falling-factorial
    moment matrix. Divide equation $r$ by
    $s_r=\max\{1,\max_n|V_{rn}|\}$. If there are more equations than unknowns,
    use the unique least-squares solution of the scaled equations.
    
    Let $\tau$ denote `validation_tolerance` and $x_*$ denote `validation_point`.
    Raise `ValueError` for invalid shapes, insufficient samples, nonfinite data
    or settings, $\tau\le0$, or $\min_j|x_*-x_j|\le10^{-12}$.
    The point $x_*$ is a finite real scalar. Reject recovered coefficients
    with $\max_n|\operatorname{Im}c_n|>\tau\max\{1,\max_n|c_n|\}$.
    Also reject inconsistent derivative data: if $Y$ denotes the supplied
    sample array and $\widetilde Y$ the samples reconstructed from the real
    coefficients, require
    $\max_{r,j}|\widetilde Y_{rj}-Y_{rj}|\le\tau\max\{1,\max_{r,j}|Y_{rj}|\}$.
    Finally, for $p(x)=\sum_{n=0}^{K-1}c_nx^n$, require
    $|p(x_*)-\det M(x_*)|\le\tau(1+|\det M(x_*)|)$; otherwise raise `ValueError`.
    Do not mutate inputs. Exact singular sample matrices are valid data.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_recover_determinant_coefficients(
    determinant_samples: np.ndarray, coefficient_tensor: np.ndarray,
    validation_point: float = 0.37, validation_tolerance: float = 1e-9,
) -> np.ndarray:
    raw = np.asarray(determinant_samples)
    tensor = np.asarray(coefficient_tensor)
    if raw.ndim not in (1,2) or tensor.ndim != 3 or tensor.shape[0] < 1 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError("invalid sample or tensor rank")
    data = raw[None] if raw.ndim == 1 else raw
    channels,count = data.shape
    length = tensor.shape[0]
    if not 1 <= channels <= 4 or count < 2 or channels*count < length:
        raise ValueError("insufficient or invalid confluent data")
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(tensor)):
        raise ValueError("data must be finite")
    if not np.isfinite(validation_point) or not np.isfinite(validation_tolerance) or validation_tolerance <= 0:
        raise ValueError("invalid validation settings")
    points = np.exp(-2j*np.pi*np.arange(count)/count)
    if np.min(np.abs(points-validation_point)) <= 1e-12:
        raise ValueError("validation point must be off grid")
    # Multiplication by x_j**r aligns each derivative with degree modulo S.
    moments = np.fft.ifft(data*points[None,:]**np.arange(channels)[:,None], axis=1)
    coefficients = np.zeros(length,dtype=complex)
    for residue in range(count):
        degrees = np.arange(residue,length,count)
        if not degrees.size:
            continue
        vandermonde = np.ones((channels,degrees.size),dtype=float)
        for order in range(1,channels):
            vandermonde[order] = vandermonde[order-1]*(degrees-order+1)
        # Scale equations together with their right-hand sides for stability.
        row_scale = np.maximum(1.0,np.max(np.abs(vandermonde),axis=1))
        coefficients[degrees] = np.linalg.lstsq(vandermonde/row_scale[:,None],moments[:,residue]/row_scale,rcond=None)[0]
    scale = max(1.0,float(np.max(np.abs(coefficients))))
    if np.max(np.abs(coefficients.imag)) > validation_tolerance*scale:
        raise ValueError("coefficients are not numerically real")
    coefficients = coefficients.real
    recovered = np.zeros_like(data,dtype=complex)
    for order in range(channels):
        derivative = np.polynomial.polynomial.polyder(coefficients,order)
        recovered[order] = np.polynomial.polynomial.polyval(points,derivative)
    if np.max(np.abs(recovered-data)) > validation_tolerance*max(1.0,float(np.max(np.abs(data)))):
        raise ValueError("inconsistent derivative samples")
    matrix = np.polynomial.polynomial.polyval(validation_point,tensor)
    direct = _oracle_compute_determinant_samples(matrix[None])[0]
    value = np.polynomial.polynomial.polyval(validation_point,coefficients)
    if abs(direct-value) > validation_tolerance*(1+abs(direct)):
        raise ValueError("off-grid determinant validation failed")
    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific cases with independent mutable inputs on both sides."""
    return [{'setup': 'import numpy as np\n'
               'coefficient_tensor=np.zeros((4,1,1)); coefficient_tensor[:,0,0]=[2.,-1.,.5,3.]\n'
               'determinant_samples=np.fft.fft(coefficient_tensor[:,0,0])',
      'call': 'recover_determinant_coefficients(determinant_samples.copy(), coefficient_tensor.copy())',
      'gold_call': '_oracle_recover_determinant_coefficients(determinant_samples.copy(), '
                   'coefficient_tensor.copy())'},
     {'setup': 'import numpy as np\n'
               'coefficient_tensor=np.zeros((2,1,1)); coefficient_tensor[0,0,0]=4.\n'
               'determinant_samples=np.fft.fft(coefficient_tensor[:,0,0])',
      'call': 'recover_determinant_coefficients(determinant_samples.copy(), coefficient_tensor.copy())',
      'gold_call': '_oracle_recover_determinant_coefficients(determinant_samples.copy(), '
                   'coefficient_tensor.copy())'},
     {'setup': 'import numpy as np\n'
               'coefficient_tensor = np.zeros((3, 1, 1))\n'
               'determinant_samples = np.ones(4)\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(determinant_samples.copy(), '
               'coefficient_tensor.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(determinant_samples.copy(), '
               'coefficient_tensor.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'c=np.array([1.,-2.,.5,3.,0.,1.2,-.7,2.,.1,-.2,.3,-.4,.5]);T=c[:,None,None];z=np.exp(-2j*np.pi*np.arange(4)/4);data=np.array([np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c,r)) '
               'for r in range(4)])',
      'call': 'recover_determinant_coefficients(data.copy(), T.copy())',
      'gold_call': '_oracle_recover_determinant_coefficients(data.copy(), T.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'T=np.zeros((13,3,3));T[:5,0,0]=[1.,0.,-2.,0.,1.];T[:5,1,1]=[1.,0.,-2.,0.,1.];T[:5,2,2]=[1.,0.,-2.,0.,1.];T[0,0,1]=3.;T[1,0,2]=2.;T[2,1,2]=-1.;c=np.polynomial.polynomial.polypow(np.array([1.,0.,-2.,0.,1.]),3);z=np.exp(-2j*np.pi*np.arange(4)/4);data=np.array([np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c,r)) '
               'for r in range(4)])',
      'call': 'recover_determinant_coefficients(data.copy(), T.copy())',
      'gold_call': '_oracle_recover_determinant_coefficients(data.copy(), T.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'c=np.array([2.,-1.,3.,0.,0.]);T=c[:,None,None];z=np.exp(-2j*np.pi*np.arange(3)/3);data=np.array([np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c,r)) '
               'for r in range(3)])',
      'call': 'recover_determinant_coefficients(data.copy(), T.copy())',
      'gold_call': '_oracle_recover_determinant_coefficients(data.copy(), T.copy())',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'c = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 1.2, -0.7, 2.0, 0.1, -0.2, 0.3, -0.4, 0.5])\n'
               'T = c[:, None, None]\n'
               'z = np.exp(-2j * np.pi * np.arange(4) / 4)\n'
               'data = np.array([np.polynomial.polynomial.polyval(z, '
               'np.polynomial.polynomial.polyder(c, r)) for r in range(4)])\n'
               'data[3, 1] += 1\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'c = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 1.2, -0.7, 2.0, 0.1, -0.2, 0.3, -0.4, 0.5])\n'
               'T = c[:, None, None]\n'
               'z = np.exp(-2j * np.pi * np.arange(4) / 4)\n'
               'data = np.array([np.polynomial.polynomial.polyval(z, '
               'np.polynomial.polynomial.polyder(c, r)) for r in range(4)])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(data[:2].copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(data[:2].copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'c = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 1.2, -0.7, 2.0, 0.1, -0.2, 0.3, -0.4, 0.5])\n'
               'T = c[:, None, None]\n'
               'z = np.exp(-2j * np.pi * np.arange(4) / 4)\n'
               'data = np.array([np.polynomial.polynomial.polyval(z, '
               'np.polynomial.polynomial.polyder(c, r)) for r in range(4)])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(data.copy(), T.copy(), validation_point=1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(data.copy(), T.copy(), '
               'validation_point=1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'c = np.array([1.0, -2.0, 0.5, 3.0, 0.0, 1.2, -0.7, 2.0, 0.1, -0.2, 0.3, -0.4, 0.5])\n'
               'T = c[:, None, None]\n'
               'z = np.exp(-2j * np.pi * np.arange(4) / 4)\n'
               'data = np.array([np.polynomial.polynomial.polyval(z, '
               'np.polynomial.polynomial.polyder(c, r)) for r in range(4)])\n'
               'T = T.copy()\n'
               'T[0, 0, 0] += 2\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'T = np.array([1.0 + 1j, 2.0])[:, None, None]\n'
               'data = np.fft.fft(T[:, 0, 0])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_recover_determinant_coefficients(data.copy(), T.copy())\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1',
      'call': 'run_model()',
      'gold_call': 'run_gold()',
      'tol': 0}]
