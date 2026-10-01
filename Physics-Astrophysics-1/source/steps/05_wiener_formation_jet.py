"""
Formation histories and their response to the delay-mixture fraction

For each row of an affine merger family, define the formation reconstruction as the minimizer of $$\|C_f x-r\|_2^2+\lambda(f)\|x\|_2^2$$ on a periodic vector of length $$2N$$. Here $$r$$ is the supplied row padded with zeros, $$C_f$$ is circular convolution with the zero-padded kernel $$\Delta t[(1-f)p_s+fp_l]$$, and $$\lambda(f)=10^{-3}\max_k|K_f(k)|^2$$, with $$K_f$$ the unnormalized discrete Fourier transform of that kernel. Return the first $$N$$ samples of the minimizer and its first two ordinary derivatives with respect to $$f$$. The merger family and the two continuously normalized delay densities are held fixed during differentiation. This response is needed because changing the delay mixture moves the formation-positivity boundary in the latent rate posterior. The penalty belongs to the changing operator and is differentiated as well. At either endpoint of $$[0,1]$$, derivatives mean limits from inside that interval.

Returns
-------
np.ndarray of shape (3,4,N), the formation family and its first two ordinary fraction derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def wiener_formation_jet(merger_family: np.ndarray, steep_density: np.ndarray,
                         shallow_density: np.ndarray, fraction: float,
                         dt: float) -> np.ndarray:
    r'''Evaluate the regularized formation family and its local fraction response.

    Parameters
    ----------
    merger_family : np.ndarray
        Finite real shape (4,N), N >= 2. Row 0 is the central merger history;
        rows 1–3 multiply the three latent rate coefficients. Units are
        Gpc^-3 yr^-1, and signed entries are allowed.
    steep_density, shallow_density : np.ndarray
        Finite real shape (N,), nonnegative with positive sum, in Gyr^-1.
        Preserve the supplied sampled amplitudes; both component densities
        have already been continuously normalized.
    fraction : float
        Finite real mixture fraction in [0,1].
    dt : float
        Finite positive sample spacing in Gyr.

    Returns
    -------
    jet : np.ndarray
        Finite shape (3,4,N). Leading rows are value, first derivative and
        second derivative with respect to fraction, in that order. These
        are ordinary derivatives, without factorial rescaling. All four
        affine rows use the same fraction-dependent inverse and penalty.

    Raises
    ------
    ValueError
        If the finite real input, array-shape, nonnegative density, fraction
        or spacing contracts fail, or the spectral result is nonfinite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_wiener_formation_jet(merger_family: np.ndarray,
                                 steep_density: np.ndarray,
                                 shallow_density: np.ndarray,
                                 fraction: float, dt: float) -> np.ndarray:
    values = (merger_family, steep_density, shallow_density, fraction, dt)
    if any(not np.isrealobj(v) for v in values):
        raise ValueError("all inputs must be real")
    try:
        family, steep, shallow = [np.asarray(v, dtype=float) for v in values[:3]]
        if not np.isscalar(fraction) or not np.isscalar(dt):
            raise ValueError("fraction and spacing must be scalars")
        f, spacing = float(fraction), float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite real arrays and scalars required") from exc
    if family.ndim != 2 or family.shape[0] != 4 or family.shape[1] < 2:
        raise ValueError("merger_family must have shape (4,N), N >= 2")
    n = family.shape[1]
    if steep.shape != (n,) or shallow.shape != (n,):
        raise ValueError("component densities must match the time grid")
    if not all(np.all(np.isfinite(v)) for v in (family, steep, shallow)):
        raise ValueError("all arrays must be finite")
    if any(np.any(v < 0.0) or not np.any(v > 0.0) for v in (steep, shallow)):
        raise ValueError("component densities must be nonnegative and nonzero")
    if not np.isfinite(f) or not 0.0 <= f <= 1.0 or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("fraction in [0,1] and finite positive spacing required")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        density = (1.0 - f) * steep + f * shallow
        delta = shallow - steep
        kernel = np.fft.rfft(density * spacing, n=2*n)
        direction = np.fft.rfft(delta * spacing, n=2*n)
        rate = np.fft.rfft(family, n=2*n, axis=1)
        # A nonnegative kernel has maximal Fourier modulus at zero frequency.
        dc, dc_first = kernel[0].real, direction[0].real
        penalty = 1e-3 * dc**2
        penalty_first = 2e-3 * dc * dc_first
        penalty_second = 2e-3 * dc_first**2
        denominator = np.abs(kernel)**2 + penalty
        first_denominator = 2.0 * np.real(kernel.conj() * direction) + penalty_first
        second_denominator = 2.0 * np.abs(direction)**2 + penalty_second
        value = rate * kernel.conj() / denominator
        first = (rate * direction.conj() - first_denominator * value) / denominator
        second = -(second_denominator * value + 2.0 * first_denominator * first) / denominator
        result = np.fft.irfft(np.stack((value, first, second)), n=2*n, axis=2)[:, :, :n]
    if not np.all(np.isfinite(result)):
        raise ValueError("the formation response must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Explicit cases keep static Studio validation aligned with execution."""
    return [{'setup': 'import numpy as np\n'
               'n=32\n'
               'f=0.0\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=64\n'
               'f=0.37\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=256\n'
               'f=0.73\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=512\n'
               'f=1.0\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=96\n'
               'f=.4\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n'
               'pl=ps.copy()\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'n=128\n'
               'f=.58\n'
               'import numpy as np\n'
               't=np.linspace(0,13.8,n)\n'
               'dt=t[1]-t[0]\n'
               'base=20+8*np.cos(t/4)+2*np.sin(t)\n'
               'family=np.array([base,.2*base*np.cos(t/5),.15*base*np.exp(-((t-4)/2)**2),-.12*base*np.sin(t/3)])\n'
               'def component(alpha):\n'
               '    d=np.zeros(n);inside=t>=.01\n'
               '    power=alpha+1\n'
               '    norm=np.log(13.8/.01) if power==0 else (13.8**power-.01**power)/power\n'
               '    d[inside]=t[inside]**alpha/norm\n'
               '    return d\n'
               'ps=component(-1.73);pl=component(-.99)\n'
               'ps*=.35;pl*=1.4;family[2]*=-3\n',
      'call': 'wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'gold_call': '_oracle_wiener_formation_jet(family.copy(),ps.copy(),pl.copy(),f,dt)',
      'tol': 2e-06}]
