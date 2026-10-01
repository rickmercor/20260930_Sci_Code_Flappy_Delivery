"""
Resolve the task-defined linearized transmission model by scattering origin.

For two projected-potential stacks B and D, define a formal exit wave by using
the grating 1+i*sigma*(a*B_j+b*D_j) at each slice, and the declared Fresnel
propagation between successive gratings. Return the complex coefficients of
this bivariate polynomial in a and b at the immediate post-final-grating plane.
These are polynomial coefficients, not derivatives or probabilities. This
diagnostic is derived from QuScope's ordered coherent propagation; it is not a
separate simulation feature claimed to ship in QuScope.

The source physics is the coherent ordered multislice map and momentum-plane detection. The scattering-order representation is task-derived; it preserves the existing linearized diagnostic and does not alter the released STEM resource boundary.

Returns
-------
np.ndarray, ((s+1)*(s+2)//2,N,N) complex unnormalized bivariate coefficients in the documented triangular order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bivariate_multislice_orders(background_stack: 'np.ndarray', contrast_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float) -> 'np.ndarray':
    """Return the unnormalized bivariate scattering-order exit amplitudes.

Parameters
----------
background_stack : array_like
    Finite real (s,N,N) projected background potentials B, in V Å; 1<=s<=6.
contrast_stack : array_like
    Finite real projected contrast potentials D, in V Å, with the same shape as B.
incident_state : array_like
    Finite complex (N,N) input amplitudes matching the potential grid. It must be nonzero and is used without renormalization.
wavelength_a : float
    Positive finite electron wavelength, in Å.
interaction_constant : float
    Finite nonnegative interaction constant in rad/(V Å).
propagation_distance_a : float
    Positive finite inter-grating propagation distance, in Å.
pixel_size_a : float
    Positive finite real-space sampling interval, in Å.

Returns
-------
result : np.ndarray
    ((s+1)*(s+2)//2,N,N) complex unnormalized bivariate coefficients in the documented triangular order.

Notes
-----
Return the unnormalized bivariate scattering-order exit amplitudes.

background_stack B and contrast_stack D: finite real arrays (s,N,N),
1<=s<=6, with projected potentials in V Å. N>=2 is a power of two;
within each (N,N) slice, axis 0=x and axis 1=y. incident_state: finite nonzero complex
(N,N) amplitude array, used without renormalization. wavelength_a,
propagation_distance_a and pixel_size_a: positive finite Å scalars.
interaction_constant: finite nonnegative rad/(V Å).

For formal dimensionless a,b, each grating is
1+i*interaction_constant*(a*B_j+b*D_j). The between-grating Fresnel
operator has reciprocal phase exp(-i*pi*wavelength_a*distance*k^2),
unshifted frequencies and orthonormal forward exp(-2*pi*i*k.r).
The exit follows the final grating immediately, with no terminal gap.

Returns complex128 (K,N,N), K=(s+1)*(s+2)//2. The coefficient of
a**p*b**q has row k=d*(d+1)//2+q, d=p+q<=s. No coefficient or state is
normalized; the output polynomial equals the complete ordered exit.
Invalid shapes, ranges or nonfinite data raise ValueError."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bivariate_multislice_orders(background_stack: 'np.ndarray', contrast_stack: 'np.ndarray', incident_state: 'np.ndarray', wavelength_a: float, interaction_constant: float, propagation_distance_a: float, pixel_size_a: float) -> 'np.ndarray':
    raw_B = np.asarray(background_stack)
    raw_D = np.asarray(contrast_stack)
    if np.iscomplexobj(raw_B) or np.iscomplexobj(raw_D):
        raise ValueError('projected potentials must be real')
    B = np.asarray(raw_B, dtype=float)
    D = np.asarray(raw_D, dtype=float)
    psi = np.asarray(incident_state, dtype=complex)
    if B.ndim != 3 or B.shape != D.shape or (not 1 <= B.shape[0] <= 6):
        raise ValueError('equal (s,N,N) stacks with 1<=s<=6 are required')
    (s, n, m) = B.shape
    if n != m or n < 2 or n & n - 1 or (psi.shape != (n, n)):
        raise ValueError('square power-of-two spatial arrays required')
    if not all((np.all(np.isfinite(v)) for v in (B, D, psi))) or np.linalg.norm(psi) == 0:
        raise ValueError('finite data and a nonzero incident wave are required')
    (lam, sigma, dz, dx) = map(float, (wavelength_a, interaction_constant, propagation_distance_a, pixel_size_a))
    if not np.all(np.isfinite([lam, sigma, dz, dx])) or min(lam, dz, dx) <= 0 or sigma < 0:
        raise ValueError('invalid phase or propagation scale')
    f = np.fft.fftfreq(n, d=dx)
    phase = np.exp(-1j * np.pi * lam * dz * (f[:, None] ** 2 + f[None, :] ** 2))
    coefficients = np.zeros((s + 1, s + 1, n, n), dtype=complex)
    coefficients[0, 0] = psi
    for j in range(s):
        previous = coefficients.copy()
        coefficients[1:] += 1j * sigma * B[j] * previous[:-1]
        coefficients[:, 1:] += 1j * sigma * D[j] * previous[:, :-1]
        if j < s - 1:
            coefficients = np.fft.ifft2(np.fft.fft2(coefficients, axes=(-2, -1), norm='ortho') * phase, axes=(-2, -1), norm='ortho')
    return np.asarray([coefficients[d - q, q] for d in range(s + 1) for q in range(d + 1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               's=3; n=8; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=1; n=4; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=2; n=8; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'D[:]=0\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=4; n=8; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'B[:]=0\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=3; n=4; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'D=B.copy()\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=6; n=8; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'B*=2.5; D*=3.0\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=3; n=8; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'sigma=0.0\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'},
     {'setup': 'import numpy as np\n'
               's=2; n=4; rng=np.random.default_rng(90210+s*100+n)\n'
               'psi=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); psi/=np.linalg.norm(psi)\n'
               'x=np.arange(n)[:,None]/n; y=np.arange(n)[None,:]/n\n'
               'B=np.array([650+300*np.cos(2*np.pi*(x+(j+1)*y))+80*j for j in range(s)])\n'
               'D=np.array([220*np.sin(2*np.pi*((j+1)*x-y))+90*np.cos(4*np.pi*y)+40*j+np.zeros_like(x) for j in '
               'range(s)])\n'
               'sigma=.0008\n'
               'D=-B.copy()\n'
               'import copy\n'
               'B_gold = copy.deepcopy(B)\n'
               'D_gold = copy.deepcopy(D)\n'
               'psi_gold = copy.deepcopy(psi)\n'
               'sigma_gold = copy.deepcopy(sigma)',
      'call': 'bivariate_multislice_orders(B,D,psi,.025,sigma,3.6,.8)',
      'gold_call': '_oracle_bivariate_multislice_orders(B_gold, D_gold, psi_gold, 0.025, sigma_gold, 3.6, 0.8)'}]
