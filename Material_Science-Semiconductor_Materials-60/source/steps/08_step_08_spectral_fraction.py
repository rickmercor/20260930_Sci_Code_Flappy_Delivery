"""
Integrate the normalized exciton oscillator measure over an energy window.

$$f_s=|A_s^\dagger d|^2,\quad R=\frac{\sum_s f_s[\arctan((h-E_s)/b)-\arctan((l-E_s)/b)]}{\pi\sum_s f_s}.$$

Returns
-------
Float oscillator fraction in [0,1] up to roundoff.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_fraction(hamiltonian, dipole, window, broadening):
    """Integrate the normalized exciton oscillator measure over an energy window.

    hamiltonian is finite Hermitian complex (n,n), n>=1; dipole is finite
    nonzero complex (n,). window=(lo,hi) is finite real with lo<hi;
    broadening>0 is finite real. Let H A_s=E_s A_s, ||A_s||=1,
    f_s=|A_s^dagger dipole|^2. Use the normalized Lorentzian of half-width
    broadening on the entire real energy line. The window fraction is
    sum_s f_s*[atan((hi-E_s)/b)-atan((lo-E_s)/b)]/(pi*sum_s f_s).
    Keep complex conjugation and sum incoherently over different eigenstates.

    Returns
    -------
    result
        Float oscillator fraction in [0,1] up to roundoff.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-Hermitian H (atol=2e-10,
        rtol=0), zero dipole, complex/invalid window or nonpositive broadening.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spectral_fraction(hamiltonian, dipole, window, broadening):
    """Integrate the normalized exciton oscillator measure over an energy window.

    hamiltonian is finite Hermitian complex (n,n), n>=1; dipole is finite
    nonzero complex (n,). window=(lo,hi) is finite real with lo<hi;
    broadening>0 is finite real. Let H A_s=E_s A_s, ||A_s||=1,
    f_s=|A_s^dagger dipole|^2. Use the normalized Lorentzian of half-width
    broadening on the entire real energy line. The window fraction is
    sum_s f_s*[atan((hi-E_s)/b)-atan((lo-E_s)/b)]/(pi*sum_s f_s).
    Keep complex conjugation and sum incoherently over different eigenstates.

    Returns
    -------
    result
        Float oscillator fraction in [0,1] up to roundoff.

    Raises
    ------
    ValueError
        For wrong shapes, nonfinite inputs, non-Hermitian H (atol=2e-10,
        rtol=0), zero dipole, complex/invalid window or nonpositive broadening.
    """
    import numpy as np
    try:
        h = np.asarray(hamiltonian, dtype=complex)
        d = np.asarray(dipole, dtype=complex)
        if np.iscomplexobj(window) or np.iscomplexobj(broadening):
            raise ValueError('real window')
        w = np.asarray(window, dtype=float)
        b = np.asarray(broadening, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError('numeric inputs') from e
    if h.ndim != 2 or len(h) == 0 or h.shape[1] != len(h) or (d.shape != (len(h),)) or (w.shape != (2,)) or (b.ndim != 0) or any((not np.all(np.isfinite(v)) for v in (h, d, w, b))) or (w[0] >= w[1]) or (b <= 0) or (np.linalg.norm(d) == 0) or (not np.allclose(h, h.conj().T, atol=2e-10, rtol=0)):
        raise ValueError('bounds or Hermiticity')
    e, a = np.linalg.eigh(h)
    f = abs(a.conj().T @ d) ** 2
    integrals = (np.arctan((w[1] - e) / b) - np.arctan((w[0] - e) / b)) / np.pi
    return float(f @ integrals / f.sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=(z+z.conj().T)/2;dip=rng.normal(size=6)+1j*rng.normal(size=6);window=[-.8,1.2];b=.17\n'
               '\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(spectral_fraction(h,dip,window,b))',
      'gold_call': 'numeric(_oracle_spectral_fraction(h,dip,window,b))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=(z+z.conj().T)/2;dip=rng.normal(size=6)+1j*rng.normal(size=6);window=[-.8,1.2];b=.17\n'
               'h=np.eye(6)*.4\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(spectral_fraction(h,dip,window,b))',
      'gold_call': 'numeric(_oracle_spectral_fraction(h,dip,window,b))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=(z+z.conj().T)/2;dip=rng.normal(size=6)+1j*rng.normal(size=6);window=[-.8,1.2];b=.17\n'
               'dip*=3j;window=[2.,4.]\n'
               '\n'
               'def numeric(value):\n'
               '    if isinstance(value, dict):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(value[k]) for k in sorted(value)])\n'
               '    if isinstance(value, (tuple,list)):\n'
               '        if not value: return np.empty(0)\n'
               '        return np.concatenate([numeric(v) for v in value])\n'
               '    return np.asarray(value).reshape(-1)\n',
      'call': 'numeric(spectral_fraction(h,dip,window,b))',
      'gold_call': 'numeric(_oracle_spectral_fraction(h,dip,window,b))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3);z=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6));h=(z+z.conj().T)/2;dip=rng.normal(size=6)+1j*rng.normal(size=6);window=[-.8,1.2];b=.17\n'
               'b=0.\n'
               '\n'
               'def raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    raise AssertionError("ValueError required")\n',
      'call': 'raises_value_error(spectral_fraction, h,dip,window,b)',
      'gold_call': 'raises_value_error(_oracle_spectral_fraction, h,dip,window,b)'}]
