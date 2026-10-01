"""
Scale the paper's mirror-symmetric nonlinear-plasma-lens lattice to one energy and dipole exponent.

Let r=E/E_r. Use the seven frozen laws L=L_r r^(1/2), l=l_r r^(1/2), beta=beta_r r^(1/2), B=B_r r^(-p), tau_x=tau_r r^p, D_x=1/tau_x, and R56=R56_r r^(-(2p+1/2)). Here E is in GeV; L,l,beta,D_x,R56 are in m; B is in T; and tau_x is in m^-1. Preserve the signs of tau_x and R56. The supported numerical domain is the subset of finite inputs for which all seven returned IEEE-754 double-precision values are finite, with L,l,beta,B positive and tau_x,D_x nonzero.

Returns
-------
Return a length-7 NumPy float array [L,l,beta,B,tau_x,D_x,R56] in the documented order and units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def scale_staging_lattice(energy_gev, exponent, reference):
    """Return scaled lattice quantities at one energy.

    Args:
        energy_gev: Positive finite beam energy in GeV.
        exponent: Finite dipole-field exponent p.
        reference: Length-7 numerical array [E_r,L_r,l_r,beta_r,B_r,tau_r,R56_r]
            in [GeV,m,m,m,T,m^-1,m].

    Returns:
        numpy.ndarray: Length-7 float array [L,l,beta,B,tau_x,D_x,R56] in
        [m,m,m,T,m^-1,m,m].

    Raises:
        ValueError: If energy is nonpositive/nonfinite, exponent is nonfinite,
            reference is not a finite length-7 physical array (E_r,L_r,l_r,
            beta_r,B_r positive and tau_r nonzero), or the scaled result is
            outside the finite supported numerical domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_scale_staging_lattice(energy_gev, exponent, reference):
    import math
    import numpy as np
    energy_gev = float(energy_gev)
    exponent = float(exponent)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (7,) or not np.all(np.isfinite(ref)):
        raise ValueError("reference must be a finite length-7 array")
    if not math.isfinite(energy_gev) or energy_gev <= 0 or not math.isfinite(exponent):
        raise ValueError("energy must be positive and exponent finite")
    er, lr, ellr, betar, br, taur, r56r = ref
    if min(er, lr, ellr, betar, br) <= 0 or taur == 0:
        raise ValueError("invalid physical reference")
    ratio = energy_gev / er
    try:
        root = math.sqrt(ratio)
        length = lr * root
        ell = ellr * root
        beta = betar * root
        field = br * ratio ** (-exponent)
        tau = taur * ratio ** exponent
        dispersion = 1.0 / tau
        r56 = r56r * ratio ** (-(2.0 * exponent + 0.5))
    except (OverflowError, ZeroDivisionError):
        raise ValueError("scaled values exceed the supported finite domain")
    out = np.array([length, ell, beta, field, tau, dispersion, r56], dtype=float)
    if (not np.all(np.isfinite(out)) or min(length, ell, beta, field) <= 0
            or tau == 0 or dispersion == 0):
        raise ValueError("scaled values exceed the supported finite domain")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])", "call":"scale_staging_lattice(50.,.6,ref)", "gold_call":"_oracle_scale_staging_lattice(50.,.6,ref)"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])", "call":"scale_staging_lattice(500.,.6,ref)", "gold_call":"_oracle_scale_staging_lattice(500.,.6,ref)"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])", "call":"scale_staging_lattice(5000.,0.,ref)", "gold_call":"_oracle_scale_staging_lattice(5000.,0.,ref)"},
        {"setup":"import numpy as np\nref=np.array([20.,1.4,2.1,.02,.8,-40.,-2e-4])", "call":"scale_staging_lattice(5.,-.2,ref)", "gold_call":"_oracle_scale_staging_lattice(5.,-.2,ref)"},
        {"setup":"import numpy as np\nref=np.array([73.,.9,7.2,.013,1.7,25.,3e-5])", "call":"scale_staging_lattice(73.,1.1,ref)", "gold_call":"_oracle_scale_staging_lattice(73.,1.1,ref)"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])", "call":"scale_staging_lattice(137.5,.48,ref)", "gold_call":"_oracle_scale_staging_lattice(137.5,.48,ref)"},
        {"setup":"import numpy as np\nref=np.array([12.,.4,1.9,.007,.35,-120.,-7e-4])", "call":"scale_staging_lattice(1e5,1/3,ref)", "gold_call":"_oracle_scale_staging_lattice(1e5,1/3,ref)"},
        {"setup":"import numpy as np\ndef _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0\nref=np.array([50.,1.,2.,.03,1.,-60.,-1e-4])", "call":"_ve(lambda: scale_staging_lattice(0.,.5,ref))", "gold_call":"_ve(lambda: _oracle_scale_staging_lattice(0.,.5,ref))"}
    ]
