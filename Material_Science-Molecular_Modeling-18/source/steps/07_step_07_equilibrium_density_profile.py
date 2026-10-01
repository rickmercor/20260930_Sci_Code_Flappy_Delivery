"""
Equilibrium density profile of the reference fluid in the slit.

At equilibrium the local density is the reservoir density reweighted by everything that makes a position different from the reservoir: the external field and the excess free-energy response of the surrounding fluid, with the reservoir residual chemical potential restoring the uniform limit far from any wall. Because the response depends on the profile, the condition is a fixed point and has to be reached by iteration; damping the update is what keeps a steeply attractive wall from driving the density through the close-packing limit on the first pass. Iterate with a damping factor of 0.10 on the density until the largest change in the undamped update falls below 1e-12 angstrom^-3. Positions the segment centre cannot reach carry zero density.

Returns
-------
numpy.ndarray, the converged density profile on the grid in angstrom^-3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equilibrium_density_profile(beta_v_ext: "np.ndarray", rho_bulk: float, radius: float, dz: float) -> "np.ndarray":
    '''Converged density profile of the hard-sphere reference in an external field.

    Parameters
    ----------
    beta_v_ext : numpy.ndarray
        Reduced external potential on the grid, positive infinity where excluded.
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.
    dz : float
        Uniform grid spacing in angstrom.

    Returns
    -------
    numpy.ndarray
        Converged density profile on the grid in angstrom^-3.

    Raises
    ------
    ValueError
        If rho_bulk is not positive, or radius or dz is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_equilibrium_density_profile(beta_v_ext: "np.ndarray", rho_bulk: float, radius: float, dz: float) -> "np.ndarray":
    if rho_bulk <= 0 or radius <= 0 or dz <= 0:
        raise ValueError("rho_bulk, radius and dz must be positive")
    bv = np.asarray(beta_v_ext, dtype=float)
    mu_res = _oracle_bulk_residual_chemical_potential(rho_bulk, radius)
    rho = np.where(np.isfinite(bv), rho_bulk, 0.0)
    for _ in range(20000):
        dF = _oracle_hard_sphere_functional_derivative(rho, radius, dz)
        with np.errstate(invalid='ignore'):
            arg = np.where(np.isfinite(bv), mu_res - dF - bv, -np.inf)
        new = rho_bulk*np.exp(np.clip(arg, -700.0, 20.0))
        err = np.max(np.abs(new - rho))
        rho = 0.90*rho + 0.10*new
        if err < 1e-12:
            break
    return rho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,14.0,1401)\nV_MODEL=steele_slit_potential(Z,14.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)\nV_GOLD=_oracle_steele_slit_potential(Z,14.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)",
            "call": "equilibrium_density_profile(V_MODEL, 0.010, 1.5675050365, 0.01)",
            "gold_call": "_oracle_equilibrium_density_profile(V_GOLD, 0.010, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,8.0,801)\nV_MODEL=steele_slit_potential(Z,8.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)\nV_GOLD=_oracle_steele_slit_potential(Z,8.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)",
            "call": "equilibrium_density_profile(V_MODEL, 0.004, 1.5675050365, 0.01)",
            "gold_call": "_oracle_equilibrium_density_profile(V_GOLD, 0.004, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,14.0,1401)\nV_MODEL=steele_slit_potential(Z,14.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)\nV_GOLD=_oracle_steele_slit_potential(Z,14.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,425.0)",
            "call": "equilibrium_density_profile(V_MODEL, 0.002, 1.5675050365, 0.01)",
            "gold_call": "_oracle_equilibrium_density_profile(V_GOLD, 0.002, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nV=np.zeros(401)",
            "call": "equilibrium_density_profile(V, 0.010, 1.5675050365, 0.02)",
            "gold_call": "_oracle_equilibrium_density_profile(V, 0.010, 1.5675050365, 0.02)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,20.0,1001)\nV_MODEL=steele_slit_potential(Z,20.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,600.0)\nV_GOLD=_oracle_steele_slit_potential(Z,20.0,3.161,488.75,3.4,28.0,0.114,3.35,0.61,600.0)",
            "call": "equilibrium_density_profile(V_MODEL, 0.005, 1.45, 0.02)",
            "gold_call": "_oracle_equilibrium_density_profile(V_GOLD, 0.005, 1.45, 0.02)",
        },
    ]
