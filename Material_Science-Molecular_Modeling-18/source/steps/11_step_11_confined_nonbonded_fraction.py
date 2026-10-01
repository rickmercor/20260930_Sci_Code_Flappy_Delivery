"""
Average hydrogen-bonding state of water held in a carbon slit pore.

Chaining the pipeline gives the quantity the task asks for. The segment size follows from the Mie potential at the working temperature, the walls impose their field on the pore grid, the reference fluid relaxes to its equilibrium profile in that field, and the association machinery then reads off how many sites stay free at each position. Because the free fraction varies strongly across the pore, the single number that characterises the confined fluid is the average over the sites actually present, that is, the profile-weighted mean of the free fraction taken over the whole pore with the trapezoidal rule on the same grid. Water is described with a Mie segment of diameter 3.161 angstrom, well depth 488.75 kelvin and exponents 6 and 52.367, carrying four association sites of depth 1210 kelvin placed 0.4 diameters from the centre with a square-well range of 0.5834 diameters; the graphite walls use a solid density of 0.114 per cubic angstrom, interlayer spacing 3.35 angstrom, site diameter 3.4 angstrom, well depth 28 kelvin and a third-term coefficient of 0.61.

Returns
-------
float, the profile-weighted mean fraction of non-bonded association sites
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def confined_nonbonded_fraction(pore_width: float, rho_bulk: float, temperature: float, n_grid: int) -> float:
    '''Profile-weighted mean fraction of non-bonded association sites in the pore.

    Parameters
    ----------
    pore_width : float
        Wall to wall separation in angstrom.
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    temperature : float
        Absolute temperature in kelvin.
    n_grid : int
        Number of equally spaced grid points spanning the pore, endpoints included.

    Returns
    -------
    float
        Profile-weighted mean fraction of non-bonded sites, between zero and one.

    Raises
    ------
    ValueError
        If pore_width, rho_bulk or temperature is not positive, or n_grid is below two.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_confined_nonbonded_fraction(pore_width: float, rho_bulk: float, temperature: float, n_grid: int) -> float:
    if pore_width <= 0 or rho_bulk <= 0 or temperature <= 0:
        raise ValueError("pore_width, rho_bulk and temperature must be positive")
    if n_grid < 2:
        raise ValueError("n_grid must be at least two")
    sigma, eps, la, lr = 3.161, 488.75, 6.0, 52.367
    eps_hb, rd, rc = 1210.0, 0.4*3.161, 0.5834*3.161
    sig_s, eps_s, rho_s, del_s, alpha = 3.4, 28.0, 0.114, 3.35, 0.61

    radius = _oracle_barker_henderson_radius(sigma, eps, la, lr, temperature)
    d = 2.0*radius
    z = np.linspace(0.0, pore_width, int(n_grid))
    dz = z[1] - z[0]
    bv = _oracle_steele_slit_potential(z, pore_width, sigma, eps, sig_s, eps_s,
                               rho_s, del_s, alpha, temperature)
    rho = _oracle_equilibrium_density_profile(bv, rho_bulk, radius, dz)
    w = _oracle_fmt_weighted_densities(rho, radius, dz)
    zg = _oracle_contact_pair_correlation(w, d)
    delta = _oracle_association_strength(zg[1], d, sigma, eps_hb, rd, rc, temperature)
    chi = _oracle_nonbonded_site_fraction(w[0], zg[0], delta)
    return float(np.trapezoid(rho*chi, z)/np.trapezoid(rho, z))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "confined_nonbonded_fraction(14.0, 0.010, 425.0, 1401)",
            "gold_call": "_oracle_confined_nonbonded_fraction(14.0, 0.010, 425.0, 1401)",
        },
        {
            "setup": "import numpy as np",
            "call": "confined_nonbonded_fraction(14.0, 0.004, 425.0, 1401)",
            "gold_call": "_oracle_confined_nonbonded_fraction(14.0, 0.004, 425.0, 1401)",
        },
        {
            "setup": "import numpy as np",
            "call": "confined_nonbonded_fraction(8.0, 0.004, 425.0, 801)",
            "gold_call": "_oracle_confined_nonbonded_fraction(8.0, 0.004, 425.0, 801)",
        },
        {
            "setup": "import numpy as np",
            "call": "confined_nonbonded_fraction(20.0, 0.010, 500.0, 1001)",
            "gold_call": "_oracle_confined_nonbonded_fraction(20.0, 0.010, 500.0, 1001)",
        },
        {
            "setup": "import numpy as np",
            "call": "confined_nonbonded_fraction(14.0, 0.010, 425.0, 201)",
            "gold_call": "_oracle_confined_nonbonded_fraction(14.0, 0.010, 425.0, 201)",
        },
    ]
