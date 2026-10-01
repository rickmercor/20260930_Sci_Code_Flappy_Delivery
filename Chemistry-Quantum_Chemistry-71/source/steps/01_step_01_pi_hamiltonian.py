"""
Step 01: Pi Hamiltonian along a conrotatory or disrotatory ring closure. One-electron pi Hamiltonian of a linear polyene closing to a ring along a symmetry-constrained electrocyclic path.

In an electrocyclic ring closure the two terminal carbon atoms of a conjugated polyene approach each other while their p orbitals rotate out of the pi plane, so that at the end they point at each other and form a sigma bond. The two terminal groups can rotate in the same sense (conrotatory motion, which keeps a twofold axis) or in opposite senses (disrotatory motion, which keeps a mirror plane). Whether the pi system along the way behaves as a Hückel ring or a Möbius ring depends on this choice and on the number of pi electrons, and that is the origin of the Woodward-Hoffmann selection rules. In a minimal pi-electron model the rotation weakens the conjugation of each terminal orbital with its neighbour, while the through-space coupling between the two terminal orbitals mixes pi-type and sigma-type overlap in proportions and with a relative sign that are set by the orbital directions.

Returns
-------
numpy.ndarray of shape (n_sites, n_sites): real symmetric pi Hamiltonian in eV at progress s along the conrotatory or disrotatory path
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pi_hamiltonian(n_sites: int, s: float, mode: str, params: dict) -> "np.ndarray":
    '''Tight-binding pi Hamiltonian of an n-site polyene at progress s along a conrotatory or disrotatory ring closure.

    Parameters
    ----------
    n_sites : int
        Number of carbon sites in the open chain, even integer >= 4.
    s : float
        Reaction progress, finite, 0 <= s <= 1 (0 open chain, 1 ring closed).
    mode : str
        'con' for conrotatory or 'dis' for disrotatory closure.
    params : dict
        'beta_double', 'beta_single' : resonance integrals in eV of the formal double and single bonds;
        'r_open', 'r_closed' : distance in angstrom between the terminal sites at s = 0 and s = 1;
        'tau_pi', 'zeta_pi' : prefactor in eV and decay constant in 1/angstrom of the pi-type terminal coupling;
        'tau_sigma', 'zeta_sigma' : prefactor in eV and decay constant in 1/angstrom of the sigma-type terminal coupling.

    Returns
    -------
    h : np.ndarray
        Real symmetric (n_sites, n_sites) matrix in eV with zero diagonal. Sites 0, ..., n_sites - 1 follow the chain; the
        bond between sites i and i + 1 has resonance integral beta_double for even i and beta_single for odd i. Interior p
        orbitals point along +z. The terminal sites lie on the x axis, site 0 at negative x and site n_sites - 1 at positive
        x, at distance r = r_open - (r_open - r_closed) s. With phi = (pi/2) s, the p orbital of site 0 has direction
        a = (sin phi, 0, cos phi), tilting towards site n_sites - 1; the p orbital b of site n_sites - 1 is the image of a
        under the conserved symmetry element, the twofold rotation about the y axis for 'con' and the reflection through
        the plane x = 0 for 'dis', taken up to an overall sign. The coupling of each terminal site with its chain neighbour
        is the bond's resonance integral times the dot product of the two p directions. The coupling between sites 0 and
        n_sites - 1 is tau_pi(r) (a . b) + (tau_sigma(r) - tau_pi(r)) (a . u)(b . u), with u = (1, 0, 0) and
        tau_pi(r) = tau_pi exp(-zeta_pi (r - r_closed)), tau_sigma(r) = tau_sigma exp(-zeta_sigma (r - r_closed)).
        Results are compared only through quantities that do not depend on the sign convention of the basis orbitals.

    Raises
    ------
    ValueError
        If n_sites is not an even integer >= 4, s is not finite or outside [0, 1], or mode is not 'con' or 'dis'.
    '''
    return h

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pi_hamiltonian(n_sites: int, s: float, mode: str, params: dict) -> "np.ndarray":
    import numpy as np
    if isinstance(n_sites, bool) or int(n_sites) != n_sites or n_sites < 4 or n_sites % 2:
        raise ValueError("n_sites must be an even integer >= 4")
    if not np.isfinite(s) or s < 0.0 or s > 1.0:
        raise ValueError("s must lie in [0, 1]")
    if mode not in ("con", "dis"):
        raise ValueError("mode must be 'con' or 'dis'")
    n = int(n_sites)
    phi = 0.5 * np.pi * s
    r = params["r_open"] - (params["r_open"] - params["r_closed"]) * s
    h = np.zeros((n, n))
    for i in range(n - 1):
        h[i, i + 1] = h[i + 1, i] = params["beta_double"] if i % 2 == 0 else params["beta_single"]
    h[0, 1] = h[1, 0] = params["beta_double"] * np.cos(phi)
    h[n - 2, n - 1] = h[n - 1, n - 2] = params["beta_double"] * np.cos(phi)
    a = np.array([np.sin(phi), 0.0, np.cos(phi)])
    b = np.array([np.sin(phi) if mode == "con" else -np.sin(phi), 0.0, np.cos(phi)])
    t_pi = params["tau_pi"] * np.exp(-params["zeta_pi"] * (r - params["r_closed"]))
    t_sigma = params["tau_sigma"] * np.exp(-params["zeta_sigma"] * (r - params["r_closed"]))
    h[0, n - 1] = h[n - 1, 0] = t_pi * (a @ b) + (t_sigma - t_pi) * a[0] * b[0]
    return h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'beta_double': -2.8, 'beta_single': -2.2, 'r_open': 3.0, 'r_closed': 1.54,"
            " 'tau_pi': -2.0, 'zeta_pi': 1.4, 'tau_sigma': 3.2, 'zeta_sigma': 1.2}\n"
            "def _spec(h):\n    h = np.asarray(h, dtype=float)\n    return np.sort(np.linalg.eigvalsh(0.5 * (h + h.T)))\n")
    return [
        # --- Normal: hexatriene, conrotatory, near the top of the path ---
        {"setup": base, "call": "_spec(pi_hamiltonian(6, 0.68, 'con', dict(P)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(6, 0.68, 'con', dict(P)))", "tol": 1e-10},
        # --- Normal: hexatriene, disrotatory, same progress ---
        {"setup": base, "call": "_spec(pi_hamiltonian(6, 0.68, 'dis', dict(P)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(6, 0.68, 'dis', dict(P)))", "tol": 1e-10},
        # --- Normal: octatetraene, disrotatory, different coupling parameters ---
        {"setup": base + "Q = dict(P, beta_double=-2.6, tau_sigma=2.9, zeta_pi=1.1)\n",
         "call": "_spec(pi_hamiltonian(8, 0.37, 'dis', dict(Q)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(8, 0.37, 'dis', dict(Q)))", "tol": 1e-10},
        # --- Boundary: open chain, where both modes coincide ---
        {"setup": base, "call": "_spec(pi_hamiltonian(4, 0.0, 'con', dict(P)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(4, 0.0, 'con', dict(P)))", "tol": 1e-10},
        # --- Boundary: closed ring, terminal pi conjugation switched off ---
        {"setup": base, "call": "_spec(pi_hamiltonian(6, 1.0, 'dis', dict(P)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(6, 1.0, 'dis', dict(P)))", "tol": 1e-10},
        # --- Edge: butadiene halfway, conrotatory ---
        {"setup": base, "call": "_spec(pi_hamiltonian(4, 0.5, 'con', dict(P)))",
         "gold_call": "_spec(_oracle_pi_hamiltonian(4, 0.5, 'con', dict(P)))", "tol": 1e-10},
        # --- Error: odd chain length ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(5, 0.5, 'con', dict(P))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(pi_hamiltonian)", "gold_call": "_probe(_oracle_pi_hamiltonian)"},
    ]
