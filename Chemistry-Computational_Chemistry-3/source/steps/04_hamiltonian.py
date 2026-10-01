"""
Evaluate the total energy of the system at a six-component Cartesian phase-space point (three position components followed by their conjugate momenta), at zero total angular momentum: the rigid-body rotational energy of the fragment, the translational kinetic energy of the departing atom, and the radial and angular potentials.

The two moments of inertia of the fragment are not equal, and their ratio is physical, not incidental.

Returns
-------
float: the total energy in kcal/mol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hamiltonian(state: "np.ndarray", params: dict) -> float:
    """Evaluate the total energy of the system at a six-component Cartesian phase-space point (three position components followed by their conjugate momenta), at zero total angular momentum: the rigid-body rotational energy of the fragment, the translational kinetic energy of the departing atom, and the radial and angular potentials.

    Parameters
    ----------
    state : np.ndarray
        Six-component phase-space point (x, y, z, px, py, pz).
    params : dict
        Mapping carrying ix, iz, m, de, re, c1, c2, ve, alpha and b.

    Returns
    -------
    energy : float
        float: the total energy in kcal/mol.

    Raises
    ------
    ValueError
        if the state does not have six components, if any moment of inertia or the mass is not positive, or if a required parameter is missing.
    """
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_hamiltonian(state: "np.ndarray", params: dict) -> float:
    s = np.asarray(state, float) if not np.iscomplexobj(state) else np.asarray(state)
    if s.shape != (6,):
        raise ValueError("state must have exactly six components (X,Y,Z,pX,pY,pZ)")
    for k in ("ix", "iz", "m", "de", "re", "c1", "c2", "ve", "alpha", "b"):
        if k not in params:
            raise ValueError(f"params is missing '{k}'")
    ix, iz, m = params["ix"], params["iz"], params["m"]
    if ix <= 0 or iz <= 0 or m <= 0:
        raise ValueError("ix, iz and m must be positive")
    R, p = s[:3], s[3:]
    lx = R[1] * p[2] - R[2] * p[1]
    ly = R[2] * p[0] - R[0] * p[2]
    lz = R[0] * p[1] - R[1] * p[0]
    r = np.sqrt(R[0] ** 2 + R[1] ** 2 + R[2] ** 2)
    vch = _oracle_radial_potential(r, params["de"], params["re"], params["c1"], params["c2"])
    ang = _oracle_angular_potential(R[0], R[1], R[2], params["ve"], params["alpha"],
                                    params["re"], params["b"])
    return (lx * lx + ly * ly) / (2.0 * ix) + lz * lz / (2.0 * iz) \
           + (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) / (2.0 * m) + vch + ang

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test-case specifications for this step."""
    return [
        {"setup": "import numpy as np",
         "call": "hamiltonian(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_hamiltonian(np.array([0.0,0.0,3.65072544,0.59303049,0.0,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"setup": "import numpy as np",
         "call": "hamiltonian(np.array([0.8,0.3,3.2,0.2,-0.1,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))",
         "gold_call": "_oracle_hamiltonian(np.array([0.8,0.3,3.2,0.2,-0.1,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))"},   # normal
        {"setup": "import numpy as np",
         "call": "hamiltonian(np.array([0.8,0.3,3.2,0.2,-0.1,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_hamiltonian(np.array([0.8,0.3,3.2,0.2,-0.1,0.05]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # boundary
        {"setup": "import numpy as np",
         "call": "hamiltonian(np.array([1e-4,5e-5,3.4,0.05,-0.02,0.1]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))",
         "gold_call": "_oracle_hamiltonian(np.array([1e-4,5e-5,3.4,0.05,-0.02,0.1]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.30))"},   # edge
        {"setup": "import numpy as np\ndef _c(fn):\n    try:\n        fn(np.array([0.0,0.0,3.0,0.1,0.0]), dict(ix=2.373409, iz=4.746818, m=0.9445, de=47.0, re=1.1, c1=7.37, c2=1.61, ve=55.0, alpha=1.0, b=0.0))\n    except ValueError:\n        return 1\n    except Exception:\n        return 0\n    return 0",
         "call": "_c(hamiltonian)", "gold_call": "_c(_oracle_hamiltonian)"},   # exception contract
    ]
