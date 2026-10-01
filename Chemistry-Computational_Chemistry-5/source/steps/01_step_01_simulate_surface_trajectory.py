"""
Step name: 01_simulate_surface_trajectory
Step description: Integrate overdamped Langevin dynamics of one tagged adsorbed hydrogen atom on the reduced free-energy surface and return its frame-by-frame coordinates.

Step scientific background: Coarse-grained catalytic trajectories are propagated at constant temperature on a free-energy surface whose reaction coordinate is an adsorbate-adsorbate separation and whose second coordinate places the adsorbate on a terrace or on an under-coordinated edge. Overdamped Langevin dynamics obey detailed balance, so the trajectory samples the Boltzmann distribution of that surface.

Returns
-------
np.ndarray: float trajectory of shape (n_frames, 3) with rows (r1, r2, s).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def simulate_surface_trajectory(n_frames: int, seed: int, dt_ps: float = 0.01,
                                d_bond: float = 0.5, d_lateral: float = 2.0,
                                kt_ev: float = 0.0387780) -> np.ndarray:
    """Propagate the tagged hydrogen atom and record one row per frame.

    The state is ``(r1, r2, s)``: the distances in angstrom from the tagged
    hydrogen to its two hydrogen neighbours, and its lateral coordinate in
    angstrom along a Rh row of period ``L = 8.10``. With
    ``b(r) = 0.040 * exp(-(r - 1.22) ** 2 / (2 * 0.30 ** 2))`` and
    ``e(s) = 0.5 * (1 + cos(2 * pi * s / L))`` the free energy in eV is

        U = sum_i [ 0.20 * (0.55 / r_i) ** 6
                    + 0.5 * 0.12 * (r_i - 2.60) ** 2
                    - 0.25 * exp(-(r_i - 0.80) ** 2 / (2 * 0.25 ** 2))
                    + b(r_i) * (1 + 3.0 * e(s)) ]
            + 0.020 * (1 - cos(6 * pi * s / L))
            + 0.045 * (1 - cos(2 * pi * s / L))
            + 0.06 * exp(-((r1 - 0.80) ** 2 + (r2 - 0.80) ** 2)
                         / (2 * 0.25 ** 2))

    so that ``s = 0`` is the under-coordinated edge site, where the saddle of
    the separation coordinate is raised fourfold.

    Conventions fixed by this step. The trajectory starts from
    ``(2.60, 2.60, 0.0)``, which is recorded as frame 0. The Gaussian
    increments are drawn once as ``rng.standard_normal((n_frames, 3))`` from
    ``np.random.default_rng(seed)``, and row ``t`` of that array advances the
    state from frame ``t`` to frame ``t + 1``. Each coordinate takes one
    Euler-Maruyama step of duration ``dt_ps``, with a mobility equal to that
    coordinate's diffusion constant divided by ``kt_ev``; ``r1`` and ``r2`` use
    ``d_bond`` and ``s`` uses ``d_lateral``. After every step ``s`` is reduced
    modulo ``L``, while ``r1`` and ``r2`` are left unwrapped.

    Parameters
    ----------
    n_frames : int
        Number of recorded frames, at least 2.
    seed : int
        Seed of the increment generator.
    dt_ps : float
        Frame spacing in picoseconds, positive.
    d_bond : float
        Diffusion constant of each hydrogen-hydrogen distance in angstrom^2
        per picosecond, positive.
    d_lateral : float
        Diffusion constant of the lateral coordinate in angstrom^2 per
        picosecond, positive.
    kt_ev : float
        Thermal energy in eV, positive.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_frames, 3)`` whose rows are ``(r1, r2, s)``.

    Raises
    ------
    ValueError
        If ``n_frames`` is not an integer of at least 2, if ``seed`` is not an
        integer, if any of ``dt_ps``, ``d_bond``, ``d_lateral`` or ``kt_ev`` is
        not a positive finite float, or if the integration drives a
        hydrogen-hydrogen distance to a non-positive value.
    """
    return trajectory

# EXPECTED RETURN
#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _surface_constants() -> tuple:
    """Return the parameters of the reduced free-energy surface."""
    return (8.10,        # lattice period, angstrom
            0.20, 0.55,  # short-range repulsion strength (eV) and range (angstrom)
            0.12, 2.60,  # separated-basin stiffness (eV/angstrom^2) and centre
            0.25, 0.80, 0.25,   # molecular well depth (eV), centre, width
            0.040, 1.22, 0.30,  # saddle height (eV), centre, width
            0.020, 0.045,       # terrace corrugation and edge depth, eV
            3.0,                # edge enhancement of the saddle
            0.06)               # double-bond exclusion penalty, eV


def _is_integer(value) -> bool:
    """Return True for a genuine integer, rejecting booleans."""
    import numpy as np
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _is_positive_float(value) -> bool:
    """Return True for a finite positive real scalar."""
    import numpy as np
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return bool(np.isfinite(float(value))) and float(value) > 0.0


def _surface_gradient(r1: float, r2: float, s: float) -> tuple:
    """Return the three partial derivatives of the free-energy surface."""
    import numpy as np

    (period, c_rep, r_core, k_sep, r_sep, d_mol, r_mol, w_mol,
     b_sad, r_sad, w_sad, a_cor, a_edge, c_edge, g_excl) = _surface_constants()

    omega = 2.0 * np.pi / period
    edge = 0.5 * (1.0 + np.cos(omega * s))
    d_edge = -0.5 * omega * np.sin(omega * s)
    scale = 1.0 + c_edge * edge

    grads = []
    saddle_s = 0.0
    for r in (r1, r2):
        saddle = b_sad * np.exp(-((r - r_sad) ** 2) / (2.0 * w_sad ** 2))
        well = d_mol * np.exp(-((r - r_mol) ** 2) / (2.0 * w_mol ** 2))
        grads.append(-6.0 * c_rep * r_core ** 6 / r ** 7
                     + k_sep * (r - r_sep)
                     + well * (r - r_mol) / w_mol ** 2
                     - saddle * (r - r_sad) / w_sad ** 2 * scale)
        saddle_s += saddle * c_edge * d_edge

    exclusion = g_excl * np.exp(-(((r1 - r_mol) ** 2) + ((r2 - r_mol) ** 2))
                                / (2.0 * w_mol ** 2))
    grads[0] -= exclusion * (r1 - r_mol) / w_mol ** 2
    grads[1] -= exclusion * (r2 - r_mol) / w_mol ** 2

    grad_s = (a_cor * 3.0 * omega * np.sin(3.0 * omega * s)
              + a_edge * omega * np.sin(omega * s)
              + saddle_s)
    return grads[0], grads[1], grad_s


import numpy as np
def _oracle_simulate_surface_trajectory(n_frames: int, seed: int, dt_ps: float = 0.01,
                                        d_bond: float = 0.5, d_lateral: float = 2.0,
                                        kt_ev: float = 0.0387780) -> np.ndarray:
    """Reference implementation (Euler-Maruyama on the reduced surface)."""
    import numpy as np

    if not (_is_integer(n_frames) and int(n_frames) >= 2):
        raise ValueError("n_frames must be an integer of at least 2")
    if not _is_integer(seed):
        raise ValueError("seed must be an integer")
    for name, value in (("dt_ps", dt_ps), ("d_bond", d_bond),
                        ("d_lateral", d_lateral), ("kt_ev", kt_ev)):
        if not _is_positive_float(value):
            raise ValueError(f"{name} must be a positive finite float")

    period = _surface_constants()[0]
    start = _surface_constants()[4]
    n_frames = int(n_frames)
    dt_ps, kt_ev = float(dt_ps), float(kt_ev)
    d_bond, d_lateral = float(d_bond), float(d_lateral)

    noise = np.random.default_rng(int(seed)).standard_normal((n_frames, 3))
    mob_bond = d_bond * dt_ps / kt_ev
    mob_lateral = d_lateral * dt_ps / kt_ev
    amp_bond = np.sqrt(2.0 * d_bond * dt_ps)
    amp_lateral = np.sqrt(2.0 * d_lateral * dt_ps)

    trajectory = np.empty((n_frames, 3), dtype=float)
    r1 = r2 = start
    s = 0.0
    for t in range(n_frames):
        trajectory[t, 0] = r1
        trajectory[t, 1] = r2
        trajectory[t, 2] = s
        if not (r1 > 0.0 and r2 > 0.0):
            raise ValueError("integration produced a non-positive hydrogen separation")
        g1, g2, gs = _surface_gradient(r1, r2, s)
        r1 = r1 - mob_bond * g1 + amp_bond * noise[t, 0]
        r2 = r2 - mob_bond * g2 + amp_bond * noise[t, 1]
        s = (s - mob_lateral * gs + amp_lateral * noise[t, 2]) % period
    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    digest = (
        "import numpy as np\n"
        "def _sig(a):\n"
        "    f = np.asarray(a, dtype=float).ravel()\n"
        "    if f.size == 0:\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(f.size, dtype=float) + 1.0)\n"
        "    return float(np.dot(f, w) + np.abs(f).mean() + 1000.0 * f[0])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": digest,
            "call": "_sig(simulate_surface_trajectory(4000, 7))",
            "gold_call": "_sig(_oracle_simulate_surface_trajectory(4000, 7))",
        },
        {
            "setup": digest,
            "call": "_sig(simulate_surface_trajectory(2500, 20260212, dt_ps=0.005, d_bond=0.8))",
            "gold_call": "_sig(_oracle_simulate_surface_trajectory(2500, 20260212, dt_ps=0.005, d_bond=0.8))",
        },
        {
            "setup": digest,
            "call": "_sig(simulate_surface_trajectory(1500, 3, kt_ev=0.0258520, d_lateral=1.0))",
            "gold_call": "_sig(_oracle_simulate_surface_trajectory(1500, 3, kt_ev=0.0258520, d_lateral=1.0))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(simulate_surface_trajectory(2, 0)[0, 0] + simulate_surface_trajectory(2, 0)[1, 2])",
            "gold_call": "float(_oracle_simulate_surface_trajectory(2, 0)[0, 0] + _oracle_simulate_surface_trajectory(2, 0)[1, 2])",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.ptp(simulate_surface_trajectory(3000, 11)[:, 2]) + np.mean(simulate_surface_trajectory(3000, 11)[:, 0]))",
            "gold_call": "float(np.ptp(_oracle_simulate_surface_trajectory(3000, 11)[:, 2]) + np.mean(_oracle_simulate_surface_trajectory(3000, 11)[:, 0]))",
        },
        {
            "setup": status,
            "call": "_status(lambda: simulate_surface_trajectory(1, 0))",
            "gold_call": "_status(lambda: _oracle_simulate_surface_trajectory(1, 0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: simulate_surface_trajectory(100, 0, dt_ps=0.0))",
            "gold_call": "_status(lambda: _oracle_simulate_surface_trajectory(100, 0, dt_ps=0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: simulate_surface_trajectory(100, 0.5))",
            "gold_call": "_status(lambda: _oracle_simulate_surface_trajectory(100, 0.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: simulate_surface_trajectory(100, 0, kt_ev=-1.0))",
            "gold_call": "_status(lambda: _oracle_simulate_surface_trajectory(100, 0, kt_ev=-1.0))",
        },
    ]
