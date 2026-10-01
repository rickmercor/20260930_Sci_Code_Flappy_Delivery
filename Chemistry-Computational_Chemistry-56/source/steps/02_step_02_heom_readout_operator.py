"""
Propagate a site-population readout of a donor-acceptor dimer backwards through its exact reduced dynamics, giving the time-dependent operator whose expectation value in any initial state is the population at a later time.

A donor-acceptor pair of pigments shares one electronic excitation between a donor site and an acceptor site that are
coupled electronically. The excitation can also recombine to the electronic ground state from either site, and the
acceptor can hand it on irreversibly to a trap (sink), which models charge separation or transfer onward. Both losses
are Markovian and are described by Lindblad jump operators of the system. Each site also couples linearly, through its
own site projector, to an independent overdamped harmonic bath whose memory time is comparable to the electronic
dynamics, so a Markovian dephasing picture is not adequate and the bath has to be treated exactly.

The hierarchical equations of motion provide that treatment for baths whose correlation functions are sums of
exponentials. They propagate the system density operator together with auxiliary density operators that carry the
bath memory, starting from a bath in thermal equilibrium that is uncorrelated with the system. Because the dynamics is
linear, the population of a chosen site at time t is a linear functional of the initial state and can be written as
the expectation value of a time-dependent operator taken in the initial state. That operator, and its rate of change,
describe how every possible initial preparation of the pair, including superpositions of donor and acceptor, feeds
the readout at a given delay.

Returns
-------
numpy.ndarray of shape (n_t, 8): [M_DD, M_AA, Re M_DA, Im M_DA] of the Heisenberg-picture site-population readout and their exact time derivatives (ps^-1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def heom_readout_operator(dimer: "np.ndarray", bath: "np.ndarray", n_matsubara: int, depth: int,
                          readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    '''Heisenberg-picture site-population readout of a dimer with Drude-Lorentz site baths, from the hierarchy.

    Parameters
    ----------
    dimer : np.ndarray
        Shape (4,), [J, gap, gamma_rec, kappa]. The electronic Hamiltonian of the singly excited pair in the basis
        (|D>, |A>) is (gap/2)(|D><D| - |A><A|) + J(|D><A| + |A><D|), with J and gap = eps_D - eps_A in cm^-1.
        Excitations recombine from each site to a ground state |g> with rate gamma_rec (jump operators
        sqrt(gamma_rec)|g><D| and sqrt(gamma_rec)|g><A|) and are trapped from the acceptor into a sink |s> with
        rate kappa (jump operator sqrt(kappa)|s><A|); both rates in ps^-1 and non-negative.
    bath : np.ndarray
        Shape (3,), [E_R, gamma_c, T]: reorganization energy (cm^-1), cutoff (cm^-1) and temperature (K) of the two
        identical, independent Drude-Lorentz baths. Site j couples to its bath through the projector |j><j|, with the
        spectral density and exponential amplitudes and rates of drude_lorentz_exponents.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term in each bath correlation function, >= 0. The omitted
        terms are dropped without any correction.
    depth : int
        Hierarchy truncation, >= 0: auxiliary density operators whose non-negative integer indices (one per site and
        exponent) sum to more than depth are set to zero. depth = 0 keeps the system density operator only, so the
        baths have no effect. The Lindblad terms act on every auxiliary density operator.
    readout_site : int
        0 for the donor population |D><D|, 1 for the acceptor population |A><A|.
    times_ps : np.ndarray
        Shape (n_t,), non-negative delays in ps. The baths are in thermal equilibrium and uncorrelated with the
        system at t = 0.

    Returns
    -------
    readout : np.ndarray
        Shape (n_t, 8), one row per delay in the input order: [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt,
        Re dM_DA/dt, Im dM_DA/dt]. M(t) is the Hermitian operator on the singly excited pair with
        Tr[M(t) rho_0] = <r| rho(t) |r> for every initial state rho_0 of the pair, where rho(t) is the system density
        operator of the truncated hierarchy and r the readout site, M_DA = <D|M(t)|A>, and the derivatives are the
        exact time derivatives of the truncated hierarchy solution (ps^-1). Energies convert to rad/ps with
        omega = 2 pi c nu, c = 2.99792458e-2 cm/ps (hbar = 1). Values are accurate to a relative 1e-9.

    Raises
    ------
    ValueError
        If depth or n_matsubara is negative, readout_site is not 0 or 1, a rate is negative, or a delay is negative.
    '''
    return readout

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _hierarchy_labels(n_modes: int, depth: int) -> list:
    """All index tuples of length n_modes with sum <= depth, ordered by total tier."""
    import itertools
    labels = []
    for tier in range(depth + 1):
        for bars in itertools.combinations(range(tier + n_modes - 1), n_modes - 1):
            prev = -1
            index = []
            for b in bars:
                index.append(b - prev - 1)
                prev = b
            index.append(tier + n_modes - 1 - prev - 1)
            labels.append(tuple(index))
    return labels


def _hierarchy_generator(dimer, bath, n_matsubara, depth):
    """Sparse generator of the truncated hierarchy on stacked row-major 2x2 blocks; block 0 is the system."""
    import numpy as np
    import scipy.sparse as sp
    to_radps = 2.0 * np.pi * 2.99792458e-2
    J, gap, g_rec, kappa = [float(v) for v in dimer]
    H = np.array([[gap / 2.0, J], [J, -gap / 2.0]]) * to_radps
    loss = np.diag([g_rec, g_rec + kappa])
    I2 = np.eye(2)
    # row-major vec: vec(A X B) = kron(A, B.T) vec(X)
    system = (-1j * (np.kron(H, I2) - np.kron(I2, H.T))
              - 0.5 * (np.kron(loss, I2) + np.kron(I2, loss.T)))
    exps = _oracle_drude_lorentz_exponents(float(bath[0]), float(bath[1]), float(bath[2]), int(n_matsubara))
    amps = exps[:, 0] + 1j * exps[:, 1]
    rates = exps[:, 2]
    modes = [(site, amps[k], rates[k]) for site in range(2) for k in range(len(rates))]
    labels = _hierarchy_labels(len(modes), depth)
    where = {lab: i for i, lab in enumerate(labels)}
    n_ado = len(labels)
    damping = np.einsum("ij,j->i", np.array(labels, dtype=float), np.array([m[2] for m in modes]))
    gen = sp.kron(sp.identity(n_ado, format="csr"), sp.csr_matrix(system), format="csr") \
        - sp.kron(sp.diags(damping), sp.identity(4), format="csr")
    rows, cols, vals = [], [], []
    for m, (site, amp, _) in enumerate(modes):
        left = np.zeros(4)
        right = np.zeros(4)
        for a in range(2):
            left[2 * site + a] = 1.0      # Q_j X
            right[2 * a + site] = 1.0     # X Q_j
        comm = np.nonzero(left - right)[0]
        for i, lab in enumerate(labels):
            if sum(lab) < depth:
                up = list(lab)
                up[m] += 1
                j = where[tuple(up)]
                rows.extend(4 * i + comm)
                cols.extend(4 * j + comm)
                vals.extend(-1j * (left - right)[comm])
            if lab[m] > 0:
                down = list(lab)
                down[m] -= 1
                j = where[tuple(down)]
                coup = -1j * lab[m] * (amp * left - np.conj(amp) * right)
                nz = np.nonzero(coup)[0]
                rows.extend(4 * i + nz)
                cols.extend(4 * j + nz)
                vals.extend(coup[nz])
    gen = gen + sp.csr_matrix((vals, (rows, cols)), shape=(4 * n_ado, 4 * n_ado))
    return gen.tocsr()


def _oracle_heom_readout_operator(dimer: "np.ndarray", bath: "np.ndarray", n_matsubara: int, depth: int,
                                  readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    import scipy.sparse.linalg as spla
    dimer = np.asarray(dimer, dtype=float)
    times = np.asarray(times_ps, dtype=float).reshape(-1)
    if depth < 0 or n_matsubara < 0:
        raise ValueError("depth and n_matsubara must be non-negative")
    if readout_site not in (0, 1):
        raise ValueError("readout_site must be 0 or 1")
    if dimer[2] < 0 or dimer[3] < 0 or np.any(times < 0):
        raise ValueError("rates and delays must be non-negative")
    gen_h = _hierarchy_generator(dimer, bath, n_matsubara, int(depth)).conj().T.tocsr()
    # adjoint propagation: Tr[M rho(t)] = y(t)^H x0 with y(t) = exp(t G^H) vec(|r><r|) in block 0
    y0 = np.zeros(gen_h.shape[0], dtype=complex)
    y0[3 * readout_site] = 1.0
    grid, inverse = np.unique(times, return_inverse=True)
    states = np.zeros((len(grid), gen_h.shape[0]), dtype=complex)
    if len(grid) > 2 and np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-9, atol=0.0):
        first = spla.expm_multiply(gen_h * grid[0], y0) if grid[0] > 0 else y0
        states[:] = spla.expm_multiply(gen_h, first, start=0.0, stop=grid[-1] - grid[0],
                                       num=len(grid), endpoint=True)
    else:
        y = y0
        t_prev = 0.0
        for i, t in enumerate(grid):
            if t > t_prev:
                y = spla.expm_multiply(gen_h * (t - t_prev), y)
            states[i] = y
            t_prev = t
    rates = (gen_h @ states.T).T
    # M_lk = conj(y_{2k+l}); M_DA = M_01 = conj(y_2)
    blocks = np.conj(states[:, :4])
    dblocks = np.conj(rates[:, :4])
    table = np.column_stack([blocks[:, 0].real, blocks[:, 3].real, blocks[:, 2].real, blocks[:, 2].imag,
                             dblocks[:, 0].real, dblocks[:, 3].real, dblocks[:, 2].real, dblocks[:, 2].imag])
    return table[inverse]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: acceptor readout of the reference dimer, one Matsubara term, shallow hierarchy ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "times = np.array([0.0, 0.03, 0.1, 0.25])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 1, 3, 1, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 1, 3, 1, times)",
            "tol": 1e-6,
        },
        # --- Boundary: depth 0 leaves only the Lindblad dynamics of the pair ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "times = np.array([0.2, 0.05])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 1, 0, 1, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 1, 0, 1, times)",
            "tol": 1e-6,
        },
        # --- Edge: donor readout, acceptor above donor, slow strong bath at 150 K with two Matsubara terms ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([70.0, -90.0, 0.05, 0.0])\n"
                     "bath = np.array([60.0, 25.0, 150.0])\n"
                     "times = np.array([0.02, 0.15, 0.4])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 2, 4, 0, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 2, 4, 0, times)",
            "tol": 1e-6,
        },
        # --- Edge: fast hot bath on a resonant homodimer with strong trapping and the Drude term only ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([120.0, 0.0, 0.0, 2.0])\n"
                     "bath = np.array([150.0, 500.0, 400.0])\n"
                     "times = np.array([0.01, 0.05, 0.12])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 0, 5, 1, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 0, 5, 1, times)",
            "tol": 1e-6,
        },
        # --- Edge: bath switched off, where a deep hierarchy must reproduce the bare Lindblad result ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([40.0, 250.0, 0.3, 0.5])\n"
                     "bath = np.array([0.0, 80.0, 300.0])\n"
                     "times = np.array([0.1, 0.6])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 1, 3, 1, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 1, 3, 1, times)",
            "tol": 1e-6,
        },
        # --- Edge: cryogenic bath with three Matsubara terms, far-detuned pair, a very short delay and repeated unsorted delays ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([45.0, 520.0, 0.02, 3.0])\n"
                     "bath = np.array([18.0, 150.0, 77.0])\n"
                     "times = np.array([0.3, 1e-4, 0.3, 0.07])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 3, 3, 1, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 3, 3, 1, times)",
            "tol": 1e-6,
        },
        # --- Normal: strongly coupled slow bath without losses on a deeper hierarchy, donor readout ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([150.0, 60.0, 0.0, 0.0])\n"
                     "bath = np.array([120.0, 30.0, 300.0])\n"
                     "times = np.array([0.05, 0.1, 0.2])\n",
            "call": "heom_readout_operator(dimer.copy(), bath.copy(), 1, 5, 0, times.copy())",
            "gold_call": "_oracle_heom_readout_operator(dimer, bath, 1, 5, 0, times)",
            "tol": 1e-6,
        },
        # --- Invalid: a readout site other than 0 or 1 must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "dimer = np.array([85.0, 140.0, 0.1, 1.0])\n"
                     "bath = np.array([27.5, 60.0, 295.0])\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(dimer, bath, 1, 2, 2, np.array([0.1]))\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(heom_readout_operator)",
            "gold_call": "run(_oracle_heom_readout_operator)",
        },
    ]
