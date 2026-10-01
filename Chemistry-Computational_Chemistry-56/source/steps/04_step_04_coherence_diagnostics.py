"""
Turn a Heisenberg-picture population readout and its rate of change into four coherence diagnostics for the preparation of a donor-acceptor pair.

Whether site coherence in the initial state of an excitonic dimer can matter for a given readout is a statement about
the readout operator, not about any particular state. Comparing each preparation with the preparation obtained by
deleting its site-basis coherences removes population effects exactly, so the largest possible difference between
the two measures how much coherence could ever change the signal. A second benchmark compares the best preparation
of any kind with the best preparation that carries no site coherence; the two are optimized independently, so this
measures a different advantage. The preparation that realizes the largest coherence gain is an equal-weight
superposition of donor and acceptor with a definite relative phase, and the speed at which the coherence-induced
change can vary at a given delay follows from the time derivative of the readout operator.

All four quantities follow from the elements of a 2 x 2 Hermitian operator and its derivative, optimized over the
states of the singly excited pair.

Returns
-------
numpy.ndarray of shape (n, 4): rows [C, Pi, phi, Gamma] for each readout row
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coherence_diagnostics(readout: "np.ndarray") -> "np.ndarray":
    '''Coherence impact, unpaired benchmark, optimal relative phase and instantaneous rate for readout operators.

    Parameters
    ----------
    readout : np.ndarray
        Shape (n, 8), rows [M_DD, M_AA, Re M_DA, Im M_DA, dM_DD/dt, dM_AA/dt, Re dM_DA/dt, Im dM_DA/dt] of a Hermitian
        2 x 2 operator M on the basis (|D>, |A>), with M_DA = <D|M|A>, and of its time derivative, as returned by
        heom_readout_operator or ensemble_readout_operator.

    Returns
    -------
    diagnostics : np.ndarray
        Shape (n, 4), rows [C, Pi, phi, Gamma]. With rho ranging over all density operators of the pair and
        G(rho) the same operator with its off-diagonal elements set to zero:
        C = sup_rho |Tr[M (rho - G(rho))]|;
        Pi = sup_rho |Tr[M rho]| - sup_sigma |Tr[M sigma]|, where sigma ranges over the diagonal density operators;
        phi in (-pi, pi] is the relative phase for which the pure state (|D> + exp(i phi)|A>)/sqrt(2) maximizes
        Tr[M (rho - G(rho))], with phi = 0 when M_DA = 0;
        Gamma = sup_rho |Tr[(dM/dt) (rho - G(rho))]|, in the units of the derivative columns.

    Raises
    ------
    ValueError
        If readout is not a two-dimensional array with 8 columns.
    '''
    return diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_coherence_diagnostics(readout: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rows = np.asarray(readout, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("readout must have shape (n, 8)")
    m_da = rows[:, 2] + 1j * rows[:, 3]
    coherence = np.abs(m_da)
    # eigenvalues of [[a, m], [m*, b]] are (a + b)/2 +- sqrt(((a - b)/2)^2 + |m|^2)
    mean = 0.5 * (rows[:, 0] + rows[:, 1])
    radius = np.sqrt((0.5 * (rows[:, 0] - rows[:, 1])) ** 2 + coherence ** 2)
    best_any = np.maximum(np.abs(mean + radius), np.abs(mean - radius))
    best_free = np.maximum(np.abs(rows[:, 0]), np.abs(rows[:, 1]))
    unpaired = best_any - best_free
    # Tr[M (rho - G(rho))] = Re(M_DA exp(i phi)) for the equal-weight superposition
    phase = np.where(coherence > 0.0, -np.angle(m_da), 0.0)
    phase = np.where(phase <= -np.pi, phase + 2.0 * np.pi, phase)
    rate = np.hypot(rows[:, 6], rows[:, 7])
    return np.column_stack([coherence, unpaired, phase, rate])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: acceptor readout rows typical of a dimer a few hundred femtoseconds after excitation ---
        {
            "setup": "import numpy as np\n"
                     "rows = np.array([[0.3521, 0.6034, -0.0935, 0.0471, 0.52, -0.77, -0.0412, -0.0366],\n"
                     "                 [0.1905, 0.7828, -0.1407, 0.3241, 5.1, -4.2, -2.7, 1.3]])\n",
            "call": "coherence_diagnostics(rows.copy())",
            "gold_call": "_oracle_coherence_diagnostics(rows)",
            "tol": 1e-9,
        },
        # --- Boundary: no coherence element, where the phase convention and a zero unpaired benchmark apply ---
        {
            "setup": "import numpy as np\n"
                     "rows = np.array([[0.2, 0.7, 0.0, 0.0, 0.1, 0.3, 0.0, -0.4],\n"
                     "                 [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])\n",
            "call": "coherence_diagnostics(rows.copy())",
            "gold_call": "_oracle_coherence_diagnostics(rows)",
            "tol": 1e-9,
        },
        # --- Edge: negative real coherence element (phase exactly pi) and an indefinite operator ---
        {
            "setup": "import numpy as np\n"
                     "rows = np.array([[0.1, -0.3, -0.25, 0.0, 0.0, 0.0, 1.0, 0.0],\n"
                     "                 [-0.6, -0.1, 0.05, -0.2, -1.0, 2.0, 0.3, 0.4],\n"
                     "                 [0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, -2e-3, 0.0]])\n",
            "call": "coherence_diagnostics(rows.copy())",
            "gold_call": "_oracle_coherence_diagnostics(rows)",
            "tol": 1e-9,
        },
        # --- Invalid: rows with the wrong number of columns must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def run(fn):\n"
                     "    try:\n"
                     "        fn(np.zeros((3, 4)))\n"
                     "        return 0\n"
                     "    except ValueError:\n"
                     "        return 1\n",
            "call": "run(coherence_diagnostics)",
            "gold_call": "run(_oracle_coherence_diagnostics)",
        },
    ]
