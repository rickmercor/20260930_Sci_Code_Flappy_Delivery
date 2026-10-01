"""
Run the whole pipeline and report the transfer rate that includes the fourth-order

correction, averaged over the frozen disorder in the donor-acceptor gap.



Split the bath once: the components with short correlation times dephase the pair and go into

the lineshape, the near-static ones do not dephase anything and instead spread the gap. Then,

for each gap drawn from that spread, form the lowest-order rate and its fourth-order

correction and add them, and average the sum over the gap distribution.



The average is a Gauss-Hermite quadrature against a standard normal weight. Use the

probabilists' nodes and weights from numpy.polynomial.hermite_e.hermegauss, which integrate

against exp(-x^2/2) and whose weights sum to sqrt(2 pi); dividing them by sqrt(2 pi) makes

them a probability average. Node i samples the gap at gap_cm + sigma_static * x_i.



Both rates are linear in the population, so averaging them separately and adding is the same

as averaging their sum; what must not be done is to average the gap first and evaluate one

rate at the mean, because both rates are strongly curved in the gap.



At every node the constant that the fourth-order rate density settles on is also evaluated and

checked against minus twice the square of the second-order rate at that same gap. The two are

equal identically, because both are built from the same integrated second-order response, so

the check costs nothing and catches a wrong diagram multiplicity or a wrong overall prefactor

in either branch. A relative disagreement above 1e-6 is treated as a fault and rejected.



Formulas

--------

sigma_static from the near-static components; x_i, w_i from hermegauss(n_nodes), w_i /= sqrt(2 pi)

C(gap) = -2 k2(gap)^2 must hold at every node to a relative 1e-6

k = sum_i w_i [ k2(gap_cm + sigma_static x_i) + k4(gap_cm + sigma_static x_i) ]



Returns

-------

float, the disorder-averaged fourth-order-corrected transfer rate in ps^-1

Returns
-------
float, the disorder-averaged fourth-order-corrected transfer rate in ps^-1
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def corrected_transfer_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs: float,
                            gap_cm: float, j_cm: float, t_max_fs: float, dt_fs: float,
                            tw_max_fs: float, dtw_fs: float, n_nodes: int) -> float:
    '''Disorder-averaged transfer rate including the fourth-order correction.

    Parameters
    ----------
    sigma_donor, sigma_acceptor : array_like
        RMS site-energy fluctuation amplitudes per bath component, in cm^-1.
    tau_fs : array_like
        Bath correlation times per component, in fs.
    tau_static_fs : float
        Components with tau_j >= tau_static_fs are static: they are excluded from the
        lineshape and instead broaden the gap.
    gap_cm : float
        Mean acceptor-minus-donor gap, in cm^-1.
    j_cm : float
        Electronic coupling between the two chromophores, in cm^-1.
    t_max_fs, dt_fs : float
        Upper limit and spacing of both coherence-interval integrals, in fs.
    tw_max_fs, dtw_fs : float
        Upper limit and spacing of the waiting-time grid, in fs.
    n_nodes : int
        Number of Gauss-Hermite nodes for the static-disorder average; must be >= 1.

    Returns
    -------
    float
        The corrected transfer rate in ps^-1.

    Raises
    ------
    ValueError
        On any bath fault listed for the lineshape step, if gap_cm or j_cm is not
        finite, if either grid limit is non-positive, if either spacing is
        non-positive or exceeds its own limit, if n_nodes is not a positive integer,
        or if at any node the plateau of the fourth-order rate density disagrees with
        minus twice the square of the second-order rate by more than a relative 1e-6.
    '''
    return rate_ps


#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_corrected_transfer_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                    gap_cm, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs,
                                    n_nodes):
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 1:
        raise ValueError("n_nodes must be a positive integer")
    gap0 = float(gap_cm)
    if not np.isfinite(gap0):
        raise ValueError("gap_cm must be finite")
    sigma_static = _oracle_static_gap_width(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / np.sqrt(2.0 * np.pi)
    total = 0.0
    for x_i, w_i in zip(nodes, weights):
        gap = gap0 + sigma_static * x_i
        k2 = _oracle_second_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap, j_cm, t_max_fs, dt_fs)
        plateau = _oracle_plateau_constant(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                           gap, j_cm, t_max_fs, dt_fs)
        expected = -2.0 * k2 * k2
        if abs(plateau - expected) > 1.0e-6 * max(1.0, abs(expected)):
            raise ValueError("plateau does not match -2 k2^2 at gap %g" % gap)
        k4 = _oracle_fourth_order_rate(sigma_donor, sigma_acceptor, tau_fs, tau_static_fs,
                                       gap, j_cm, t_max_fs, dt_fs, tw_max_fs, dtw_fs)
        total += w_i * (k2 + k4)
    return float(total)


#

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = (
        'import copy\n'
        'def _fp(a):\n'
        '    a = np.asarray(a)\n'
        '    if np.iscomplexobj(a):\n'
        '        a = np.concatenate([np.real(a).ravel(), np.imag(a).ravel()])\n'
        '    a = np.asarray(a, dtype=float).ravel()\n'
        '    w = 1.0 + (np.arange(a.size) % 97) / 97.0\n'
        '    return float(np.sum(w * a / (1.0 + np.abs(a))))\n'
        'def _v(fn):\n'
        '    try:\n'
        '        out = fn()\n'
        '        if isinstance(out, (tuple, list)):\n'
        '            return float(sum(_fp(x) * (i + 1) for i, x in enumerate(out)))\n'
        '        return _fp(out)\n'
        '    except ValueError:\n'
        '        return -1.2345e6\n'
        '    except Exception:\n'
        '        return -9.8765e6\n'
        'SD = [90.0, 110.0, 70.0]\n'
        'SA = [90.0, 110.0, 70.0]\n'
        'TAU = [40.0, 180.0, 3000.0]\n'
        'FULL = dict(tau_static_fs=1000.0, gap_cm=120.0, j_cm=26.0, t_max_fs=200.0,\n'
        '            dt_fs=2.0, tw_max_fs=2400.0, dtw_fs=4.0, n_nodes=9)\n'
        'FAST = dict(tau_static_fs=1000.0, gap_cm=120.0, j_cm=26.0, t_max_fs=160.0,\n'
        '            dt_fs=8.0, tw_max_fs=800.0, dtw_fs=16.0, n_nodes=5)\n'
    )
    base = "import numpy as np\n" + guard
    return [
        # normal: the production settings of the problem, which give the reported answer
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **FULL))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **FULL))'},
        # normal: a cheap grid and a coarse average, structurally identical
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **FAST))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **FAST))'},
        # boundary: a single node collapses the average onto the mean gap
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=1)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=1)))'},
        # boundary: no static component, so the average is trivial and only memory remains
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, tau_static_fs=5000.0)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, tau_static_fs=5000.0)))'},
        # edge: a threshold that also moves the 180 fs mode into the static class
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, tau_static_fs=180.0)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, tau_static_fs=180.0)))'},
        # edge: a weak coupling, where the correction is a small fraction of the total
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, j_cm=8.0)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, j_cm=8.0)))'},
        # edge: a zero mean gap, symmetric about resonance
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, gap_cm=0.0)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, gap_cm=0.0)))'},
        # invalid: a zero node count
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=0)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=0)))'},
        # invalid: a non-integer node count
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=4.5)))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate(copy.deepcopy(SD), copy.deepcopy(SA), copy.deepcopy(TAU), **dict(FAST, n_nodes=4.5)))'},
        # invalid: an empty bath specification
        {"setup": base,
         "call": '_v(lambda: corrected_transfer_rate([], [], [], **FAST))',
         "gold_call": '_v(lambda: _oracle_corrected_transfer_rate([], [], [], **FAST))'},
    ]
