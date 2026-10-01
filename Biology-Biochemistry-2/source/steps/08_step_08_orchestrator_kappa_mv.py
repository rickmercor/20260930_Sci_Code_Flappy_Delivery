"""
Orchestrator: the full graded pipeline, from lattice to the Boltzmann slope in mV.

The orchestrator applies the complete configuration of the task: it builds the graded 4+4 checkerboard lattice, assembles the couplon generator for every pulse energy of the supplied family, obtains the exact ensemble-averaged flux course for each, extracts the peak flux from each course, fits the treatment's Boltzmann form to the peak curve, and converts the slope factor from kT to mV with the supplied factor (7.14 mV per kT, i.e. from the treatment -1 kT corresponds to +7.14 mV, and the resting midpoint is at V_rest + |eps_bar| * 7.14). Per the pipeline contract every step's reference implementation is reused via its $_oracle_$ name - the graded path never re-binds a public step entry point.

Returns
-------
kappa_mv : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def couplon_kappa_mv(n_per_class=4):
    """Boltzmann slope factor of the peak flux-voltage curve, in mV, for the graded couplon.

    Parameters
    ----------
    n_per_class : int
        Channels per class in the lattice (4 in the graded configuration). Kept as an
        argument so callers can build smaller exact configurations for smoke tests.

    Returns
    -------
    float
        The positive Boltzmann slope factor of the peak flux-voltage curve in mV,
        kappa(kT) * 7.14 mV/kT.

    Raises
    ------
    ValueError
        If n_per_class is less than 2 or not an integer.
    """
    kappa_mv = 0.0
    return kappa_mv  # placeholder to complete!

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_couplon_kappa_mv(n_per_class=4):
    """Reference implementation for couplon_kappa_mv."""
    import numpy as np

    MV_PER_KT = 7.14
    EPS_FAMILY = (0., -2., -4., -6., -7., -8., -9., -10., -12., -14.)

    if isinstance(n_per_class, bool) or int(n_per_class) != n_per_class or int(n_per_class) < 2:
        raise ValueError("n_per_class must be an integer >= 2")
    n_per_class = int(n_per_class)

    is_v, adjacency = _oracle_build_lattice_adjacency(n_per_class)

    # Flux observable per global state: open V channels count 1, open C channels count
    # R_CV = 5; inactivated channels count as closed. Rebuilt here with the same
    # mixed-radix encoding the generator uses, so the orchestrator is self-contained.
    is_v_bool = np.asarray(is_v, dtype=bool)
    dims = tuple(2 if v else 4 for v in is_v_bool)
    n_states = int(np.prod(dims))
    digits = np.stack(np.unravel_index(np.arange(n_states), dims), axis=1)
    occ = (digits == 1)
    flux_per_state = (occ & is_v_bool[None, :]).sum(axis=1) \
        + 5.0 * (occ & ~is_v_bool[None, :]).sum(axis=1)

    peaks = np.empty(len(EPS_FAMILY), dtype=float)
    for i, eps_v in enumerate(EPS_FAMILY):
        rows, cols, values = _oracle_assemble_generator(is_v, adjacency, eps_v)
        trace = _oracle_solve_master_equation(rows, cols, values, flux_per_state)
        peaks[i] = _oracle_extract_flux_features(trace)[0]

    _fmax, _eps_bar, kappa = _oracle_fit_boltzmann(np.array(EPS_FAMILY), peaks)
    return float(kappa * MV_PER_KT)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = "import numpy as np\n"
    invalid = setup + (
        "def run_model(n):\n"
        "    try:\n"
        "        couplon_kappa_mv(n)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(n):\n"
        "    try:\n"
        "        _oracle_couplon_kappa_mv(n)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Boundary: the smallest checkerboard, 2+2 (four channels, 256 states).
        {"setup": setup,
         "call": "couplon_kappa_mv(2)",
         "gold_call": "9.558848645857086",
         "tol": 1e-6},
        # Edge: intermediate couplon size 3+3.
        {"setup": setup,
         "call": "couplon_kappa_mv(3)",
         "gold_call": "9.26984439410635",
         "tol": 1e-6},
        # Invalid: non-integer size.
        {"setup": invalid,
         "call": "run_model(2.5)",
         "gold_call": "run_gold(2.5)"},
        # Invalid: size below the lattice minimum.
        {"setup": invalid,
         "call": "run_model(1)",
         "gold_call": "run_gold(1)"},
    ]
