"""
Construct the source-protocol pairwise HSQC separation and assignment-cost matrices from two active peak sets.

HSQC spectra are unordered peak coordinates, and the two axes use distinct supplied scales and functional ranges. Preserve axis and matrix alignment; the pinned source defines the separation, boundary treatment, and remote-pair cost convention.

Returns
-------
A tuple containing the source-defined separation matrix, assignment-cost matrix, and scalar functional radius.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hsqc_cost_matrices(
    spectrum_a: "np.ndarray",
    spectrum_b: "np.ndarray",
    sigma_h: float,
    sigma_c: float,
    functional_h: float,
    functional_c: float,
    penalty_factor: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Return source-defined pairwise separation and assignment-cost matrices.

    Rows are unordered ``(1H, 13C)`` peaks. Preserve axis order, matrix
    alignment, and the supplied scale, range, and penalty parameters.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_hsqc_cost_matrices(
    spectrum_a: "np.ndarray",
    spectrum_b: "np.ndarray",
    sigma_h: float,
    sigma_c: float,
    functional_h: float,
    functional_c: float,
    penalty_factor: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    import numpy as np

    a = np.asarray(spectrum_a, dtype=float)
    b = np.asarray(spectrum_b, dtype=float)

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[1:] != (2,)
        or b.shape[1:] != (2,)
    ):
        raise ValueError("spectra must have shape (n_peaks, 2)")

    if len(a) == 0 or len(b) == 0:
        raise ValueError("spectra must be nonempty")

    if min(sigma_h, sigma_c) <= 0 or min(functional_h, functional_c) < 0:
        raise ValueError("invalid scale")

    delta_h = (a[:, None, 0] - b[None, :, 0]) / sigma_h
    delta_c = (a[:, None, 1] - b[None, :, 1]) / sigma_c
    distances = np.hypot(delta_h, delta_c)

    tolerance = float(
        np.hypot(
            functional_h / sigma_h,
            functional_c / sigma_c,
        )
    )

    costs = np.where(
        distances <= tolerance,
        distances,
        distances + penalty_factor,
    )

    return distances, costs, tolerance

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' d,c,t=x; return np.concatenate((d.ravel(),c.ravel(),np.array([t])))\n'
               'a=np.array([[1.0,10.0]]); b=np.array([[1.01,10.2]])',
      'call': 'pack(hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(0.01), '
              'copy.deepcopy(0.2), copy.deepcopy(0.5), copy.deepcopy(2.5), copy.deepcopy(1.0)))',
      'gold_call': 'pack(_oracle_hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), '
                   'copy.deepcopy(0.01), copy.deepcopy(0.2), copy.deepcopy(0.5), '
                   'copy.deepcopy(2.5), copy.deepcopy(1.0)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' d,c,t=x; return np.concatenate((d.ravel(),c.ravel(),np.array([t])))\n'
               'a=np.array([[1.0,10.0]]); b=np.array([[1.5,12.5]])',
      'call': 'pack(hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(0.01), '
              'copy.deepcopy(0.2), copy.deepcopy(0.5), copy.deepcopy(2.5), copy.deepcopy(1.0)))',
      'gold_call': 'pack(_oracle_hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), '
                   'copy.deepcopy(0.01), copy.deepcopy(0.2), copy.deepcopy(0.5), '
                   'copy.deepcopy(2.5), copy.deepcopy(1.0)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack(x):\n'
               ' d,c,t=x; return np.concatenate((d.ravel(),c.ravel(),np.array([t])))\n'
               'a=np.array([[1.0,10.0]]); b=np.array([[1.5001,12.5]])',
      'call': 'pack(hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(0.01), '
              'copy.deepcopy(0.2), copy.deepcopy(0.5), copy.deepcopy(2.5), copy.deepcopy(1.0)))',
      'gold_call': 'pack(_oracle_hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), '
                   'copy.deepcopy(0.01), copy.deepcopy(0.2), copy.deepcopy(0.5), '
                   'copy.deepcopy(2.5), copy.deepcopy(1.0)))'},
     {'setup': 'import copy\n'
               'import numpy as np\n'
               'def pack_rect(x):\n'
               ' d,c,t=x; return '
               'np.concatenate((np.asarray(d.shape,dtype=float),np.asarray(c.shape,dtype=float),d.ravel(),c.ravel(),np.array([t])))\n'
               'a=np.array([[1.00,10.0],[1.24,16.0]]); '
               'b=np.array([[1.02,10.4],[1.30,18.2],[0.85,8.0]])',
      'call': 'pack_rect(hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), '
              'copy.deepcopy(0.02), copy.deepcopy(0.4), copy.deepcopy(0.1), copy.deepcopy(2.0), '
              'copy.deepcopy(3.5)))',
      'gold_call': 'pack_rect(_oracle_hsqc_cost_matrices(copy.deepcopy(a), copy.deepcopy(b), '
                   'copy.deepcopy(0.02), copy.deepcopy(0.4), copy.deepcopy(0.1), '
                   'copy.deepcopy(2.0), copy.deepcopy(3.5)))'}]
