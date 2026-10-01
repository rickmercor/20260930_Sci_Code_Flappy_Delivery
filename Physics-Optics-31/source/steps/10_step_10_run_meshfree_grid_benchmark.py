"""
Compose every public scientific step and return the endpoint amplitude error.

A compact nondimensional run preserves the paper's comparison while replacing production convergence sizes with deterministic testable ones.

Returns
-------
float: deterministic finite nonnegative result from all nine earlier steps.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def run_meshfree_grid_benchmark(
    n_nodes: int = 32,
    nx: int = 48,
    ny: int = 40,
    dt: float = .02,
    n_steps: int = 40,
    field_amplitude: float = .12,
    gamma_scale: float = 4.0,
    seed: int = 23,
) -> float:
    """Run the prescribed two-dimensional quasi-static benchmark.

    Defaults use hbar=mass=charge=1, q0=(-2.4,.65), p0=(4,0), gamma0=diag(.8,4),
    R=.4, dielectric=-24.061+1.5068j, omega=1.3, x=[-5,3], and y=[-2,2].
    Include both endpoints: x_i=-5+8*i/(nx-1), i=0,...,nx-1, and
    y_j=-2+4*j/(ny-1), j=0,...,ny-1. Grid counts ``nx`` and ``ny`` are
    integers at least 8; ``n_steps`` is a nonnegative integer; ``n_nodes``
    is a positive power of two, including one. ``dt`` and ``gamma_scale``
    are positive finite reals, and ``field_amplitude`` is a finite real.
    ``seed`` is the integer Sobol seed.

    Returns
    -------
    float
        Unit-normalized momentum-amplitude shape discrepancy.

    Raises
    ------
    ValueError
        If grid or step counts are invalid, ``n_nodes`` violates the power-of-two
        contract, or ``dt``, ``gamma_scale``, or ``field_amplitude`` violates
        its stated finite-scalar contract.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral, Real
import numpy as np

def _oracle_run_meshfree_grid_benchmark(
    n_nodes: int = 32,
    nx: int = 48,
    ny: int = 40,
    dt: float = .02,
    n_steps: int = 40,
    field_amplitude: float = .12,
    gamma_scale: float = 4.0,
    seed: int = 23,
) -> float:
    """Reference orchestrator calling only prior oracle functions."""
    for value, name, lower in ((nx,"nx",8),(ny,"ny",8),(n_steps,"n_steps",0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < lower:
            raise ValueError(f"{name} has an invalid integer value")
    for value, name in ((dt,"dt"),(gamma_scale,"gamma_scale"),(field_amplitude,"field_amplitude")):
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if dt <= 0 or gamma_scale <= 0:
        raise ValueError("dt and gamma_scale must be positive")
    q0 = np.array([-2.4, .65]); p0 = np.array([4.0, 0.0])
    gamma0 = np.diag([.8, 4.0]); gamma = float(gamma_scale) * gamma0
    x = np.linspace(-5.0, 3.0, int(nx), endpoint=True)
    y = np.linspace(-2.0, 2.0, int(ny), endpoint=True)
    probe = _oracle_evaluate_quasistatic_dipole(0.0, np.array([[.4, 0.0]]), field_amplitude, .4, -24.061+1.5068j, 1.3)[0][0]
    contrast = abs((-24.061+1.5068j - 1) / (-24.061+1.5068j + 1))
    calibrated_field = -probe / (.4 * contrast)
    nodes = _oracle_generate_weighted_sobol_nodes(q0, p0, gamma0, gamma, n_nodes, 1.0, seed)
    coeffs = _oracle_compute_gaussian_expansion_coefficients(nodes, q0, p0, gamma0, gamma, 1.0)
    q, p, Q, P, S = _oracle_initialize_hagedorn_state(nodes, gamma, 1.0)
    q, p, Q, P, S = _oracle_propagate_hagedorn_state(q, p, Q, P, S, dt, n_steps, 1.0, calibrated_field, .4, -24.061+1.5068j, 1.3, 1.0)
    mesh = _oracle_reconstruct_tgwp_momentum(x, y, coeffs, q, p, Q, P, S, 1.0)
    initial = _oracle_initialize_grid_wavefunction(x, y, q0, p0, gamma0, 1.0)
    reference = _oracle_propagate_split_step_reference(initial, x, y, dt, n_steps, 1.0, calibrated_field, .4, -24.061+1.5068j, 1.3, 1.0, 1.0)
    return _oracle_compute_momentum_amplitude_error(mesh, reference, x[1]-x[0], y[1]-y[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return complete differential tests for normal and boundary inputs."""
    return [{'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc',
      'call': 'float(run_meshfree_grid_benchmark(8,36,30,.025,28,.09,3.5,11))',
      'gold_call': 'float(_oracle_run_meshfree_grid_benchmark(8,36,30,.025,28,.09,3.5,11))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc',
      'call': 'float(run_meshfree_grid_benchmark(16,32,28,.025,0,.05,3.,7))',
      'gold_call': 'float(_oracle_run_meshfree_grid_benchmark(16,32,28,.025,0,.05,3.,7))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc',
      'call': 'float(run_meshfree_grid_benchmark(64,40,32,.015,30,0.,5.,41))',
      'gold_call': 'float(_oracle_run_meshfree_grid_benchmark(64,40,32,.015,30,0.,5.,41))'},
     {'setup': 'import numpy as np\n'
               'from numbers import Integral, Real\n'
               'from scipy.stats import norm, qmc',
      'call': 'float(run_meshfree_grid_benchmark(1,16,12,.02,0,0.,4.,23))',
      'gold_call': 'float(_oracle_run_meshfree_grid_benchmark(1,16,12,.02,0,0.,4.,23))'}]
