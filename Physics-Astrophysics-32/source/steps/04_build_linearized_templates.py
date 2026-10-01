"""
Build the mixed-mode parent design used by the analytic amplitude posterior.

For each trial state divide the intrinsic frequencies by the paired remnant mass ratio and form $x_{22}=\mu_{22}e^{-i\omega_{220}t}$ and $x_{32}=\mu_{32}e^{-i\omega_{320}t}$. A complex column $x$ maps $(\Re A,\Im A)$ to real data columns $(\Re x,\Im x)$ and $(-\Im x,\Re x)$ in `(Re h[0:7], Im h[0:7])` order. Carry the nonlinear response as metadata for step 07 without linearizing it.

Returns
-------
`numpy.ndarray` of shape `(n,18,4)`. Rows `0:14` are the real parent design. Row 14 is `(spin,mass,Re omega220,Im omega220)`, row 15 is `(Re omega320,Im omega320,Re omega640,Im omega640)`, row 16 is `(Re R64^22,Im R64^22,Re R64^23,Im R64^23)`, and row 17 holds both complex projections. A request using only the 220 parent leaves design columns `2:4` at exact zero. Every component is compared at `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_linearized_templates(
    data: np.ndarray,
    table: np.ndarray,
    quadratic_response: np.ndarray,
    include_mixed_parent: bool = True,
) -> np.ndarray:
    """Build the real parent design and pack the child-response metadata.

    Parameters
    ----------
    data : numpy.ndarray, shape (7, 9)
        Finite ringdown record with strictly increasing sample times.
    table : numpy.ndarray, shape (n, 16)
        Interpolated state table in the exact step-02 column order.
    quadratic_response : numpy.ndarray, shape (n, 4)
        Real and imaginary parts of the intrinsic spherical-64 responses
        `(R64^22, R64^23)` from step 03.
    include_mixed_parent : bool, default=True
        Exact boolean. False leaves the 320 design columns at exact zero;
        true activates them.

    Returns
    -------
    numpy.ndarray, shape (n, 18, 4)
        Rows `0:14` contain the real parent design. Rows `14:18` contain,
        respectively, `(spin,mass,Re omega220,Im omega220)`,
        `(Re omega320,Im omega320,Re omega640,Im omega640)`, both complex
        responses, and both complex spherical projections. Components are
        compared at tolerance `1e-9`.

    Raises
    ------
    ValueError
        If the flag is not an exact boolean; if an array has an incompatible
        shape or nonfinite entry; if times are not strictly increasing; if a
        mass is not positive; or if a physical frequency lacks positive real
        and negative imaginary parts.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""Build the mixed-mode parent design used by the analytic amplitude posterior.

For each trial state divide the intrinsic frequencies by the paired remnant-mass
ratio and form the spherical-22 columns

$$x_{22}=\mu_{22}e^{-i\omega_{220}t},\qquad
x_{32}=\mu_{32}e^{-i\omega_{320}t}.$$

A complex column ``x`` maps amplitudes ``(Re A, Im A)`` to the real data order
``(Re h[0:7], Im h[0:7])`` through columns ``(Re x, Im x)`` and
``(-Im x, Re x)``. The nonlinear response is carried as metadata for the later
quadratic-moment calculation; it is not linearized here.

Returns
-------
`numpy.ndarray` of shape `(n, 18, 4)`. Rows `0:14` are the real parent design.
Row 14 is `(spin, mass_ratio, Re omega220, Im omega220)`, row 15 is
`(Re omega320, Im omega320, Re omega640, Im omega640)`, row 16 is
`(Re R64^22, Im R64^22, Re R64^23, Im R64^23)`, and row 17 is
`(Re mu22, Im mu22, Re mu32, Im mu32)`. With a 220-only parent, design
columns `2:4` are exact zero while all physical metadata remain populated.
Every returned component is compared at tolerance `1e-9`.
"""
import numpy as np
import numpy as np
def _template_real_design(columns: list[np.ndarray]) -> np.ndarray:
    packed = []
    for column in columns:
        packed.extend((
            np.concatenate((column.real, column.imag)),
            np.concatenate((-column.imag, column.real)),
        ))
    return np.column_stack(packed)
def _oracle_build_linearized_templates(
    data: np.ndarray,
    table: np.ndarray,
    quadratic_response: np.ndarray,
    include_mixed_parent: bool = True,
) -> np.ndarray:
    record = np.asarray(data, dtype=float)
    states = np.asarray(table, dtype=float)
    responses = np.asarray(quadratic_response, dtype=float)
    if record.shape != (7, 9) or not np.all(np.isfinite(record)):
        raise ValueError("data must have finite shape (7, 9)")
    if np.any(np.diff(record[:, 0]) <= 0.0):
        raise ValueError("sample times must be strictly increasing")
    if states.ndim != 2 or states.shape[1] != 16 or states.shape[0] == 0 or not np.all(np.isfinite(states)):
        raise ValueError("table must have nonempty finite shape (n, 16)")
    if responses.shape != (states.shape[0], 4) or not np.all(np.isfinite(responses)):
        raise ValueError("quadratic_response must have finite shape (n, 4)")
    if type(include_mixed_parent) is not bool:
        raise ValueError("include_mixed_parent must be an exact boolean")
    if np.any(states[:, 1] <= 0.0):
        raise ValueError("remnant-mass ratios must be positive")
    frequency_pairs = np.stack((states[:, 2:8:2], states[:, 3:8:2]), axis=2)
    if np.any(frequency_pairs[:, :, 0] <= 0.0) or np.any(frequency_pairs[:, :, 1] >= 0.0):
        raise ValueError("frequencies require positive real and negative imaginary parts")

    time = record[:, 0]
    output = np.zeros((states.shape[0], 18, 4), dtype=float)
    for index, state in enumerate(states):
        spin, mass = state[:2]
        omega220 = (state[2] + 1j * state[3]) / mass
        omega320 = (state[4] + 1j * state[5]) / mass
        omega640 = (state[6] + 1j * state[7]) / mass
        mu22 = state[8] + 1j * state[9]
        mu32 = state[10] + 1j * state[11]
        parent220 = mu22 * np.exp(-1j * omega220 * time)
        parent320 = mu32 * np.exp(-1j * omega320 * time)
        design = _template_real_design([parent220, parent320])
        if not include_mixed_parent:
            design[:, 2:4] = 0.0
        output[index, :14] = design
        output[index, 14] = (spin, mass, omega220.real, omega220.imag)
        output[index, 15] = (
            omega320.real, omega320.imag, omega640.real, omega640.imag
        )
        output[index, 16] = responses[index]
        output[index, 17] = (mu22.real, mu22.imag, mu32.real, mu32.imag)
    if not np.all(np.isfinite(output)):
        raise ValueError("template construction produced nonfinite output")
    return output

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables, fixture_synthesize_quadratic_response):\n    import numpy as np\n    data = fixture_load_ringdown_data(0.87, 1.08)\n    tab = fixture_interpolate_spin_tables(np.array([0.73, 0.781, 0.84]), np.array([0.97, 1.0, 1.03]))\n    rsp = fixture_synthesize_quadratic_response(tab)\n    return fn_under_test(data, tab, rsp, True)',
            'call': 'run_case(build_linearized_templates, load_ringdown_data, interpolate_spin_tables, synthesize_quadratic_response)',
            'gold_call': 'run_case(_oracle_build_linearized_templates, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables, _oracle_synthesize_quadratic_response)',
        },
        {
            'setup': 'def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables):\n    import numpy as np\n    data = fixture_load_ringdown_data(1.12, 0.94)\n    tab = fixture_interpolate_spin_tables(np.array([0.7, 0.86]))\n    rsp = np.array([[0.004, -0.002, 0.001, 0.003], [-0.003, 0.005, -0.002, 0.004]])\n\n    def packed(fn):\n        out = fn()\n        return np.concatenate(([int(np.all(out[:, :14, 2:4] == 0.0)), int(np.any(out[:, 16] != 0.0))], out.ravel()))\n    return packed(lambda: fn_under_test(data, tab, rsp, False))',
            'call': 'run_case(build_linearized_templates, load_ringdown_data, interpolate_spin_tables)',
            'gold_call': 'run_case(_oracle_build_linearized_templates, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables)',
        },
        {
            'setup': 'def run_case(fn_under_test, fixture_load_ringdown_data, fixture_interpolate_spin_tables):\n    import numpy as np\n\n    def catches(fn):\n        try:\n            fn()\n        except ValueError:\n            return 1\n        except Exception:\n            return 2\n        return 0\n    data = fixture_load_ringdown_data()\n    tab = fixture_interpolate_spin_tables(np.array([0.76]))\n    rsp = np.zeros((1, 4))\n    bad_time = data.copy()\n    bad_time[2, 0] = bad_time[1, 0]\n    bad_finite = data.copy()\n    bad_finite[0, 1] = np.nan\n    bad_mass = tab.copy()\n    bad_mass[0, 1] = 0.0\n    bad_frequency = tab.copy()\n    bad_frequency[0, 3] = 0.0\n    return np.array([catches(lambda: fn_under_test(bad_time, tab, rsp)), catches(lambda: fn_under_test(bad_finite, tab, rsp)), catches(lambda: fn_under_test(data, bad_mass, rsp)), catches(lambda: fn_under_test(data, bad_frequency, rsp)), catches(lambda: fn_under_test(data, tab, np.zeros((1, 5)))), catches(lambda: fn_under_test(data, tab, rsp, 1))])',
            'call': 'run_case(build_linearized_templates, load_ringdown_data, interpolate_spin_tables)',
            'gold_call': 'run_case(_oracle_build_linearized_templates, _oracle_load_ringdown_data, _oracle_interpolate_spin_tables)',
        },
    ]
