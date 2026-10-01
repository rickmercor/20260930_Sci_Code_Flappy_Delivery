"""
Compose the logarithmic (111)-versus-(001) CMOKE anisotropy advantage.

Step 07 must call the Step 01-06 public functions by their declared names. Require a non-Boolean integer n_angles that is at least 24 and divisible by 12; raise ValueError otherwise. For j=0,...,n_angles-1, define beta_j=2*pi*j/n_angles, alpha_j=beta_j+0.07*sin(5*beta_j+0.17)+0.03*cos(7*beta_j-0.31), and w_j=1+0.2*cos(3*beta_j+0.4)+0.1*sin(8*beta_j-0.2). Require finite warped angles and positive weights. For each cut, evaluate all eight directions, separate the channels, and take the weighted complex Fourier coefficient (2/sum(w))*sum_j w_j*channel_j*exp(-i*m*alpha_j), using m=4 for (001) and m=3 for (111). Define the cubic norm as the Euclidean norm across channels 0 and 3 and both polarizations; define the quadratic norm analogously across channels 1 and 2. The per-cut CMOKE/QMOKE ratio is cubic_norm/quadratic_norm. For each cut define fourier_scale=(2/sum(w))*norm(sum(w[:,None,None]*abs(complex_channels),axis=0)) and zero_tolerance=512*numpy.finfo(float).eps*fourier_scale. Treat a norm at or below that tolerance as numerically zero. Require both quadratic norms to be nonzero; set either ratio to exactly 0.0 when its cubic norm is numerically zero. Each ratio is a nonnegative norm quotient and therefore lies in the real log1p domain; require the computed ratios to remain finite and raise ValueError otherwise. Return numpy.log1p(ratio_111)-numpy.log1p(ratio_001), rounded once with numpy.round(...,10). This logarithmic contrast retains sensitivity to the absolute scale of each normalized cut response while compressing the dynamic range. The warped weighted scan is a new computational diagnostic and is not a value reported in the source.

Returns
-------
advantage : float Log1p contrast of the (111) and (001) cubic-to-quadratic harmonic ratios.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_cmoke_crystal_advantage(epsilon_d: complex, k_linear: complex, g_s: complex, two_g44: complex, h123: complex, h125: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex, n_angles: int) -> float:
    '''Return the rounded crystal-cut anisotropy advantage.

    Parameters
    ----------
    epsilon_d : complex
        Finite nonzero scalar diagonal permittivity.
    k_linear, g_s, two_g44, h123, h125 : complex
        Finite scalar magneto-optic coefficients in the stated convention.
    a_s, b_s, a_p, b_p : complex
        Finite scalar optical weights for s and p polarization.
    n_angles : int
        Non-Boolean grid size, at least 24 and divisible by 12.

    Returns
    -------
    advantage : float
        Log1p contrast of the (111) and (001) cubic-to-quadratic harmonic ratios.

    Raises
    ------
    ValueError
        An input violates its contract, a quadratic norm is numerically zero,
        or a computed ratio is nonfinite. The background specifies the norm threshold.'''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_cmoke_crystal_advantage(epsilon_d: complex, k_linear: complex, g_s: complex, two_g44: complex, h123: complex, h125: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex, n_angles: int) -> float:
    if isinstance(n_angles, (bool, np.bool_)) or not isinstance(n_angles, (int, np.integer)):
        raise ValueError('n_angles must be an integer')
    n_angles = int(n_angles)
    if n_angles < 24 or n_angles % 12:
        raise ValueError('n_angles must be at least 24 and divisible by 12')
    beta = 2.0 * np.pi * np.arange(n_angles, dtype=float) / n_angles
    alpha = beta + 0.07 * np.sin(5.0 * beta + 0.17) + 0.03 * np.cos(7.0 * beta - 0.31)
    weights = 1.0 + 0.2 * np.cos(3.0 * beta + 0.4) + 0.1 * np.sin(8.0 * beta - 0.2)
    weight_sum = float(np.sum(weights))
    if not np.all(np.isfinite(alpha)) or not np.all(weights > 0.0) or weight_sum <= 0.0:
        raise ValueError('warped scan must have finite angles and positive weights')
    design = _oracle_build_eight_directional_design(alpha)
    ratios = {}
    for cut, harmonic in (('001', 4), ('111', 3)):
        scan_rows = []
        for row in design:
            angle = float(row[0, 0])
            magnetization = row[:, 1:]
            cubic = _oracle_evaluate_rotated_cmoke_cubic_response(magnetization, h123, h125, cut, angle)
            quadratic = _oracle_evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, cut, angle)
            orders = _oracle_pack_magnetooptic_orders(magnetization, k_linear, quadratic, cubic)
            scan_rows.append(_oracle_compute_third_order_kerr_angles(orders, epsilon_d, a_s, b_s, a_p, b_p))
        separated = _oracle_separate_eight_directional_channels(np.stack(scan_rows))
        complex_channels = separated[..., 0] + 1j * separated[..., 1]
        phase = np.exp(-1j * harmonic * alpha)
        coefficient = 2.0 / weight_sum * np.tensordot(weights * phase, complex_channels, axes=(0, 0))
        cubic_norm = float(np.linalg.norm(coefficient[[0, 3], :]))
        quadratic_norm = float(np.linalg.norm(coefficient[[1, 2], :]))
        absolute_fourier_scale = float(2.0 / weight_sum * np.linalg.norm(np.sum(weights[:, None, None] * np.abs(complex_channels), axis=0)))
        zero_tolerance = 512.0 * np.finfo(float).eps * absolute_fourier_scale
        if quadratic_norm <= zero_tolerance:
            raise ValueError('quadratic anisotropy norms must be nonzero')
        if cubic_norm <= zero_tolerance:
            cubic_norm = 0.0
        ratio = cubic_norm / quadratic_norm
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError('anisotropy ratios must be finite and nonnegative')
        ratios[cut] = ratio
    return float(np.round(np.log1p(ratios['111']) - np.log1p(ratios['001']), 10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    composition_setup = (
        'import numpy as np\n'
        "_composition_names = ['evaluate_rotated_cmoke_cubic_response', 'evaluate_rotated_qmoke_quadratic_response', 'pack_magnetooptic_orders', 'compute_third_order_kerr_angles', 'build_eight_directional_design', 'separate_eight_directional_channels']\n"
        'def _capture_composition(function, args):\n'
        '    namespace = function.__globals__\n'
        '    hits = {name: 0 for name in _composition_names}\n'
        '    depth = {name: 0 for name in _composition_names}\n'
        '    originals = {}\n'
        '    def bindings(step_name):\n'
        '        return [key for key, value in list(namespace.items())\n'
        "                if callable(value) and (key == step_name or key.endswith('_' + step_name))]\n"
        '    def first_argument(call_args, call_kwargs, parameter):\n'
        '        if call_args:\n'
        '            return call_args[0]\n'
        '        if parameter in call_kwargs:\n'
        '            return call_kwargs[parameter]\n'
        '        if call_kwargs:\n'
        '            return next(iter(call_kwargs.values()))\n'
        "        raise AssertionError('instrumented step received no arguments')\n"
        '    def fingerprinted(name, implementation):\n'
        '        def wrapped(*call_args, **call_kwargs):\n'
        '            hits[name] += 1\n'
        '            if depth[name]:\n'
        '                return implementation(*call_args, **call_kwargs)\n'
        '            depth[name] += 1\n'
        '            try:\n'
        '                result = np.array(implementation(*call_args, **call_kwargs), copy=True)\n'
        '            finally:\n'
        '                depth[name] -= 1\n'
        "            if name == 'evaluate_rotated_cmoke_cubic_response':\n"
        '                result *= 1.031\n'
        "            elif name == 'evaluate_rotated_qmoke_quadratic_response':\n"
        '                result *= 0.971\n'
        "            elif name == 'pack_magnetooptic_orders':\n"
        '                result *= np.array([1.007, 0.983, 1.029], dtype=float)[None, :, None, None]\n'
        "            elif name == 'compute_third_order_kerr_angles':\n"
        "                orders = np.asarray(first_argument(call_args, call_kwargs, 'order_packets'), dtype=float)\n"
        '                energy = np.sqrt(np.sum(orders * orders, axis=(1, 2, 3)))\n'
        '                result[..., 0] *= (1.0 + 0.021 * np.tanh(energy))[:, None]\n'
        '                result[..., 1] *= (1.0 - 0.017 * np.tanh(energy))[:, None]\n'
        "            elif name == 'build_eight_directional_design':\n"
        '                result[..., 1] *= 1.013\n'
        '                result[..., 2] *= 0.991\n'
        "            elif name == 'separate_eight_directional_channels':\n"
        "                scan = np.asarray(first_argument(call_args, call_kwargs, 'kerr_scan'), dtype=float)\n"
        '                weights = np.arange(1, 33, dtype=float).reshape(1, 8, 2, 2)\n'
        '                factor = 1.0 + 0.125 * np.tanh(np.sum(np.abs(scan) * weights, axis=(1, 2, 3)))\n'
        '                result[:, [0, 3], :, :] *= factor[:, None, None, None]\n'
        '            return result\n'
        '        return wrapped\n'
        '    try:\n'
        '        for name in _composition_names:\n'
        '            for key in bindings(name):\n'
        '                originals[key] = namespace[key]\n'
        '                namespace[key] = fingerprinted(name, originals[key])\n'
        '        value = function(*args)\n'
        '    finally:\n'
        '        for key, implementation in originals.items():\n'
        '            namespace[key] = implementation\n'
        '    return np.array([float(value)] + [float(hits[name] > 0) for name in _composition_names])\n'
    )
    asymmetric_setup = (
        'import numpy as np\n'
        'def _run_asymmetric_zero_case(function, args):\n'
        '    namespace = function.__globals__\n'
        "    depth = {'cubic': 0, 'separator': 0}\n"
        '    originals = {}\n'
        '    def bindings(step_name):\n'
        '        return [key for key, value in list(namespace.items())\n'
        "                if callable(value) and (key == step_name or key.endswith('_' + step_name))]\n"
        '    def zero_001_cubic(implementation):\n'
        '        def wrapped(*call_args, **call_kwargs):\n'
        "            if depth['cubic']:\n"
        '                return implementation(*call_args, **call_kwargs)\n'
        "            depth['cubic'] += 1\n"
        '            try:\n'
        '                result = implementation(*call_args, **call_kwargs)\n'
        '            finally:\n'
        "                depth['cubic'] -= 1\n"
        "            cut = call_args[3] if len(call_args) > 3 else call_kwargs['crystal_cut']\n"
        "            return np.zeros_like(result) if cut == '001' else result\n"
        '        return wrapped\n'
        '    def zero_roundoff_odd_channels(implementation):\n'
        '        def wrapped(*call_args, **call_kwargs):\n'
        "            if depth['separator']:\n"
        '                return implementation(*call_args, **call_kwargs)\n'
        "            depth['separator'] += 1\n"
        '            try:\n'
        '                result = np.array(implementation(*call_args, **call_kwargs), copy=True)\n'
        '            finally:\n'
        "                depth['separator'] -= 1\n"
        '            if np.linalg.norm(result[:, [0, 3], :, :]) < 1e-12:\n'
        '                result[:, [0, 3], :, :] = 0.0\n'
        '            return result\n'
        '        return wrapped\n'
        '    try:\n'
        "        for key in bindings('evaluate_rotated_cmoke_cubic_response'):\n"
        '            originals[key] = namespace[key]\n'
        '            namespace[key] = zero_001_cubic(originals[key])\n'
        "        for key in bindings('separate_eight_directional_channels'):\n"
        '            originals[key] = namespace[key]\n'
        '            namespace[key] = zero_roundoff_odd_channels(originals[key])\n'
        '        return function(*args)\n'
        '    finally:\n'
        '        for key, implementation in originals.items():\n'
        '            namespace[key] = implementation\n'
    )
    return [
        {
            "setup": '# Case: normal\n# Coverage: normal (all six predecessor dependencies instrumented)\n' + composition_setup,
            "call": '_capture_composition(compute_cmoke_crystal_advantage, (-9.6+14.2j,.034+.021j,.018-.011j,-.007+.016j,.0042+.0028j,-.0011+.0007j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,48))',
            "gold_call": '_capture_composition(_oracle_compute_cmoke_crystal_advantage, (-9.6+14.2j,.034+.021j,.018-.011j,-.007+.016j,.0042+.0028j,-.0011+.0007j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,48))',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (minimum n_angles=24, predecessor dependencies instrumented)\n' + composition_setup,
            "call": '_capture_composition(compute_cmoke_crystal_advantage, (-12.1+8.7j,-.019+.026j,.013+.019j,-.011+.004j,.0031-.0044j,-.0008+.0012j,.81-.07j,.08-.12j,-.47+.22j,-.10+.15j,24))',
            "gold_call": '_capture_composition(_oracle_compute_cmoke_crystal_advantage, (-12.1+8.7j,-.019+.026j,.013+.019j,-.011+.004j,.0031-.0044j,-.0008+.0012j,.81-.07j,.08-.12j,-.47+.22j,-.10+.15j,24))',
        },
        {
            "setup": '# Case: edge\n# Coverage: edge (small magneto-optic coefficients)\n',
            "call": 'compute_cmoke_crystal_advantage(-7.2+11.5j,.021e-8-.018e-8j,.027e-8+.006e-8j,.003e-8-.014e-8j,-.005e-8+.003e-8j,.0014e-8+.0009e-8j,.64+.11j,-.09+.04j,-.52-.13j,.14+.08j,36)',
            "gold_call": '_oracle_compute_cmoke_crystal_advantage(-7.2+11.5j,.021e-8-.018e-8j,.027e-8+.006e-8j,.003e-8-.014e-8j,-.005e-8+.003e-8j,.0014e-8+.0009e-8j,.64+.11j,-.09+.04j,-.52-.13j,.14+.08j,36)',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (numerically zero (001) cubic numerator only)\n' + asymmetric_setup,
            "call": '_run_asymmetric_zero_case(compute_cmoke_crystal_advantage, (-9.6+14.2j,0j,.018-.011j,-.007+.016j,.0042+.0028j,-.0011+.0007j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,24))',
            "gold_call": '_run_asymmetric_zero_case(_oracle_compute_cmoke_crystal_advantage, (-9.6+14.2j,0j,.018-.011j,-.007+.016j,.0042+.0028j,-.0011+.0007j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,24))',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (exactly zero cubic numerators for both cuts)\n',
            "call": 'compute_cmoke_crystal_advantage(-9.6+14.2j,0j,.018-.011j,-.007+.016j,0j,0j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,24)',
            "gold_call": '_oracle_compute_cmoke_crystal_advantage(-9.6+14.2j,0j,.018-.011j,-.007+.016j,0j,0j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,24)',
        },
        {
            "setup": '# Case: edge\n# Coverage: edge (sub-tolerance cubic numerators for both cuts)\n',
            "call": 'compute_cmoke_crystal_advantage(-9.6+14.2j,0j,.018-.011j,-.007+.016j,1e-300+0j,0j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,48)',
            "gold_call": '_oracle_compute_cmoke_crystal_advantage(-9.6+14.2j,0j,.018-.011j,-.007+.016j,1e-300+0j,0j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j,48)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, True)',
            "gold_call": '_value_error_code(_oracle_compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, True)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 23)',
            "gold_call": '_value_error_code(_oracle_compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 23)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 25)',
            "gold_call": '_value_error_code(_oracle_compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 25)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(compute_cmoke_crystal_advantage, np.nan + 0j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 24)',
            "gold_call": '_value_error_code(_oracle_compute_cmoke_crystal_advantage, np.nan + 0j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0.73 - 0.18j, 0.11 + 0.06j, -0.61 + 0.27j, 0.16 - 0.09j, 24)',
        },
        {
            "setup": '# Case: edge\n\n' + exception_setup,
            "call": '_value_error_code(compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0j, 0.11 + 0.06j, 0j, 0.16 - 0.09j, 24)',
            "gold_call": '_value_error_code(_oracle_compute_cmoke_crystal_advantage, -9.6 + 14.2j, 0.034 + 0.021j, 0.018 - 0.011j, -0.007 + 0.016j, 0.0042 + 0.0028j, -0.0011 + 0.0007j, 0j, 0.11 + 0.06j, 0j, 0.16 - 0.09j, 24)',
        },
    ]
