"""
Compose the source-derived TE/TM layer-order contrast diagnostic.

The scalar compares the equal incoherent TE/TM averages of transmitted nonspecular power from the two possible layer orderings. Constitutive profiles and their thickness fractions move together; there is no transverse reflection, phase translation, or conjugation. Compose the earlier public functions, reaching reciprocal_fourier through bergmann_generator, and exclude the specular channel from each sum.

Returns
-------
return contrast: finite Python float (Q_forward-Q_reverse)/(Q_forward+Q_reverse), or exactly 0.0 if the denominator is zero; Q is the equal incoherent average of transmitted powers for j>=1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def layer_reversal_contrast(
    materials,
    fractions,
    incident_sine,
    reciprocal_ratio,
    thickness,
    order_count,
):
    """Return the scalar contrast of nonspecular transmitted efficiencies.

    materials is a finite numeric array with shape (2,2,H), H>=1. Axis 0
    lists the two layers in forward order; axis 1 is epsilon then mu;
    axis 2 lists coefficients of exp(1j*h*K*y), h=0,...,H-1. Each profile
    has a constant coefficient strictly dominating its harmonic tail.
    fractions has shape (2,), is real and nonnegative, and sums to one
    within 1e-12. Reversing the stack reverses both layers and their
    associated fractions, without conjugating or translating harmonics.
    Use grating_channels to construct orders 0,...,order_count-1. All open
    nonnegative orders must be included: the next sine must exceed one.
    Extra evanescent orders are allowed; grazing is excluded as in Step 1.
    For each stack order use bergmann_generator (which uses
    reciprocal_fourier), ordered_transfer_series, global_transfer_series,
    scattering_series, and diffraction_efficiencies. For TE set
    (alpha,beta)=(mu,epsilon); for TM set (epsilon,mu). For each ordering
    let Q be the equal incoherent average of TE and TM transmitted power
    in j>=1. Return (Q_forward-Q_reverse)/(Q_forward+Q_reverse), or 0.0
    when the denominator is exactly zero. thickness and channel parameters
    satisfy the earlier contracts. Raise ValueError for invalid inputs,
    preserve input arrays, and return a finite Python float.

    Raise ValueError if the computation produces nonfinite coefficients
    or efficiencies; do not return nonfinite results.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_layer_reversal_contrast(
    materials,
    fractions,
    incident_sine,
    reciprocal_ratio,
    thickness,
    order_count,
):
    import numpy as np

    materials = np.asarray(materials)
    fractions = np.asarray(fractions)
    if (
        materials.ndim != 3
        or materials.shape[:2] != (2, 2)
        or materials.shape[2] == 0
        or materials.dtype.kind not in "iufc"
        or not np.all(np.isfinite(materials))
    ):
        raise ValueError("materials must have finite numeric shape (2,2,H).")
    if fractions.shape != (2,):
        raise ValueError(
            "Expected one thickness fraction for each of two layers."
        )

    sines, cosines = _oracle_grating_channels(
        incident_sine, reciprocal_ratio, order_count
    )
    if sines[-1] + reciprocal_ratio <= 1:
        raise ValueError("All open nonnegative orders must be retained.")

    nonspecular = []
    for ordering in (np.array([0, 1]), np.array([1, 0])):
        average_power = 0.0
        for alpha_index, beta_index in ((1, 0), (0, 1)):
            generators = np.array([
                _oracle_bergmann_generator(
                    sines,
                    materials[layer, alpha_index],
                    materials[layer, beta_index],
                )
                for layer in ordering
            ])
            state = _oracle_ordered_transfer_series(
                generators, fractions[ordering]
            )
            transfer = _oracle_global_transfer_series(state, cosines)
            amplitudes = _oracle_scattering_series(transfer)
            powers = _oracle_diffraction_efficiencies(
                amplitudes, cosines, thickness
            )
            average_power += 0.5 * float(np.sum(powers[1, 1:]))
        nonspecular.append(average_power)

    denominator = nonspecular[0] + nonspecular[1]
    if denominator == 0:
        return 0.0
    return float((nonspecular[0] - nonspecular[1]) / denominator)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [{'setup': '\n'
               'materials = np.array(\n'
               '    [\n'
               '        [\n'
               '            [2.4 + 0.45j, 0.35, 0.04j],\n'
               '            [1.2 + 0.14j, 0.10 * np.exp(0.3j), 0.015],\n'
               '        ],\n'
               '        [\n'
               '            [3.1 + 0.38j, 0.30 * np.exp(0.8j), -0.025j],\n'
               '            [1.6 + 0.15j, 0.12 * np.exp(-0.55j), 0.012j],\n'
               '        ],\n'
               '    ]\n'
               ')\n',
      'call': 'layer_reversal_contrast(materials, [0.37, 0.63], -0.6, 0.39, 0.08, 5)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.37, 0.63], -0.6, 0.39, 0.08, 5)'},
     {'setup': '\n'
               'materials = np.array(\n'
               '    [\n'
               '        [\n'
               '            [3.1 + 0.38j, 0.30 * np.exp(0.8j), -0.025j],\n'
               '            [1.6 + 0.15j, 0.12 * np.exp(-0.55j), 0.012j],\n'
               '        ],\n'
               '        [\n'
               '            [2.4 + 0.45j, 0.35, 0.04j],\n'
               '            [1.2 + 0.14j, 0.10 * np.exp(0.3j), 0.015],\n'
               '        ],\n'
               '    ]\n'
               ')\n',
      'call': 'layer_reversal_contrast(materials, [0.63, 0.37], -0.6, 0.39, 0.08, 5)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.63, 0.37], -0.6, 0.39, 0.08, 5)'},
     {'setup': '\nmaterials = np.array([[[2.0], [1.0]], [[1.4], [1.2]]])\n',
      'call': 'layer_reversal_contrast(materials, [0.2, 0.8], -0.4, 0.6, 0.1, 3)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.2, 0.8], -0.4, 0.6, 0.1, 3)'},
     {'setup': '\n'
               'materials = np.tile(\n'
               '    np.array([[[2.0 + 0.2j, 0.1], [1.1 + 0.1j, 0.05j]]]),\n'
               '    (2, 1, 1),\n'
               ')\n',
      'call': 'layer_reversal_contrast(materials, [0.35, 0.65], -0.5, 0.6, 0.07, 3)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.35, 0.65], -0.5, 0.6, 0.07, 3)'},
     {'setup': '\n'
               'materials = np.array(\n'
               '    [\n'
               '        [[2.0 + 0.2j, 0.1], [1.0, 0]],\n'
               '        [[1.4 + 0.1j, 0.05j], [1.2, 0]],\n'
               '    ]\n'
               ')\n',
      'call': 'layer_reversal_contrast(materials, [0.2, 0.8], -0.4, 0.6, 0.0, 3)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.2, 0.8], -0.4, 0.6, 0.0, 3)'},
     {'setup': '\n'
               'materials = np.array(\n'
               '    [\n'
               '        [\n'
               '            [2.4 + 0.45j, 0.35, 0.04j],\n'
               '            [1.2 + 0.14j, 0.10 * np.exp(0.3j), 0.015],\n'
               '        ],\n'
               '        [\n'
               '            [3.1 + 0.38j, 0.30 * np.exp(0.8j), -0.025j],\n'
               '            [1.6 + 0.15j, 0.12 * np.exp(-0.55j), 0.012j],\n'
               '        ],\n'
               '    ]\n'
               ')\n',
      'call': 'layer_reversal_contrast(materials, [0.37, 0.63], -0.6, 0.39, 0.08, 8)',
      'gold_call': '_oracle_layer_reversal_contrast(materials, [0.37, 0.63], -0.6, 0.39, 0.08, 8)'},
     {'setup': 'materials = np.array([[[2.4 + 0.45j, 0.35, 0.04j], [1.2 + 0.14j, 0.1 * np.exp(0.3j), '
               '0.015]], [[3.1 + 0.38j, 0.3 * np.exp(0.8j), -0.025j], [1.6 + 0.15j, 0.12 * '
               'np.exp(-0.55j), 0.012j]]])\n'
               '\n'
               'def rejects_nonfinite(fn):\n'
               '    try:\n'
               '        fn(materials, [0.37, 0.63], -0.6, 0.39, 1e+200, 5)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': 'rejects_nonfinite(layer_reversal_contrast)',
      'gold_call': 'rejects_nonfinite(_oracle_layer_reversal_contrast)'}]
