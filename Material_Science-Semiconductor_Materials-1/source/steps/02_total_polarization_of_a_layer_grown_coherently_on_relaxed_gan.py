"""
Total polarization of a group-III nitride layer grown coherently on a thick relaxed GaN buffer.

The layer is forced to the in-plane lattice constant of the buffer while its own free-standing constant
follows the composition, so its in-plane strain is (a_GaN - a(x)) / a(x), measured against the layer's own
lattice constant a(x). Its polarization along the c axis has two parts: the spontaneous polarization the
material carries unstrained, and the part the strain induces. The layer has a free top surface, so it
carries no stress along the c axis. Every alloy constant is linear in Al fraction between the binary
endpoints. The buffer itself is the relaxed binary, so passing it in at zero Al fraction returns its
spontaneous value alone.

Inputs: al_fraction: float in [0, 1] gan, aln: dicts keyed a (a-axis lattice constant, m), psp (spontaneous polarization, C/m^2, negative),
  e31 and e33 (piezoelectric constants, C/m^2), c13 and c33 (elastic constants, Pa), eps_r

 Returns: float, magnitude of the total polarization of the layer in C/m^2

 Raises: ValueError if al_fraction is outside [0, 1], a required key is missing, or a lattice constant or c33 is not positive.

Returns
-------
float, magnitude of the total polarization of the layer in C/m^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pseudomorphic_layer_polarization(al_fraction: float, gan: dict, aln: dict) -> float:
    """Al mole fraction and the GaN and AlN parameter dictionaries -> magnitude of the total
    polarization of the coherently strained layer in C/m^2.
 
    Raises ValueError if al_fraction is outside [0, 1], a key is missing, or a lattice constant or c33
    is not positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _mix(gan: dict, aln: dict, key: str, x: float) -> float:
    return x * aln[key] + (1.0 - x) * gan[key]

def _oracle_pseudomorphic_layer_polarization(al_fraction: float, gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (0.0 <= al_fraction <= 1.0):
        raise ValueError('Al mole fraction must lie in [0, 1]')
    for name, d in (('gan', gan), ('aln', aln)):
        for key in ('a', 'psp', 'e31', 'e33', 'c13', 'c33', 'eps_r'):
            if key not in d:
                raise ValueError('%s is missing the key %s' % (name, key))
        if not (d['a'] > 0.0 and d['c33'] > 0.0):
            raise ValueError('%s needs a positive lattice constant and c33' % name)
    a_layer = _mix(gan, aln, 'a', al_fraction)
    strain = (gan['a'] - a_layer) / a_layer
    e31 = _mix(gan, aln, 'e31', al_fraction)
    e33 = _mix(gan, aln, 'e33', al_fraction)
    c13 = _mix(gan, aln, 'c13', al_fraction)
    c33 = _mix(gan, aln, 'c33', al_fraction)
    p_pz = 2.0 * strain * (e31 - e33 * c13 / c33)
    p_sp = _mix(gan, aln, 'psp', al_fraction)
    return abs(p_sp) + abs(p_pz)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n",
         "call": "round(pseudomorphic_layer_polarization(0.31, GAN, ALN), 12)",
         "gold_call": "round(_oracle_pseudomorphic_layer_polarization(0.31, GAN, ALN), 12)"},
        {"setup": "import numpy as np\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n",
         "call": "round(pseudomorphic_layer_polarization(0.0, GAN, ALN), 12)",
         "gold_call": "round(_oracle_pseudomorphic_layer_polarization(0.0, GAN, ALN), 12)"},
        {"setup": "import numpy as np\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n",
         "call": "round(pseudomorphic_layer_polarization(1.0, GAN, ALN), 12)",
         "gold_call": "round(_oracle_pseudomorphic_layer_polarization(1.0, GAN, ALN), 12)"},
        {"setup": "import numpy as np\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\ndef run_model():\n    try:\n        pseudomorphic_layer_polarization(1.2, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_pseudomorphic_layer_polarization(1.2, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
