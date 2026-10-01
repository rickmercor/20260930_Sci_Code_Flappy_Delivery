"""
Propagate the intracavity field one physical round trip at a time with the paper's delayed recursive phase and a zero stored field before the first sample.

Apply the retarded complex-amplitude convention from the cited cavity-dynamics source as specialized to one physical round trip by the main paper. Ra and Rb are intensity reflectivities; initialize the stored field to zero and preserve the drive order and relative-path phase sign.

Returns
-------
Return one one-dimensional complex NumPy array with one field value per drive row.
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def propagate_cavity_field(drive, wavelength_m, reflectivity_a, reflectivity_b):
    """Return the complex intracavity field on the round-trip grid."""
    return np.empty(0, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_cavity_field(drive, wavelength_m, reflectivity_a, reflectivity_b):
    """Round-trip recurrence E_j=t_a E_in,j+q exp(-4 pi i delta_d/lambda) E_{j-1}."""
    import math
    import numpy as np
    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 2 or drive.shape[1] != 4 or drive.shape[0] < 1:
        raise ValueError("drive must have shape (n,4)")
    if not np.all(np.isfinite(drive)):
        raise ValueError("drive must be finite")
    optical_inputs = np.asarray(
        [wavelength_m, reflectivity_a, reflectivity_b], dtype=float
    )
    if not np.all(np.isfinite(optical_inputs)):
        raise ValueError("optical inputs must be finite")
    if wavelength_m <= 0 or not (0 < reflectivity_a < 1 and 0 < reflectivity_b < 1):
        raise ValueError("invalid wavelength or reflectivity")
    input_field = drive[:, 2] + 1j * drive[:, 3]
    t_a = math.sqrt(1.0 - float(reflectivity_a))
    q = math.sqrt(float(reflectivity_a) * float(reflectivity_b))
    phase = np.exp(-4j * math.pi * drive[:, 1] / float(wavelength_m))
    field = np.empty(drive.shape[0], dtype=complex)
    previous = 0.0j
    for j in range(drive.shape[0]):
        previous = t_a * input_field[j] + q * phase[j] * previous
        field[j] = previous
    return field

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nd=np.column_stack((np.arange(5.),np.zeros(5),np.ones(5),np.zeros(5)))", "call":"propagate_cavity_field(d,1.0,0.81,0.64)", "gold_call":"_oracle_propagate_cavity_field(d,1.0,0.81,0.64)"},
        {"setup":"import numpy as np\nd=np.column_stack((np.arange(4.),np.full(4,0.25),np.ones(4),np.zeros(4)))", "call":"propagate_cavity_field(d,1.0,0.81,0.64)", "gold_call":"_oracle_propagate_cavity_field(d,1.0,0.81,0.64)"},
        {"setup":"import numpy as np\nd=np.array([[0.,0.,1.,0.],[1.,0.1,0.,1.],[2.,-0.2,0.5,-0.25]])", "call":"propagate_cavity_field(d,0.8,0.9,0.7)", "gold_call":"_oracle_propagate_cavity_field(d,0.8,0.9,0.7)"},
        {"setup":"import numpy as np\nd=np.array([[0.,0.,0.,1.]])", "call":"propagate_cavity_field(d,1.2,0.75,0.5)", "gold_call":"_oracle_propagate_cavity_field(d,1.2,0.75,0.5)"},
        {"setup":"import numpy as np\nd=np.column_stack((np.arange(6.),np.linspace(-.2,.2,6),np.cos(np.arange(6.)),np.sin(np.arange(6.))))", "call":"propagate_cavity_field(d,0.9,0.95,0.85)", "gold_call":"_oracle_propagate_cavity_field(d,0.9,0.95,0.85)"},
        {"setup":"import numpy as np\nd=np.column_stack((np.arange(7.),np.full(7,.5),np.ones(7),np.ones(7)))", "call":"propagate_cavity_field(d,2.0,0.99,0.999)", "gold_call":"_oracle_propagate_cavity_field(d,2.0,0.99,0.999)"},
        {"setup":"import numpy as np\nd=np.column_stack((np.arange(8.),np.linspace(0,.7,8),np.linspace(1,.3,8),np.linspace(-.2,.4,8)))", "call":"propagate_cavity_field(d,1.1,0.6,0.8)", "gold_call":"_oracle_propagate_cavity_field(d,1.1,0.6,0.8)"}
    ]
