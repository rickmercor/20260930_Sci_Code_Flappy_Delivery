"""
Convert the complex field into intracavity power and the approximate demodulated PDH signal.

Recover the demodulated PDH convention from the main paper's cavity observables. Preserve the complex input reference, stated demodulation phase, signed quadrature, and row order; return power and PDH columns.

Returns
-------
Return one real NumPy array of shape (n,2), ordered [power, PDH].
"""

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def cavity_observables(field, drive, demodulation_phase):
    """Return an n-by-2 real array with columns [power, PDH]."""
    return np.empty((0, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cavity_observables(field, drive, demodulation_phase):
    """Return columns [intracavity power, PDH] using Eq. (3)."""
    import numpy as np
    field = np.asarray(field, dtype=complex)
    drive = np.asarray(drive, dtype=float)
    if field.ndim != 1 or drive.ndim != 2 or drive.shape != (field.size, 4):
        raise ValueError("field and drive shapes disagree")
    if (not np.all(np.isfinite(field)) or not np.all(np.isfinite(drive))
            or not np.isfinite(float(demodulation_phase))):
        raise ValueError("inputs must be finite")
    input_field = drive[:, 2] + 1j * drive[:, 3]
    power = np.abs(field)**2
    pdh = -np.imag(np.exp(1j * float(demodulation_phase)) * np.conj(input_field) * field)
    return np.column_stack((power, pdh)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nf=np.array([1+2j,2-1j]); d=np.array([[0,0,1,0],[1,0,1,0]],float)", "call":"cavity_observables(f,d,0.)", "gold_call":"_oracle_cavity_observables(f,d,0.)"},
        {"setup":"import numpy as np\nf=np.array([1j,-1j]); d=np.array([[0,0,1,0],[1,0,0,1]],float)", "call":"cavity_observables(f,d,0.)", "gold_call":"_oracle_cavity_observables(f,d,0.)"},
        {"setup":"import numpy as np\nf=np.array([1+0j,1+1j,2-1j]); d=np.column_stack((np.arange(3),np.zeros(3),np.ones(3),np.zeros(3)))", "call":"cavity_observables(f,d,.25)", "gold_call":"_oracle_cavity_observables(f,d,.25)"},
        {"setup":"import numpy as np\nf=np.array([2-3j]); d=np.array([[0,0,0,1]],float)", "call":"cavity_observables(f,d,np.pi/2)", "gold_call":"_oracle_cavity_observables(f,d,np.pi/2)"},
        {"setup":"import numpy as np\nf=np.array([0j,1j,2j]); d=np.column_stack((np.arange(3),np.zeros(3),np.ones(3),np.ones(3)))", "call":"cavity_observables(f,d,-.4)", "gold_call":"_oracle_cavity_observables(f,d,-.4)"},
        {"setup":"import numpy as np\nf=np.array([1-1j,-2+.5j,.3+.7j]); d=np.array([[0,0,.2,.1],[1,0,-.3,.8],[2,0,.5,-.4]],float)", "call":"cavity_observables(f,d,1.1)", "gold_call":"_oracle_cavity_observables(f,d,1.1)"},
        {"setup":"import numpy as np\nf=np.exp(1j*np.linspace(0,1,9)); d=np.column_stack((np.arange(9),np.zeros(9),np.cos(np.arange(9)),np.sin(np.arange(9))))", "call":"cavity_observables(f,d,.73)", "gold_call":"_oracle_cavity_observables(f,d,.73)"}
    ]
