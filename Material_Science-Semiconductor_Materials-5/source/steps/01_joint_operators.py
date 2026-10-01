"""
Construct the coupled emitter–probe Hamiltonian family.

The phonon interaction picture uses the complete joint Hamiltonian for each coupling scale.

Returns
-------
result : np.ndarray     Complex shape (N+4,D,D), D=2**(N+1), in order H0, V, A, dot lowering,     and the N probe lowering operators. H(lambda)=H0+lambda*V is the     joint rotating-frame Hamiltonian; H0 has units ps^-1 and V is     dimensionless. A is the dot excited-state projector. Tensor order     is dot, probe 1, ..., probe N, each in ground, excited order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_operators(delta: float, omega: float, detunings: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    """Construct the joint Hamiltonian family and embedded operators.

    Parameters
    ----------
    delta : float
        Detuning omega0'-omega_L of the dot's polaron-shifted transition
        above the laser, in ps^-1.
    omega : float
        Real Rabi frequency in ps^-1; the drive amplitude is omega/2.
    detunings : np.ndarray
        Real shape (N,), 0<=N<=3, probe detunings omega_m-omega_L in ps^-1.
    directions : np.ndarray
        Real dimensionless shape (N,). Probe m has exchange coupling
        lambda*directions[m], where lambda is real and has units ps^-1.

    Returns
    -------
    result : np.ndarray
        Complex shape (N+4,D,D), D=2**(N+1), in order H0, V, A, dot lowering,
        and the N probe lowering operators. H(lambda)=H0+lambda*V is the
        joint rotating-frame Hamiltonian; H0 has units ps^-1 and V is
        dimensionless. A is the dot excited-state projector. Tensor order
        is dot, probe 1, ..., probe N, each in ground, excited order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_joint_operators(delta: float, omega: float, detunings: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    n=len(detunings)
    low=np.array([[0.,1.],[0.,0.]],complex)
    ops=[]
    for k in range(n+1):
        op=np.ones((1,1),complex)
        for j in range(n+1):
            op=np.kron(op,low if j==k else np.eye(2))
        ops.append(op)
    a=ops[0].conj().T@ops[0]
    h0=delta*a+omega*(ops[0]+ops[0].conj().T)/2
    exchange=np.zeros_like(h0)
    for d,r,v in zip(detunings,directions,ops[1:]):
        h0+=d*v.conj().T@v
        exchange+=r*(ops[0].conj().T@v+ops[0]@v.conj().T)
    return np.array([h0,exchange,a,*ops])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independently initialized scientific cases."""
    return [{'setup': 'import numpy as np\n'
               '## Joint coupling direction 1\n'
               'delta = 0.013\n'
               'omega = 0.22\n'
               'detunings = np.array([-0.45, -0.64, -0.84], dtype=float)\n'
               'directions = np.array([1.0, 1.2, 0.9], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 2\n'
               'delta = 0.0\n'
               'omega = 0.05\n'
               'detunings = np.array([], dtype=float)\n'
               'directions = np.array([], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 3\n'
               'delta = 0.0\n'
               'omega = 0.0\n'
               'detunings = np.array([0.0], dtype=float)\n'
               'directions = np.array([0.0], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 4\n'
               'delta = 0.1\n'
               'omega = 0.08\n'
               'detunings = np.array([0.1], dtype=float)\n'
               'directions = np.array([0.8], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 5\n'
               'delta = -0.08\n'
               'omega = 0.12\n'
               'detunings = np.array([-0.08, 0.04], dtype=float)\n'
               'directions = np.array([0.7, 1.1], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 6\n'
               'delta = 0.0\n'
               'omega = 0.2\n'
               'detunings = np.array([0.0, 0.0], dtype=float)\n'
               'directions = np.array([1.0, 1.0], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 7\n'
               'delta = 0.03\n'
               'omega = 0.0\n'
               'detunings = np.array([-0.3, 0.3], dtype=float)\n'
               'directions = np.array([0.0, 0.9], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 8\n'
               'delta = 0.02\n'
               'omega = -0.17\n'
               'detunings = np.array([-0.4, -0.2, 0.1], dtype=float)\n'
               'directions = np.array([0.8, 1.2, 0.7], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 9\n'
               'delta = -0.03\n'
               'omega = 0.24\n'
               'detunings = np.array([0.2, 0.2, 0.2], dtype=float)\n'
               'directions = np.array([1.0, 1.0, 1.0], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11},
     {'setup': 'import numpy as np\n'
               '## Joint coupling direction 10\n'
               'delta = 0.0\n'
               'omega = 0.22\n'
               'detunings = np.array([-0.01], dtype=float)\n'
               'directions = np.array([-1.3], dtype=float)\n',
      'call': 'joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'gold_call': '_oracle_joint_operators(delta, omega, detunings.copy(), directions.copy())',
      'tol': 1e-11}]
