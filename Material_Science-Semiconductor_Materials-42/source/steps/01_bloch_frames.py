"""
Evaluate the supplied lattice-gauge Hamiltonian and its canonical Bloch frames.

Evaluate the supplied lattice-gauge Hamiltonian and its canonical Bloch frames.

This constructed three-orbital benchmark uses the supplied paper’s method and Supplementary Information (SI). Its Hamiltonian, observations and discretization are synthetic. Lengths are in Å, momenta in Å⁻¹, and energies in eV. The lattice is square with \(a=3.2\), cell area \(a^2\), orthogonal point orbitals at \(\boldsymbol\tau/a=((0,0),(0.37,0.12),(0.16,0.43))\), and heights \(z_a/d=(0,0.29,-0.23)\). With \(x=ak_x,y=ak_y\), the upper triangle of the lattice-gauge Bloch Hamiltonian is
\[
H_{00}=-\mu+0.17(\cos x+\cos y),\quad H_{11}=\mu+0.11\cos x-0.08\cos y,\quad H_{22}=\mu+1.15+0.09\cos(x+y),
\]
\[
H_{01}=0.44+0.31e^{-ix}+0.23e^{-iy},\quad H_{02}=0.29-0.19e^{ix}+0.21e^{-i(x+y)},\quad H_{12}=0.16+0.13e^{-iy};\qquad H_{ba}=H_{ab}^*.
\]
Eigenvalues \(e_b\) increase with \(b=0,1,2\); eigenvector columns \(U_{ab}\) are normalized. Occupations are \((1,0,0)\), spin degeneracy is two, temperature is zero, and the vertical Tamm–Dancoff kernel is the attractive direct term. The public eigenvector interface fixes each column's largest-magnitude component to real nonnegative, with the lowest component index resolving ties.

Returns
-------
return result  # complex ndarray, shape (K,4,3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bloch_frames(kpoints: np.ndarray, mass: float) -> np.ndarray:
    """Parameters
    ----------
    kpoints : finite real ndarray, shape (K,2), K>=1
        Ordered momenta (kx,ky), in inverse angstroms.
    mass : finite positive float
        Hamiltonian parameter mu, in eV, in the stated three-orbital model.
    Returns
    -------
    complex ndarray, shape (K,4,3)
        Row 0 holds the three ascending eigenenergies in eV (zero imaginary part).
        Rows 1:4 hold dimensionless eigenvector columns, in orbital order.
        Each column uses the largest-component real-nonnegative gauge; lowest
        orbital index breaks a largest-magnitude tie.
    Raises
    ------
    ValueError : malformed/nonfinite momenta or nonpositive/nonfinite mass."""
    return np.zeros((len(kpoints),4,3),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bloch_frames(kpoints, mass):
    k = np.asarray(kpoints, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or not len(k) or not np.isfinite(k).all() or not np.isfinite(mass) or mass <= 0:
        raise ValueError('finite nonempty (K,2) momenta and positive mass required')
    x, y = 3.2 * k[:, 0], 3.2 * k[:, 1]
    h = np.zeros((len(k), 3, 3), dtype=complex)
    h[:, 0, 0] = -mass + .17 * (np.cos(x) + np.cos(y))
    h[:, 1, 1] = mass + .11 * np.cos(x) - .08 * np.cos(y)
    h[:, 2, 2] = mass + 1.15 + .09 * np.cos(x + y)
    h[:, 0, 1] = .44 + .31 * np.exp(-1j*x) + .23 * np.exp(-1j*y)
    h[:, 0, 2] = .29 - .19 * np.exp(1j*x) + .21 * np.exp(-1j*(x+y))
    h[:, 1, 2] = .16 + .13 * np.exp(-1j*y)
    h += np.triu(h, 1).conj().swapaxes(-1, -2)
    energy, vectors = np.linalg.eigh(h)
    # A documented gauge fixes the interface, not a physical observable.
    for i in range(len(k)):
        for band in range(3):
            j = int(np.argmax(np.abs(vectors[i, :, band])))
            vectors[i, :, band] *= np.exp(-1j*np.angle(vectors[i, j, band]))
    return np.concatenate((energy[:, None, :], vectors), axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nk=np.array([[0, 0]])\n', 'call': 'bloch_frames(k,1.1)', 'gold_call': '_oracle_bloch_frames(k,1.1)'}, {'setup': 'import numpy as np\nk=np.array([[0.31, -0.22], [-0.31, 0.22]])\n', 'call': 'bloch_frames(k,1.3)', 'gold_call': '_oracle_bloch_frames(k,1.3)'}, {'setup': 'import numpy as np\nk=np.array([[0.9817477042468103, 0], [0.9817477042468103, 0.9817477042468103]])\n', 'call': 'bloch_frames(k,1.6)', 'gold_call': '_oracle_bloch_frames(k,1.6)'}, {'setup': 'import numpy as np\nk=np.array([[0.19, 0.23], [2.1534954084936206, 0.23]])\n', 'call': 'bloch_frames(k,0.8)', 'gold_call': '_oracle_bloch_frames(k,0.8)'}, {'setup': 'import numpy as np\n\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: bloch_frames(np.array([[0.,0.]]),-1.))', 'gold_call': '_exception_code(lambda: _oracle_bloch_frames(np.array([[0.,0.]]),-1.))'}]
