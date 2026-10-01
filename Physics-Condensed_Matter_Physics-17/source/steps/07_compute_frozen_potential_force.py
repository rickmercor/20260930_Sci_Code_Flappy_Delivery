"""
Reconstruct the energy-weighted density from the unextracted density action and return the Hamiltonian, overlap-Pulay, and total electronic force contributions.

Forces in nonorthogonal coordinates contain an overlap derivative contracted with an energy-weighted density. In a finite compressed calculation, forming this kernel before spatial extraction is not interchangeable with multiplying an already extracted density.

Returns
-------
np.ndarray of shape (3,), ordered Hamiltonian force, overlap-Pulay force, total force.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def compute_frozen_potential_force(overlap_map: np.ndarray, hamiltonian: np.ndarray, density_action: np.ndarray, density: np.ndarray, overlap_tangent: np.ndarray, hamiltonian_tangent: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float) -> np.ndarray:
    """Compute a frozen-potential nonorthogonal electronic force.

    Parameters
    ----------
    overlap_map : np.ndarray
        Reconstructed real symmetric inverse-square-root M, shape (n,n).
        The inverse-overlap approximation is the ordered product M M.
    hamiltonian : np.ndarray
        Physical real symmetric H in the original basis, shape (n,n), not the
        sparse orthogonal Hamiltonian used to construct the density Krylov space.
    density_action : np.ndarray
        Unextracted original-coordinate density action X, shape (n,m).
    density : np.ndarray
        Already extracted real symmetric density rho, shape (n,n).
    overlap_tangent : np.ndarray
        Real symmetric dS/dx_a, shape (n,n), in inverse bohr.
    hamiltonian_tangent : np.ndarray
        Real symmetric dH/dx_a at frozen charge and interaction kernel, (n,n),
        in eV/bohr.
    distances : np.ndarray
        Nonnegative finite symmetric (n,n) distances with zero diagonal.
    colors : np.ndarray
        Contiguous integer-valued (n,) color labels covering the m action columns.
    signs : np.ndarray
        Real (n,) values +/-1; both integer and floating dtypes are accepted.
    keep_radius : float
        Finite nonnegative inclusive radius for the energy-weighted extraction.

    Returns
    -------
    force_terms : np.ndarray
        Real length-3 vector [F_H,F_S,F_H+F_S] in eV/bohr, where
        F_H=-Tr(rho dH) and F_S=Tr(rho_E dS). Form the original-coordinate
        energy-weighted action by inverse-overlap times H times X, then use
        extract_sparse_operator with zero magnitude threshold for rho_E.
        There is no spin factor, repulsive term, or charge derivative.

    Raises
    ------
    ValueError
        If any matrix is nonfinite, complex, nonsymmetric where required, or
        shape-incompatible, or any extraction constraint fails. Symmetry uses
        absolute tolerance 1e-12.
    """
    return np.zeros(3,dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_frozen_potential_force(overlap_map: np.ndarray, hamiltonian: np.ndarray, density_action: np.ndarray, density: np.ndarray, overlap_tangent: np.ndarray, hamiltonian_tangent: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float) -> np.ndarray:
    import numpy as np
    if any(np.iscomplexobj(a) for a in (overlap_map,hamiltonian,density_action,density,overlap_tangent,hamiltonian_tangent)):
        raise ValueError("all matrices must be real")
    m=np.asarray(overlap_map,dtype=float);h=np.asarray(hamiltonian,dtype=float)
    x=np.asarray(density_action,dtype=float);rho=np.asarray(density,dtype=float)
    ds=np.asarray(overlap_tangent,dtype=float);dh=np.asarray(hamiltonian_tangent,dtype=float)
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or m.shape[0]<1:
        raise ValueError("overlap map must be a square matrix")
    n=m.shape[0]
    for a in (m,h,rho,ds,dh):
        if a.shape!=(n,n) or not np.all(np.isfinite(a)) or not np.allclose(a,a.T,rtol=0,atol=1e-12):
            raise ValueError("matrices must be finite, symmetric and shape-compatible")
    if x.ndim!=2 or x.shape[0]!=n or x.shape[1]<1 or not np.all(np.isfinite(x)):
        raise ValueError("density action must be a finite block")
    xe=m@(m@(h@x))
    rhoe=_oracle_extract_sparse_operator(xe,distances,colors,signs,keep_radius,0.0)
    band=-float(np.sum(rho*dh.T))
    pulay=float(np.sum(rhoe*ds.T))
    return np.array([band,pulay,band+pulay],dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': 'import numpy as np\n'
               'M=np.array([[1.,.1],[.1,.9]]);H=np.array([[-.5,.2],[.2,.1]]);X=np.array([[.6,.2],[-.1,-.3]]);rho=np.array([[.6,-.15],[-.15,.3]]);dS=np.array([[.02,.1],[.1,-.03]]);dH=np.array([[.3,-.2],[-.2,.1]]);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);s=np.array([1.,-1.])',
      'call': 'compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)',
      'gold_call': '_oracle_compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)'},
     {'setup': 'import numpy as np\n'
               'M=np.eye(2);H=np.diag([-1.,2.]);X=np.diag([.7,.2]);rho=X.copy();dS=np.zeros((2,2));dH=np.diag([.3,-.1]);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);s=np.ones(2)',
      'call': 'compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)',
      'gold_call': '_oracle_compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)'},
     {'setup': 'import numpy as np\n'
               'M=np.array([[1.1,.08,0.],[.08,.9,.04],[0.,.04,1.0]]);H=np.array([[-.5,.2,.05],[.2,.1,-.15],[.05,-.15,.3]]);X=np.array([[.6,.15],[.2,-.3],[-.4,.2]]);rho=np.array([[.6,.025,0.],[.025,.3,0.],[0.,0.,.4]]);dS=np.array([[.01,.1,.03],[.1,-.02,.05],[.03,.05,.04]]);dH=np.zeros((3,3));D=np.array([[0.,1.,3.],[1.,0.,2.],[3.,2.,0.]]);c=np.array([0,1,0]);s=np.array([1.,-1.,-1.])',
      'call': 'compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)',
      'gold_call': '_oracle_compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)'},
     {'setup': 'import numpy as np\n'
               'M=np.eye(2);H=np.array([[0.,1.],[0.,0.]]);X=np.eye(2);rho=np.eye(2);dS=np.eye(2);dH=np.eye(2);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);s=np.ones(2)\n'
               'def run_model():\n'
               '    try:\n'
               '        compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'M=np.eye(2);H=np.eye(2);X=np.eye(2);rho=np.eye(2);dS=np.eye(3);dH=np.eye(2);D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);s=np.ones(2)\n'
               'def run_model():\n'
               '    try:\n'
               '        compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compute_frozen_potential_force(M,H,X,rho,dS,dH,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
