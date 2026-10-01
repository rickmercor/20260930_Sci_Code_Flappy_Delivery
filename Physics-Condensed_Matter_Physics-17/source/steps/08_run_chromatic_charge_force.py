"""
Orchestrate the finite-SCF force benchmark, rebuilding the compressed Hamiltonian and fixed-population density after each charge update and once at the final charge.

Charge feedback changes the physical Hamiltonian and can change the sparse transformed support. The final force must use a density recomputed at the final mixed charge; a stale density or an initial-state-only evaluation changes the specified observable.

Returns
-------
float: final frozen-potential electronic force, rounded to 10 decimal places.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def run_chromatic_charge_force(positions: np.ndarray | None=None, beta: float=32.0, electron_count: float=8.0, mixing: tuple[float,...]=(0.6,0.35,0.75), hamiltonian_threshold: float=0.135, density_depth: int=2, force_index: int=0) -> float:
    """Run the deterministic charge-feedback force calculation.

    Parameters
    ----------
    positions : np.ndarray or None
        Exactly 18 finite strictly increasing positions in [0,24) bohr. None uses
        [0,.72,1.83,3.05,4.12,5.55,6.44,7.90,9.15,10.02,11.48,12.73,
         14.05,15.22,16.87,18.31,20.12,22.01]. Orbital order is fixed.
    beta : float
        Positive finite inverse temperature, eV**(-1).
    electron_count : float
        One-spin particle count in (0,18), also within every density-root bracket.
    mixing : tuple of float
        Finite mixing factors in [0,1]. Each factor specifies one update;
        the empty tuple is valid and makes no updates. Recompute the density
        once after the last update; do not make a further charge update.
    hamiltonian_threshold : float
        Nonnegative finite offdiagonal magnitude cutoff for the reconstructed
        orthogonal Hamiltonian; equality is retained after symmetrization.
    density_depth : int
        Positive canonical Krylov depth for each density evaluation.
    force_index : int
        Orbital displaced for the force tangent, in [0,18).

    Returns
    -------
    force : float
        Final frozen-potential electronic force in eV/bohr, rounded only at
        the end to 10 decimal places with ties to even.

    Raises
    ------
    ValueError
        If any argument violates its contract, the initial overlap is not
        positive definite, or any fixed-population density problem has invalid
        spectral weights or fails to bracket its requested population.

    Notes
    -----
    Call the preceding public step functions; do not reimplement their work.
    Length is 24, overlap exclusion/keep radii are 4.9/2.4, overlap depth is 3,
    overlap magnitude cutoff is .006, density exclusion/keep radii are 7.3/3.6,
    and transformed-Hamiltonian keep radius is 3.4. Both density kernels use
    zero magnitude threshold. The canonical rank tolerance is 1e-12.
    Overlap signs: [1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1].
    Density signs: [-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1].
    First obtain the inverse-square-root action Y and its extracted map M.
    Gamma=1.1*I+.15*exp(-D/3) in eV is elementwise and remains fixed.
    Initial q_i=.12*cos(2*pi*i/18)+.04*sin(4*pi*i/18), minus its arithmetic mean.
    At charge q use V=Gamma@q and H=H0+.5*S*(V[:,None]+V[None,:]).
    Reconstruct the transformed Hamiltonian from the compressed action M@H@Y,
    using the overlap colors/signs, radius 3.4 and hamiltonian_threshold.
    Obtain mu,rho,X through solve_fixed_population_density. For each supplied
    alpha update q=(1-alpha)*q+alpha*(diag(rho@S)-electron_count/18).
    At the final q hold q and Gamma fixed, use dH=dH0+.5*dS*(V_i+V_j),
    and call compute_frozen_potential_force using the physical H and X.
    Retain binary64 values throughout and do not mutate input arrays.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_chromatic_charge_force(positions: np.ndarray | None=None, beta: float=32.0, electron_count: float=8.0, mixing: tuple[float, ...]=(0.6, 0.35, 0.75), hamiltonian_threshold: float=0.135, density_depth: int=2, force_index: int=0) -> float:
    import numpy as np
    from numbers import Real, Integral
    if positions is None:
        x=np.array([0.,.72,1.83,3.05,4.12,5.55,6.44,7.90,9.15,10.02,11.48,12.73,14.05,15.22,16.87,18.31,20.12,22.01])
    else:
        if np.iscomplexobj(positions):raise ValueError("positions must be real")
        x=np.asarray(positions,dtype=float)
    if x.shape!=(18,):raise ValueError("exactly 18 positions are required")
    if isinstance(beta,(bool,np.bool_)) or not isinstance(beta,Real) or not np.isfinite(beta) or beta<=0:
        raise ValueError("beta must be finite and positive")
    if isinstance(electron_count,(bool,np.bool_)) or not isinstance(electron_count,Real) or not np.isfinite(electron_count) or not 0<electron_count<18:
        raise ValueError("electron_count must be in (0,18)")
    if np.iscomplexobj(mixing):raise ValueError("mixing must be real")
    alphas=np.asarray(mixing,dtype=float)
    if alphas.ndim!=1 or not np.all(np.isfinite(alphas)) or np.any(alphas<0) or np.any(alphas>1):
        raise ValueError("mixing must be a vector of factors in [0,1]")
    if isinstance(hamiltonian_threshold,(bool,np.bool_)) or not isinstance(hamiltonian_threshold,Real) or not np.isfinite(hamiltonian_threshold) or hamiltonian_threshold<0:
        raise ValueError("Hamiltonian threshold must be finite and nonnegative")
    if isinstance(density_depth,(bool,np.bool_)) or not isinstance(density_depth,Integral) or density_depth<1:
        raise ValueError("density_depth must be positive integer")
    d,s,h0,ds,dh0=_oracle_build_periodic_chain_tangent(x,24.0,force_index)
    signs_s=np.array([1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1])
    signs_r=np.array([-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1])
    c_s,o=_oracle_build_colored_signed_probes(d,4.9,signs_s)
    c_r,r=_oracle_build_colored_signed_probes(d,7.3,signs_r)
    q_s=_oracle_canonical_block_krylov_basis(s,o,3)
    y=_oracle_projected_inverse_sqrt_action(s,o,q_s)
    m=_oracle_extract_sparse_operator(y,d,c_s,signs_s,2.4,0.006)
    gamma=1.1*np.eye(18)+0.15*np.exp(-d/3.0)
    idx=np.arange(18)
    charge=0.12*np.cos(2*np.pi*idx/18)+0.04*np.sin(4*np.pi*idx/18)
    charge=charge-charge.mean()
    for iteration in range(alphas.size+1):
        potential=gamma@charge
        h=h0+0.5*s*(potential[:,None]+potential[None,:])
        kh_action=m@(h@y)
        kh=_oracle_extract_sparse_operator(kh_action,d,c_s,signs_s,3.4,hamiltonian_threshold)
        mu,rho,xrho=_oracle_solve_fixed_population_density(s,m,kh,r,d,c_r,signs_r,beta,electron_count,density_depth,3.6)
        if iteration<alphas.size:
            output_charge=np.sum(rho*s.T,axis=1)-float(electron_count)/18
            charge=(1-alphas[iteration])*charge+alphas[iteration]*output_charge
    h_tangent=dh0+0.5*ds*(potential[:,None]+potential[None,:])
    terms=_oracle_compute_frozen_potential_force(m,h,xrho,rho,ds,h_tangent,d,c_r,signs_r,3.6)
    return float(round(float(terms[2]),10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': '',
      'call': 'run_chromatic_charge_force()',
      'gold_call': '_oracle_run_chromatic_charge_force()'},
     {'setup': '',
      'call': 'run_chromatic_charge_force(mixing=())',
      'gold_call': '_oracle_run_chromatic_charge_force(mixing=())'},
     {'setup': '',
      'call': 'run_chromatic_charge_force(mixing=(0.6,))',
      'gold_call': '_oracle_run_chromatic_charge_force(mixing=(0.6,))'},
     {'setup': '',
      'call': 'run_chromatic_charge_force(hamiltonian_threshold=0.1349)',
      'gold_call': '_oracle_run_chromatic_charge_force(hamiltonian_threshold=0.1349)'},
     {'setup': '',
      'call': 'run_chromatic_charge_force(force_index=17)',
      'gold_call': '_oracle_run_chromatic_charge_force(force_index=17)'},
     {'setup': '\n'
               'def run_model():\n'
               '    try:\n'
               '        run_chromatic_charge_force(mixing=(0.6,1.1))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_run_chromatic_charge_force(mixing=(0.6,1.1))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': '\n'
               'def run_model():\n'
               '    try:\n'
               '        run_chromatic_charge_force(electron_count=18.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_run_chromatic_charge_force(electron_count=18.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
