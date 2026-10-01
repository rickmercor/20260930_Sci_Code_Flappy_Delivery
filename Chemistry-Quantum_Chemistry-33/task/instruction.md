# Chemistry-Quantum_Chemistry-33

## Background

This audit uses a deterministic float64 tensor fixture to test a source-derived molecular-orbital representation, symmetry-preserving communication, amplitude readouts, and a coupled-cluster energy contraction. All seeded arrays, analytic weights, graph geometry, checksums, and the scalar reduction are benchmark conventions. The paper-specific choices are intentionally omitted from this background because recovering them is part of the scientific task. Each recovered choice changes at least one downstream amplitude, energy term, diagnostic, or the final value of J.

## Problem

The attached paper presents a symmetry-aware molecular-orbital architecture for coupled-cluster singles and doubles. Explain how the paper motivates the molecular-orbital architecture, and recover the source-specific query/key normalization, signed attention aggregation, and odd correlation-order choices.

Implement the seven ordered steps in float64. The default call is compute_molecular_orbital_audit(seed=33027, atom_count=6, occupied_count=3, virtual_count=3, radial_channels=4, hidden_channels=4, heads=3, cutoff=2.25, epsilon=1e-8). Let P=occupied_count+virtual_count and M=3. Use default_rng(seed). Draw coefficients from Normal(0,0.42) with shape (P,A,R,M), set shell_counts=1+((3*arange(A)+1) mod R), and use W[k,r]=(sin((k+1)(r+2)/5)+0.21*cos((k+2)(r+1)/7))/sqrt(R). Split orbitals into fragments after ceil(P/2), and set every attention score between unequal fragment labels to zero. Use Q[h,j,k]=0.19*sin((h+1)(j+2)(k+1)/9), K[h,j,k]=0.17*cos((h+2)(j+1)(k+2)/11), V[h,j,k]=0.15*sin((h+3)(j+1)+(k+2)/8), and head_weights[h]=0.5+0.2*cos(h+1). Put atom a at angle 2*pi*a/A, radius 1.15+0.08*sin(3*angle), position [radius*cos(angle),radius*sin(angle),0.22*sin(2*angle+0.3)]. Use local maps L[j,k]=0.075*cos((j+1)(k+2)/4) and C[j,k]=0.012*sin((j+2)(k+1)/5). Let H=hidden_channels+2. The T1 maps are 0.23*cos((j+1)(k+2)/6), 0.31*sin((j+2)(h+1)/7), and 0.18*cos(h+1). Occupied energies are -1.10-0.17*arange(O); virtual energies are 0.25+0.21*arange(V). Draw raw g from Normal(0,0.09), then set g=0.25*(raw-raw.swapaxes(0,1)-raw.swapaxes(2,3)+raw.transpose(1,0,3,2)); MP2 is g divided by the occupied-pair minus virtual-pair denominator. The T2 maps are 0.20*sin((j+2)(k+1)/5), 0.27*cos((j+1)(h+2)/8), and 0.14*sin(h+1).

Benchmark conventions for this finite fixture: Step 1 zero-pads masked radial slots and applies the shared channel map. Define embedding_checksum = dot(arange(1,N+1), embedded.ravel(order='C'))/N. In Step 3, each ordered center-neighbor pair with distinct atoms and distance d < cutoff contributes exp(-d/cutoff) times its neighbor features. The updated state is the residual features plus the linear map of the aggregate plus the cubic map of aggregate*sum_m(aggregate**2). In Step 2, output_norms has shape (heads,P) and records each head-orbital output norm before renormalization and head mixing. Define t1_checksum = dot(arange(1,N+1), T1.ravel(order='C')) with no division by N. In Step 5, normalize each orbital separately over atom, channel, and magnetic axes before forming occupied-virtual pair features; use the factorized product pair(i,a)*pair(j,b) for the correction, without a second pair-feature normalization. For the finite fixture, the three magnetic components are real orthonormal vector coordinates. Both readout scalar pair contractions use the unscaled Euclidean magnetic dot product, summed over atoms while retaining the output channel; the analytic maps are specified in this convention. For the doubles correction, apply the normalized exchange projector to the raw readout u: correction_ijab=(u_ijab-u_jiab-u_ijba+u_jiba)/4. Treat all amplitudes and integrals in the supplied antisymmetric spin-orbital convention. No additional transformer layer-normalization block or residual is included beyond the explicitly stated stages. These are fixture rules, not additional paper facts to infer.

Report seventeen diagnostics. embedding_checksum and t1_checksum are the upstream checksums. cross_score_max is max(abs(scores)) over unequal-fragment pairs; valid_score_mean is mean(abs(scores)) over equal-fragment pairs. directed_edges is the Step 3 edge count. state_norm=norm(updated), attention_norm=mean(output_norms), pair_norm=norm(t1_pairs), padding_zero is the maximum absolute padded coefficient in masked radial slots, t2_checksum=dot(arange(1,N+1),abs(T2.ravel()))/N in C order, correction_ratio=norm(correction)/(norm(mp2)+epsilon), and exchange_residual is the largest occupied- or virtual-index antisymmetry mismatch. correlation_energy, singles_energy, doubles_energy, amplitude_norm, and contraction_checksum are the five Step 6 returns. Use amplitude_norm=sqrt(sum(T1**2)+sum(T2**2)). Define tau_ijab=T2_ijab+T1_ia*T1_jb-T1_ib*T1_ja and contraction_checksum=dot(arange(1,N+1),(tau*g).ravel(order='C'))/N, where N=O*O*V*V. The energy terms use the antisymmetrized-integral CC expression, with singles_energy denoting its T1-product contribution and doubles_energy its T2 contribution. Finally compute J=exp(-abs(correlation_energy))*(1+valid_score_mean+0.25*correction_ratio)/(1+amplitude_norm+0.02*state_norm+0.01*attention_norm). Round only J to eight decimal places. The public orchestrator must call public Steps 1 through 6 in order.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_embed_localized_orbitals

Goal
----
Validate coefficients with shape (P,A,R,M), integer shell_counts with shape (A,), and weights with shape (K,R). Zero every radial slot r >= shell_counts[a], then compute embedded[p,a,k,m] = sum_r weights[k,r]*padded[p,a,r,m]. Return padded, embedded, and dot(arange(1,N+1), embedded.ravel())/N in C order.

```python
def embed_localized_orbitals(coefficients: "np.ndarray", shell_counts: "np.ndarray", weights: "np.ndarray") -> tuple:
    """Return padded coefficients, embedded orbital graphs, and a checksum.

    Parameters
    ----------
    coefficients : np.ndarray
        Float array with shape (P, A, R, M). Entries at radial indices greater
        than or equal to shell_counts[a] are ignored as element-dependent padding.
    shell_counts : np.ndarray
        Integer array with shape (A,) and values in [1, R].
    weights : np.ndarray
        Shared equivariant radial map with shape (K, R).

    Returns
    -------
    padded : np.ndarray
        Coefficients after invalid radial slots are set to zero, shape (P,A,R,M).
    embedded : np.ndarray
        Shared-weight orbital graph states with shape (P,A,K,M).
    checksum : float
        Weighted checksum of embedded in C order.

    Raises
    ------
    ValueError
        If an input has an invalid shape, non-finite values, or an invalid shell count.
    """
    return None, None, None
```

### Step 2

02_apply_signed_mo_attention

Goal
----
Validate embedded shape (P,A,K,M), integer fragment_ids shape (P,), three weight tensors shape (H,K,K), head_weights shape (H,), and positive epsilon. Project query, key, and value channels with the supplied head matrices. Recover the source normalization and aggregation rule, mask scores for unequal fragment labels, combine heads with head_weights, and return mixed states, the H by P by P score tensor, and the pre-renormalization H by P output norms.

```python
def apply_signed_mo_attention(embedded: "np.ndarray", fragment_ids: "np.ndarray", q_weights: "np.ndarray", k_weights: "np.ndarray", v_weights: "np.ndarray", head_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return mixed states, masked attention scores, and output norms.

    The implementation must recover from the source paper the normalization,
    attention aggregation, and fragment-coupling choices that preserve the stated
    orbital sign, rotation, and separated-fragment behavior.

    Parameters
    ----------
    embedded : np.ndarray
        Orbital states with shape (P,A,K,M).
    fragment_ids : np.ndarray
        Integer fragment label for each orbital, shape (P,).
    q_weights, k_weights, v_weights : np.ndarray
        Head-specific channel maps with shape (H,K,K).
    head_weights : np.ndarray
        Finite head-combination coefficients with shape (H,).
    epsilon : float
        Positive normalization stabilizer.

    Returns
    -------
    mixed : np.ndarray
        Attention-mixed states with shape (P,A,K,M).
    scores : np.ndarray
        Masked signed scores with shape (H,P,P).
    output_norms : np.ndarray
        Pre-renormalization output norms with shape (H,P).

    Raises
    ------
    ValueError
        If shapes, fragment labels, numeric values, or epsilon are invalid.
    """
    return None, None, None
```

### Step 3

03_apply_odd_local_mixing

Goal
----
Validate features shape (P,A,K,M), positions shape (A,3), two channel maps shape (K,K), and positive cutoff. For each ordered atom pair inside the strict cutoff, add exp(-distance/cutoff)*features[:,neighbor] to the center aggregate and require at least one edge. Form the rotationally equivariant odd cubic as aggregate*sum(aggregate*aggregate,axis=-1,keepdims=True). Apply linear_weights to aggregate and cubic_weights to that cubic, add both to the residual features, and return updated, aggregate, and the directed-edge count.

```python
def apply_odd_local_mixing(features: "np.ndarray", positions: "np.ndarray", cutoff: float, linear_weights: "np.ndarray", cubic_weights: "np.ndarray") -> tuple:
    """Return updated orbital states, the local aggregate, and edge count.

    Recover from the source paper which polynomial orders are admissible and why
    the finite message-passing cutoff is required. This benchmark uses the
    documented radial factor exp(-d/cutoff) and an explicit residual connection.

    Parameters
    ----------
    features : np.ndarray
        Orbital states with shape (P,A,K,M).
    positions : np.ndarray
        Atomic positions with shape (A,3).
    cutoff : float
        Positive finite neighbor cutoff.
    linear_weights, cubic_weights : np.ndarray
        Channel maps with shape (K,K).

    Returns
    -------
    updated : np.ndarray
        Residual plus odd local update, shape (P,A,K,M).
    aggregate : np.ndarray
        Distance-weighted neighbor aggregate, shape (P,A,K,M).
    directed_edge_count : int
        Number of directed atom pairs inside the cutoff.

    Raises
    ------
    ValueError
        If shapes, values, or cutoff are invalid, or if no directed edge exists.
    """
    return None, None, None
```

### Step 4

04_readout_single_amplitudes

Goal
----
Compute the source-derived singles readout for the supplied orbital states and maps. Return T1 with shape (O,V), invariant pair features with shape (O,V,K), and the requested signed T1 checksum. Follow the finite-fixture unscaled real-vector scalar-coupling convention stated in the problem statement.

```python
def readout_single_amplitudes(features: "np.ndarray", occupied_count: int, pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return T1 amplitudes, invariant pair features, and a checksum.

    Use the finite-fixture scalar-coupling and checksum conventions in the
    problem statement. Recover the source readout method for the supplied maps.

    Parameters
    ----------
    features : np.ndarray
        Final orbital states with shape (P,A,K,M).
    occupied_count : int
        Number of occupied orbitals, strictly between zero and P.
    pair_weights : np.ndarray
        Shared channel map with shape (K,K).
    hidden_weights : np.ndarray
        Readout hidden map with shape (K,H).
    output_weights : np.ndarray
        Readout output map with shape (H,).
    epsilon : float
        Positive finite normalization stabilizer.

    Returns
    -------
    t1 : np.ndarray
        Singles amplitudes with shape (n_occ,n_virt).
    pair_features : np.ndarray
        Invariant contracted features with shape (n_occ,n_virt,K).
    sign_checksum : float
        Weighted checksum of T1 in C order.

    Raises
    ------
    ValueError
        If shapes, counts, values, or epsilon are invalid.
    """
    return None, None, None
```

### Step 5

05_readout_double_amplitudes

Goal
----
Compute the source-derived doubles readout using the supplied orbital states, MP2 amplitudes, and readout maps. Return T2, its correction relative to MP2, and the largest occupied- or virtual-index exchange residual. Use the finite-fixture pair normalization, unscaled scalar coupling, factorized pair product, and normalized exchange projector stated in the problem statement.

```python
def readout_double_amplitudes(features: "np.ndarray", occupied_count: int, mp2_amplitudes: "np.ndarray", pair_weights: "np.ndarray", hidden_weights: "np.ndarray", output_weights: "np.ndarray", epsilon: float = 1e-8) -> tuple:
    """Return antisymmetric T2 amplitudes, corrections, and residual.

    Use the orbital normalization, scalar-coupling, factorized pair-product,
    and normalized exchange-projector fixture conventions in the problem statement.
    Recover the source readout method for the supplied maps and MP2 baseline.

    Parameters
    ----------
    features : np.ndarray
        Final orbital states with shape (P,A,K,M).
    occupied_count : int
        Number of occupied orbitals, strictly between zero and P.
    mp2_amplitudes : np.ndarray
        Baseline shape (O,O,V,V), antisymmetric in occupied and virtual pairs.
    pair_weights : np.ndarray
        Channel map with shape (K,K).
    hidden_weights : np.ndarray
        Readout hidden map with shape (K,H).
    output_weights : np.ndarray
        Readout output map with shape (H,).
    epsilon : float
        Positive finite normalization stabilizer.

    Returns
    -------
    t2 : np.ndarray
        Doubles amplitudes with shape (O,O,V,V).
    correction : np.ndarray
        Learned correction with the same shape as t2.
    exchange_residual : float
        Largest occupied- or virtual-pair antisymmetry mismatch.

    Raises
    ------
    ValueError
        If shapes, antisymmetry, counts, values, or epsilon are invalid.
    """
    return None, None, None
```

### Step 6

06_compute_cc_correlation_audit

Goal
----
Validate T1 shape (O,V), T2 and antisymmetrized integrals shape (O,O,V,V), finite values, and separate occupied- and virtual-index antisymmetry for both rank-four tensors. Form tau=T2+T1_ia*T1_jb-T1_ib*T1_ja. Return total correlation energy, 0.25*sum((T1_ia*T1_jb-T1_ib*T1_ja)*g), 0.25*sum(T2*g), the joint T1/T2 norm, and the weighted C-order checksum of tau*g.

```python
def compute_cc_correlation_audit(t1: "np.ndarray", t2: "np.ndarray", antisymmetrized_integrals: "np.ndarray") -> tuple:
    """Return correlation-energy components and amplitude diagnostics.

    Both T2 and g must be antisymmetric separately under occupied-index exchange
    and virtual-index exchange. The contraction uses
    tau_ijab=T2_ijab+T1_ia*T1_jb-T1_ib*T1_ja.

    Parameters
    ----------
    t1 : np.ndarray
        Singles amplitudes with shape (O,V).
    t2 : np.ndarray
        Doubles amplitudes with shape (O,O,V,V).
    antisymmetrized_integrals : np.ndarray
        Integrals <ij||ab> with the same shape and antisymmetries as t2.

    Returns
    -------
    correlation_energy : float
        Sum of singles and doubles contributions.
    singles_contribution : float
        0.25*sum((T1_ia*T1_jb-T1_ib*T1_ja)*g_ijab).
    doubles_contribution : float
        0.25*sum(T2_ijab*g_ijab).
    amplitude_norm : float
        Joint Euclidean norm of T1 and T2.
    contraction_checksum : float
        Weighted C-order checksum dot(arange(1,N+1), (tau*g).ravel())/N,
        where N=O*O*V*V and all indices have the order (i,j,a,b).

    Raises
    ------
    ValueError
        If shapes, values, dimensions, or required antisymmetries are invalid.
    """
    return None, None, None, None, None
```

### Step 7

07_compute_molecular_orbital_audit

Goal
----
This final orchestrator uses the default signature values shown below and default_rng(seed). Construct every seeded array and analytic weight exactly as specified in the problem statement. Form true antisymmetrized integrals g=0.25*(raw-raw.swapaxes(0,1)-raw.swapaxes(2,3)+raw.transpose(1,0,3,2)) and MP2=g/denominator. Call public Steps 1 through 6 in order. Compute all seventeen diagnostics using the definitions in the problem statement, including mean absolute valid scores and the absolute weighted T2 checksum. Compute the disclosed J, round only J to eight decimal places, and return J, diagnostics, T1, T2, and scores.

```python
def compute_molecular_orbital_audit(seed: int = 33027, atom_count: int = 6, occupied_count: int = 3, virtual_count: int = 3, radial_channels: int = 4, hidden_channels: int = 4, heads: int = 3, cutoff: float = 2.25, epsilon: float = 1e-8) -> tuple:
    """Return J, exactly seventeen named diagnostics, T1, T2, and scores.

    Construct the fully disclosed deterministic fixture and call public Steps 1
    through 6 in order. Only J is rounded, to eight decimal places.

    Parameters
    ----------
    seed : int
        Seed for default_rng.
    atom_count : int
        Number of atoms, from 3 through 9.
    occupied_count, virtual_count : int
        Positive occupied and virtual counts, each at most 5.
    radial_channels, hidden_channels : int
        Channel counts, each from 2 through 6.
    heads : int
        Attention heads, from 1 through 4.
    cutoff, epsilon : float
        Positive finite cutoff and stabilizer.

    Returns
    -------
    j : float
        Audit score rounded to eight decimal places.
    diagnostics : dict
        Exactly amplitude_norm, attention_norm, contraction_checksum,
        correlation_energy, correction_ratio, cross_score_max, directed_edges,
        doubles_energy, embedding_checksum, exchange_residual, pair_norm,
        padding_zero, singles_energy, state_norm, t1_checksum, t2_checksum,
        and valid_score_mean.
    t1 : np.ndarray
        Shape (occupied_count,virtual_count).
    t2 : np.ndarray
        Shape (occupied_count,occupied_count,virtual_count,virtual_count).
    scores : np.ndarray
        Shape (heads,P,P), where P=occupied_count+virtual_count.

    Raises
    ------
    ValueError
        If a configuration value or upstream dependency is invalid.
    """
    return None, None, None, None, None
```
