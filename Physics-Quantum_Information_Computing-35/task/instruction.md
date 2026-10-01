# Physics-Quantum_Information_Computing-35

## Background

Logical magic-state preparation circuits contain non-Clifford operations and therefore are not directly covered by ordinary Gottesman-Knill simulation. A useful special case arises when the relevant logical Clifford is a Pauli-square-root Clifford: controlled versions of these gates have structured Pauli-error propagation, so sampled circuit-level Pauli faults can be represented by a polynomial-size Clifford error rather than by a full state vector. Fresh-ancilla measurement schedules are especially convenient because propagated controlled-Pauli factors can be commuted through later measurement rounds while any induced ancilla-ancilla Clifford factors remain at the end of the circuit.

Once a noisy preparation is reduced to a noiseless magic state followed by an end Clifford, fidelity can be estimated without tracking arbitrary global phases. Expanding the target density matrix in a Pauli basis converts the target fidelity into a weighted sum of Pauli expectation values, which can be evaluated through stabilizer-compatible simulations and renormalized by the postselection acceptance probability. The paper's full stabilizer-based approach separates error propagation from fidelity estimation and avoids a simulation cost exponential in the number of non-Clifford protocol gates. This small verification benchmark retains the compact propagation and Pauli-expectation contraction, while permitting dense evaluation of the completed factor list; that verification implementation does not claim the full method's scaling.

## Problem

A scalable logical magic-state simulator can replace a noisy non-Clifford preparation circuit by a noiseless target state acted on by a compact end-of-circuit Clifford error, provided circuit-level Pauli faults are propagated with the special structure of controlled Pauli-square-root Clifford (PSC) measurements. For this deterministic benchmark use the binary Pauli encoding `[phase,x...,z...]` for `i^phase X^x Z^z`, with phase modulo 4, and the canonical PSC data `alpha_power=1`, `p_code=[0,1,0,0,0]`, `q_codes=[[3,1,0,1,0],[2,0,0,0,1]]`, where `alpha=exp(i*pi*alpha_power/4)` and the two `q_codes` are mutually commuting Hermitian Paulis; every measurement round has a fresh control initialized and finally postselected in `|+>`, and no control is reused.

The two-qubit target is `|phi> = (cos(pi/8)|0> + sin(pi/8)|1>) tensor |0>`. Each row `[c,d0,d1]` below is a Pauli layer immediately before one controlled-PSC round, using `0=I,1=X,2=Y,3=Z`; propagate the faults with the paper's controlled-PSC factorization and recursive fresh-ancilla error-propagation rules, keeping the end error as a compact ordered Clifford-factor list throughout fault propagation. For this small verification benchmark, the subsequent postselection and Pauli-expectation contraction may evaluate that completed list using dense matrices; the compact-propagation requirement applies before that evaluation. The eight four-round schedules are `[[[0,0,0],[0,0,0],[0,0,0],[0,0,0]], [[1,1,0],[0,0,0],[0,0,0],[0,0,0]], [[0,0,0],[2,3,0],[0,0,0],[0,0,0]], [[3,1,1],[0,2,0],[0,0,0],[0,0,0]], [[0,0,2],[1,0,0],[0,3,1],[0,0,0]], [[2,1,3],[3,0,1],[1,2,0],[0,3,3]], [[0,2,2],[0,0,0],[2,1,0],[3,0,2]], [[1,3,0],[2,0,1],[3,1,2],[1,2,3]]]` with probabilities `[0.3271,0.1437,0.1079,0.0913,0.0842,0.0746,0.0934,0.0778]` in the same order.

For each schedule evaluate the accepted target contribution with the paper's phase-insensitive Pauli-rank fidelity construction, then combine schedules by probability-weighting the acceptance `A_s` and acceptance-weighted overlap `N_s` separately and return `sum_s p_s N_s / sum_s p_s A_s`. For auditability, keep the reasoning compact but include the PSC-square Pauli, the propagated factorization for a control-`X`/target-`XI` fault, the four gate-type counts `(Pauli, controlled-Pauli, PSC, CZ)` for schedules 2 and 8, representative nonzero Pauli coefficients of `|phi><phi|`, the `(A_s,N_s)` pairs for schedules 1, 2, 5, and 8, and the two final ensemble aggregates before taking their ratio; do not round intermediate values. For the control-X/target-XI diagnostic, use left-to-right matrix-product order $P'\,C(Q)\,(I_c\otimes U)^b$, where $P'$ is the joint control-and-target Pauli, $C(Q)$ is the controlled target Pauli, and $b\in\{0,1\}$. For the two gate-count diagnostics, use the corresponding recursive factorization convention: retain one Pauli factor per measurement round, including identity factors; emit a controlled-Pauli factor only when its target Pauli is not exactly $+I$ (nontrivial scalar phases count as nonidentity), emit a target-$U$ factor only when $b=1$, and retain every induced CZ factor. Do not delete, merge, cancel, or reorder generated factors before counting. Report the final conditional fidelity to twelve decimal places.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_psc_square_pauli.py

Goal
----
Recover the Pauli square of a Pauli-square-root Clifford from canonical data.

```python
def psc_square_pauli(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray") -> "np.ndarray":
    '''Return the encoded Pauli equal to U^2 for the canonical PSC U.

    Parameters
    ----------
    alpha_power : int
        Integer a in {0,...,7} for alpha = exp(i*pi*a/4).
    p_code : np.ndarray
        Length-(1+2k) integer Pauli code [phase, x..., z...] for P, using
        i^phase X^x Z^z.
    q_codes : np.ndarray
        Integer array of shape (m, 1+2k) containing mutually commuting
        Hermitian Pauli codes Q_j. Use the canonical convention
        U = alpha * P * exp(+pi*i/4 * sum_j Q_j), with P to the left of
        the positive quarter-turn exponential.

    Returns
    -------
    np.ndarray
        Length-(1+2k) integer Pauli-group code for U^2, with phase modulo 4.
    '''
    return square_code
```

### Step 2

02_psc_conjugate_pauli.py

Goal
----
Conjugate a Pauli by a canonically represented Pauli-square-root Clifford.

```python
def psc_conjugate_pauli(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    '''Return the encoded Pauli U R U^dagger, omitting the irrelevant alpha phase.

    Parameters
    ----------
    p_code : np.ndarray
        Length-(1+2k) encoded Pauli prefactor P.
    q_codes : np.ndarray
        Shape-(m,1+2k) mutually commuting Hermitian Pauli codes Q_j for
        U = alpha * P * exp(+pi*i/4 * sum_j Q_j), with this exact positive
        rotation sign and factor order.
    pauli_code : np.ndarray
        Length-(1+2k) Pauli-group code for R.

    Returns
    -------
    np.ndarray
        Length-(1+2k) integer code for U R U^dagger.
    '''
    return conjugated_code
```

### Step 3

03_psc_commutator_signature.py

Goal
----
Extract the source Pauli commutator and its PSC parity relation.

```python
def psc_commutator_signature(p_code: "np.ndarray", q_codes: "np.ndarray", pauli_code: "np.ndarray") -> "np.ndarray":
    '''Return Q' together with whether Q' commutes or anticommutes with U.

    Parameters
    ----------
    p_code, q_codes
        Canonical PSC Pauli data in the same encoding as earlier steps.
    pauli_code : np.ndarray
        Encoded target Pauli R.

    Returns
    -------
    np.ndarray
        Length-(2+2k) integer vector [q_phase, q_x..., q_z..., parity], where
        parity is 0 if U Q' U^dagger = Q' and 1 if it equals -Q'.
    '''
    return signature
```

### Step 4

04_controlled_psc_fault_factor.py

Goal
----
Factor an arbitrary Pauli fault propagated through a controlled PSC.

```python
def controlled_psc_fault_factor(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", control_label: int, target_code: "np.ndarray") -> "np.ndarray":
    '''Return a compact factorization of G F G^dagger for G=controlled-U.

    The returned factors use the fixed product order
    P_prime * controlled(Q) * (I_control tensor U)^b.

    Parameters
    ----------
    alpha_power : int
        Integer eighth-root phase exponent of the canonical PSC U.
    p_code, q_codes
        Canonical PSC Pauli data.
    control_label : int
        Fault Pauli on the fresh control: 0=I, 1=X, 2=Y, 3=Z.
    target_code : np.ndarray
        Length-(1+2k) encoded Pauli fault on the target register.

    Returns
    -------
    np.ndarray
        Integer vector [control_label, P'_target_code, Q_code, b]. Its length is
        2*(1+2k)+2; b is 0 or 1.
    '''
    return factor_code
```

### Step 5

05_compact_psc_schedule_propagation.py

Goal
----
Propagate a fresh-ancilla PSC measurement schedule as a compact Clifford gate list.

```python
def compact_psc_schedule_propagation(
    alpha_power: int,
    p_code: "np.ndarray",
    q_codes: "np.ndarray",
    fault_schedule: "np.ndarray",
) -> "np.ndarray":
    """Return the ordered compact end-error list for a pre-gate fault schedule.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    fault_schedule : np.ndarray
        Integer array of shape (r, k + 1). Row t contains the fresh-control
        Pauli label followed by k target labels, each using
        0=I, 1=X, 2=Y, 3=Z. The row is a Pauli layer immediately before
        controlled-U round t. Controls use zero-based round indices.

    Returns
    -------
    np.ndarray
        Integer array with row width 4 + 2*k. Rows are in left-to-right
        matrix-product order. The complete row formats are:

        Type 0: [0, control, control_label, phase, x..., z...].
            A Pauli on the named control and the k targets. The x and z
            blocks each contain k bits.

        Type 1: [1, control, -1, phase, x..., z...].
            Controlled-Q on the named control and the k targets. The x
            and z blocks each contain k bits.

        Type 2: [2, -1, -1, 0, x..., z...].
            Target-U, with both k-entry x and z blocks identically zero.
            The two unused index fields must be -1; every field after
            them must be zero.

        Type 3: [3, old_control, new_control, 0, x..., z...].
            CZ between the existing factor's control and the fresh
            control for the current round, in that order. Both k-entry
            x and z blocks are identically zero.

        Keep exactly one type-0 row for every measurement round, including
        an identity Pauli row. For round t, prepend local rows in the
        order type 0, optional nonidentity type 1, optional type 2, then
        the propagated older rows. Emit the local type-2 row only when
        the controlled-fault factorization has b=1.

        When propagating an older type-0 row, keep it and place any
        induced nonidentity type-1 row immediately after it. When
        propagating an older type-1 row, place any induced CZ row
        immediately before that type-1 row. Do not simplify, merge,
        cancel, drop, or reorder rows.

        An empty schedule returns an integer array of shape
        (0, 4 + 2*k).
    """
    return gate_rows
```

### Step 6

06_postselected_pauli_rank_statistics.py

Goal
----
Evaluate postselected phase-insensitive statistics from the compact propagated error.

```python
def postselected_pauli_rank_statistics(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedule: "np.ndarray") -> "np.ndarray":
    '''Return acceptance and acceptance-weighted target overlap for one schedule.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    target_state : np.ndarray
        Normalized k-qubit target state of length 2^k.
    fault_schedule : np.ndarray
        Shape-(r,k+1) pre-gate Pauli schedule.

    Returns
    -------
    np.ndarray
        Real length-2 vector [A,N], where A is postselection acceptance and N
        is the acceptance-weighted phase-insensitive target overlap.
    '''
    return statistics
```

### Step 7

07_ensemble_compact_magic_fidelity.py

Goal
----
Combine exact compact-propagation branches into an accepted-ensemble magic-state fidelity.

```python
def ensemble_compact_magic_fidelity(alpha_power: int, p_code: "np.ndarray", q_codes: "np.ndarray", target_state: "np.ndarray", fault_schedules: "np.ndarray", probabilities: "np.ndarray") -> float:
    '''Return the exact conditional fidelity of the accepted finite ensemble.

    Parameters
    ----------
    alpha_power, p_code, q_codes
        Canonical PSC data.
    target_state : np.ndarray
        Normalized k-qubit target state.
    fault_schedules : np.ndarray
        Shape-(N,r,k+1) collection of pre-gate Pauli schedules.
    probabilities : np.ndarray
        Length-N nonnegative schedule probabilities summing to one.

    Returns
    -------
    float
        Probability-weighted accepted target overlap divided by total acceptance.
    '''
    return fidelity
```
