# Material_Science-Semiconductor_Materials-75

## Background

Spin qubits in silicon are attractive because the host can be isotopically purified until nuclear spins almost vanish, leaving very long coherence times. The difficulty lies in the band structure rather than in coherence. Bulk silicon has six equivalent conduction-band minima along the cubic axes, and biaxial strain in a Si/SiGe quantum well lifts four of them, leaving a pair along the growth direction. That surviving pair is the valley degree of freedom. It is a genuine two-level system sitting underneath the spin, and if the energy separating its two states is not comfortably larger than the scale on which the qubit is driven, gate operations mix spin with valley and the computational subspace is no longer purely spin.

Valley splitting has therefore become one of the central materials problems in silicon quantum electronics, and it is a stubborn one. The splitting is not a bulk property; it is produced at the interface, where the sharp confinement potential mixes states that are far apart in momentum. In the standard picture the mixing is scattering from a scalar potential: the abrupt band offset, and above all the random arrangement of germanium atoms in the alloy barrier, present the electron with a potential whose matrix element between the two valley states is off-diagonal. Because the arrangement of germanium is random, the resulting splitting is random too. Measured device to device it follows a broad distribution with weight close to zero, and dots that land in that low tail are effectively unusable. A good deal of experimental effort has gone into engineering the interface, deliberately alloying it, or growing oscillating germanium concentrations, all with the aim of pushing that distribution away from zero.

Underneath this work sits an assumption that is rarely examined. The valley pair is habitually described as a pseudospin, with the two valley states playing the role of spin up and spin down and the three Pauli matrices in that space treated as though they were the three components of a spin. If that were right, the symmetry properties of the valley operators would be the familiar ones, and in particular all three would be odd under time reversal, as the components of a spin are. Whether the analogy is right is not a matter of taste: the two valley states are Bloch functions belonging to a definite irreducible representation of the group of the wave vector at the X point of the diamond structure, and their transformation properties are fixed by that representation and by nothing else.

The X point is an unusual place. The diamond structure is non-symmorphic: it contains a glide operation built from a mirror and a quarter translation along the body diagonal, the translation that carries one atom of the two-atom basis onto the other. That fractional translation does not commute with the rest of the point group, and repairing the mismatch forces a face-diagonal lattice translation into the group of the wave vector. On the crystal that translation does nothing, but on a Bloch function carrying the phase of the X point it acts with a minus sign, and that single fact eliminates every one-dimensional representation from consideration. Only two-dimensional representations survive, which is why the band is degenerate at X in the first place and why the valleys exist at all.

Determining the transformation law of the valley operators therefore requires the explicit basis functions of that representation, and the literature on this point is thin and not uniformly reliable. The group of the wave vector at X of the diamond lattice is worked out in only a few places; one classic treatment carries an error in its character table, the modern presentation that corrects it substitutes the wrong mirrors, and the basis functions given in the older source, while perfectly valid as a basis, do not connect continuously onto the two bands running from X toward the zone centre. That last point is the decisive one. Two choices of basis inside the same two-dimensional space are related by a unitary rotation, and such a rotation permutes the valley Pauli matrices among themselves. A basis chosen for algebraic convenience rather than for compatibility with the bands it must reduce to will therefore assign the wrong symmetry character, and the wrong time-reversal parity, to each valley operator.

Getting this right has a practical payoff. Once the correct transformation law is known, one can ask which combinations of the fields available in a real device, the strain, the applied magnetic field, the built-in electrical asymmetry of a gate-defined dot, its shape, may appear multiplied by a valley operator in the effective Hamiltonian. Combinations that pass both the crystal test and the time-reversal test act on the valleys exactly as a magnetic field acts on a spin, and they reach operators that scalar potential scattering cannot. A contribution of that kind does not vanish when the germanium arrangement happens to be unfavourable, so it offers a route to lifting the low tail of the valley-splitting distribution rather than merely shifting its mean. How far that protection goes on a real wafer is a separate question. The same random germanium that scatters the electron also distorts the local electrostatics, so the in-plane asymmetry of a dot is itself partly random, and every valley-magnetic channel built from that asymmetry fluctuates with it. The same analysis also recovers a momentum-dependent term in the two-valley Hamiltonian of silicon that is of the same order as the ordinary kinetic terms yet is routinely omitted, and which ties the valley splitting of a small dot to the shape and orientation of its confining potential and, through the kinetic momentum, to any magnetic field threading the dot.

## Problem

Silicon is a poor host for a spin qubit in one specific respect: the conduction band has two nearly degenerate valley states, and unless they are split by an energy comfortably larger than the qubit operating scale, gate operations leak out of the spin subspace. The conventional account of that splitting is scattering from the scalar potential of randomly placed germanium atoms in the barrier alloy, whose matrix element between the two valley states is off-diagonal, so the splitting is broadly distributed across devices with weight close to zero and dots landing in that low tail are unusable. That account is incomplete, because it silently assumes the valley pair behaves like a spin, so that only the operators a scalar potential can reach are available at all.

Work at the X point of the Brillouin zone of the diamond structure, where the two valley states originate, using units in which the cubic lattice constant is 2*pi for the plane waves and the crystal operations. Establish how the three valley Pauli matrices transform under the crystal symmetry of that point and under time reversal, and use both to decide which combinations of the background fields of a real device may multiply which valley operator. The background fields are a polar vector P lying in the plane that describes the in-plane asymmetry of the dot, the vertical electric field Fz applied by the gates, the applied magnetic field B, the strain tensor eps, the ellipticity tensor Q of the dot, and the in-plane momentum entering through the product kx ky. Here Q is the traceless symmetric in-plane director tensor Q = n n^T - I/2, with n the unit vector along the dot's major axis, so it records the orientation of the dot and not how elongated it is. Test the following twenty-nine candidate combinations, in this index order:

   0  eps_xy                             15  Bz (eps_xx - eps_yy)
   1  eps_xx - eps_yy                    16  Bx eps_xz + By eps_yz
   2  eps_zz                             17  Bx eps_yz + By eps_xz
   3  eps_xx + eps_yy                    18  Bz eps_xy
   4  kx ky                              19  Q_xy
   5  Px By - Py Bx                      20  (Px Bx - Py By) Q_xy
   6  Px Bx - Py By                      21  (Px By + Py Bx)(Q_xx - Q_yy)
   7  Px By + Py Bx                      22  Px Py
   8  Px Bx + Py By                      23  Bx By
   9  (Px By - Py Bx) eps_zz             24  Fz
  10  (Px By - Py Bx)(eps_xx + eps_yy)   25  Fz eps_xy
  11  (Px Bx - Py By) eps_xy             26  Px eps_yz + Py eps_xz
  12  (Px By + Py Bx)(eps_xx - eps_yy)   27  Px eps_xz + Py eps_yz
  13  Bz (Px By + Py Bx)                 28  Bz (Bx^2 - By^2)
  14  Bz (Px Bx + Py By)

Index the valley operators 0 to 3, with 0 the identity, which carries sign +1 under every operation and is even under time reversal.

The device is a gate-defined dot in a strained silicon quantum well grown along z, with crystal axes x and y in the plane. Treat the electron as strictly two-dimensional: it is confined harmonically in the plane and moves in the out-of-plane component of the magnetic field, while the in-plane component acts only through the candidate combinations. The momentum in kx ky is measured in ordinary units, not in the plane-wave units above, and the product is expressed in units of one over the square of the cubic lattice constant, so the number that multiplies its coupling constant is the dimensionless expectation of kx ky times a^2 in the dot ground state. Use:
- dot confinement energies 1.10 meV along the major axis and 1.45 meV along the minor axis, major axis at 30.0 degrees to the crystal x axis
- in-plane effective mass 0.19 m_e, cubic lattice constant 5.431 angstrom
- magnetic field of 1.50 T in the plane at 75.0 degrees to the crystal x axis, plus an out-of-plane component of 0.20 T
- built-in polar vector of magnitude 1.00 at 45.0 degrees to the crystal x axis
- vertical electric field Fz = 6.0, in megavolts per metre
- strain eps_xx = 3.0e-3, eps_yy = 1.8e-3, eps_zz = -1.2e-3, eps_xy = 6.0e-4, eps_xz = 2.0e-4, eps_yz = 1.0e-4
- coupling constants, one per candidate term in the index order above, in microelectron volts per unit of that term (magnetic fields in tesla, Fz in megavolts per metre):
  4.20e4, 3.10e4, 2.50e3, 1.90e3, 1.80e5, 5.60e1, 4.40e1, 3.90e1, 2.80e1, 7.30e3, 6.10e3, 8.80e3, 9.40e3, 2.15e1, 1.75e1, 6.00e4, 1.40e3, 1.10e3, 9.00e2, 1.30e2, 4.70e1, 3.50e1, 3.00e1, 4.00e1, 1.20e1, 1.20e4, 1.50e5, 1.20e5, 6.00e1

Each allowed term contributes its coupling constant times its value to the component of the valley-magnetic field belonging to the valley operator it may couple to, and terms allowed only with the identity shift both valleys together and do not contribute. The splitting of a device is twice the length of the vector of the three valley components of its traceless two-by-two valley Hamiltonian.

Now consider a wafer of nominally identical dots that differ only in where the germanium atoms happen to sit. The germanium matrix element is a circular complex Gaussian of zero mean whose two real components are independent with standard deviation 90.0 microelectron volts, and it enters only those valley operators a real scalar potential is able to reach. The same random arrangement also gives every dot its own in-plane asymmetry: the polar vector of a dot is the built-in vector above plus a random in-plane vector whose x and y components are independent Gaussians of zero mean and standard deviation 0.30 each, drawn independently of the germanium matrix element. Every other field is the same in every dot.

In your reasoning give the time-reversal parity of each valley Pauli matrix in index order. Give the number of candidates admitted for each valley operator by the crystal symmetry alone, the number that survive once time reversal is imposed as well, and name the candidates that time reversal removes. For a dot whose polar vector equals the built-in one, give each component of the valley-magnetic field in microelectron volts and the splitting that dot would show with no alloy disorder. Give the standard deviation across the wafer of each valley-magnetic component carried by an operator that is odd under time reversal, and the percentage of devices that would fall below 240.0 microelectron volts if every dot had exactly the built-in polar vector. Also quote three published results from the wider literature on valley splitting in Si/SiGe: the valley splitting that tight-binding simulations obtained by combining a shear strain eps_xy of 0.15 percent with a long-period germanium concentration oscillation of 2.5 percent average germanium; the correlation length of the valley splitting measured continuously under a single gate along the channel of an industrially fabricated device with a 2.8 percent germanium quantum well, together with the mean valley splitting measured across that channel; and the share of devices that atom-probe based simulations predicted would exceed 100 microelectron volts once 5 percent germanium is added to the quantum well. Your final answer must be a single number: the percentage of devices on the wafer whose valley splitting falls below 240.0 microelectron volts.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Free-electron states on the conduction-band shell at X

Goal
----
Enumerate the plane-wave triples of the free-electron states at the X point of the diamond structure that lie on a requested dimensionless energy shell, sorted lexicographically.

```python
def plane_wave_shell_at_X(eps_shell: int, n_max: int) -> np.ndarray:
    '''Plane-wave triples of the free-electron states at the X point of the diamond structure.

    Parameters
    ----------
    eps_shell : int
        Dimensionless free-electron energy p . p of the requested shell; positive integer.
    n_max : int
        Search range for each component of the reciprocal-lattice index triple n, which is
        scanned over -n_max .. n_max; positive integer.

    Returns
    -------
    result : np.ndarray
        Real array of length 3*M holding the M admissible triples p flattened row-major, the
        rows first sorted in ascending lexicographic order. M may be zero.

    Raises
    ------
    ValueError
        If eps_shell < 1 or n_max < 1.
    '''
    return result  # placeholder
```

### Step 2

The group of the wave vector at the X point

Goal
----
Build the thirty-two operations of the group of the wave vector at X as point matrices with their translations, in a fixed deterministic order.

```python
def wave_vector_group_at_X(axis: int) -> np.ndarray:
    '''The thirty-two operations {R | t} of the group of the wave vector at the X point.

    Parameters
    ----------
    axis : int
        Cubic axis of the X point, 0 for x, 1 for y and 2 for z.

    Returns
    -------
    result : np.ndarray
        Real array of length 384. The first 288 entries are the thirty-two point matrices R
        flattened row-major in the sorted order described above; the remaining 96 entries are
        the matching translations t in units of pi, flattened row-major in the same order.

    Raises
    ------
    ValueError
        If axis is not 0, 1 or 2, or if the assembled complex does not close into thirty-two
        distinct operations.
    '''
    return result  # placeholder
```

### Step 3

The two Bloch functions that spawn the valleys

Goal
----
Project the plane-wave shell onto the X1 representation and impose compatibility with the bands running from X to the zone centre, returning the two valley basis functions.

```python
def valleyor_basis(shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    '''The two X1 basis functions expressed over the plane-wave shell.

    Parameters
    ----------
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array of plane-wave triples.
    group : np.ndarray
        Output of step 02: length-384 real array holding the thirty-two operations.

    Returns
    -------
    result : np.ndarray
        Real array of length 32: the real parts of the two-by-eight coefficient matrix flattened
        row-major, followed by its imaginary parts in the same order.

    Raises
    ------
    ValueError
        If the supplied shell does not contain exactly eight plane waves.
    '''
    return result  # placeholder
```

### Step 4

Matrix representation of the group on the valley doublet

Goal
----
Represent every operation of the group by a two-by-two matrix on the valley doublet.

```python
def valley_representation(basis: np.ndarray, shell: np.ndarray, group: np.ndarray) -> np.ndarray:
    '''Two-by-two matrices representing the group of the wave vector on the valley doublet.

    Parameters
    ----------
    basis : np.ndarray
        Output of step 03: length-32 real array packing the two-by-eight coefficient matrix.
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array.
    group : np.ndarray
        Output of step 02: length-384 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 256: the real parts of the thirty-two two-by-two matrices flattened
        in operation order and then row-major, followed by the imaginary parts in the same order.

    Raises
    ------
    ValueError
        If the shell, group and basis arrays do not have matching, well-formed shapes.
    '''
    return result  # placeholder
```

### Step 5

Crystal signs and time-reversal parities of the valley operators

Goal
----
Return the sign of each valley Pauli matrix under each crystal operation together with its time-reversal parity.

```python
def tau_symmetry_data(rep: np.ndarray, basis: np.ndarray, shell: np.ndarray) -> np.ndarray:
    '''Crystal signs and time-reversal parities of the three valley Pauli matrices.

    Parameters
    ----------
    rep : np.ndarray
        Output of step 04: length-256 real array packing the thirty-two representation matrices.
    basis : np.ndarray
        Output of step 03: length-32 real array packing the valley doublet.
    shell : np.ndarray
        Output of step 01 for the eight-fold shell: length-24 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 99. The first 96 entries are the signs s[j, g] for valley index
        j = 1, 2, 3 and operation g, flattened row-major with j varying slowest. The final three
        entries are the time-reversal parities of the three valley Pauli matrices in the same
        order. All entries are +1 or -1.

    Raises
    ------
    ValueError
        If some valley Pauli matrix is not mapped onto plus or minus itself, which signals that
        the supplied basis is not the X1 doublet.
    '''
    return result  # placeholder
```

### Step 6

Which background fields may couple to which valley operator

Goal
----
Decide, for each of twenty-nine candidate field combinations, which valley operators it may couple to under both the crystal symmetry and time reversal.

```python
def allowed_valley_channels(group: np.ndarray, tau_data: np.ndarray) -> np.ndarray:
    '''Table of which candidate field combinations may couple to which valley operator.

    Parameters
    ----------
    group : np.ndarray
        Output of step 02: length-384 real array holding the thirty-two operations.
    tau_data : np.ndarray
        Output of step 05: length-99 real array of crystal signs and time-reversal parities.

    Returns
    -------
    result : np.ndarray
        Real array of length 116 holding a twenty-nine by four table of ones and zeros, flattened
        row-major. Entry (k, j) is 1 when candidate term k may couple to valley operator j under
        both the crystal symmetry and time reversal, and 0 otherwise.

    Raises
    ------
    ValueError
        If the group array or the tau data does not have the expected length.
    '''
    return result  # placeholder
```

### Step 7

Kinetic momentum correlation and ellipticity tensor of the dot

Goal
----
Return the ground-state expectation of the symmetrised product of the in-plane kinetic momenta of the elliptical dot in a perpendicular magnetic field, and the ellipticity tensor of the dot.

```python
def dot_geometry(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float,
                 m_t_rel: float, a_ang: float, Bz_T: float) -> np.ndarray:
    '''Ground-state kinetic momentum correlation and ellipticity tensor of the elliptical dot.

    Parameters
    ----------
    hw_maj_meV : float
        Confinement energy along the major axis in millielectron volts; positive and not larger
        than hw_min_meV.
    hw_min_meV : float
        Confinement energy along the minor axis in millielectron volts; positive.
    alpha_deg : float
        Angle of the major axis to the crystal x axis, in degrees.
    m_t_rel : float
        In-plane effective mass in units of the free electron mass; positive.
    a_ang : float
        Cubic lattice constant in angstrom; positive.
    Bz_T : float
        Magnetic field component perpendicular to the plane of the dot, in tesla.

    Returns
    -------
    result : np.ndarray
        Real array of length 4 holding the ground-state expectation of (kx ky + ky kx)/2 for the
        kinetic momenta, times the squared lattice constant, converged to a relative precision of
        1e-10, then the xy, xx and yy entries of the ellipticity tensor.

    Raises
    ------
    ValueError
        If either confinement energy is not positive, if hw_maj_meV exceeds hw_min_meV, or if the
        effective mass or lattice constant is not positive.
    '''
    return result  # placeholder
```

### Step 8

The three components of the valley-magnetic field

Goal
----
Sum the allowed channels with their coupling constants into the three components of the valley-magnetic field of a dot with a given polar vector.

```python
def valley_magnetic_components(coeffs: np.ndarray, allowed: np.ndarray, P_vec: np.ndarray,
                               B_vec: np.ndarray, Fz_MV_per_m: float, strain: np.ndarray,
                               dot_geom: np.ndarray) -> np.ndarray:
    '''The three components of the valley-magnetic field, in microelectron volts.

    Parameters
    ----------
    coeffs : np.ndarray
        Length-29 real array of coupling constants, one per candidate term, in microelectron
        volts per unit of that term.
    allowed : np.ndarray
        Output of step 06: length-116 real array holding the twenty-nine by four table.
    P_vec : np.ndarray
        Length-3 real array holding the in-plane polar vector of the dot; its z entry is unused.
    B_vec : np.ndarray
        Length-3 real array holding the magnetic field in tesla.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three real symmetric strain tensor, dimensionless.
    dot_geom : np.ndarray
        Output of step 07: length-4 real array.

    Returns
    -------
    result : np.ndarray
        Real array of length 3 holding the components of the valley-magnetic field belonging to
        valley operators 1, 2 and 3, in microelectron volts.

    Raises
    ------
    ValueError
        If coeffs does not hold exactly twenty-nine entries.
    '''
    return result  # placeholder
```

### Step 9

Wafer statistics with a random polar vector

Goal
----
Average the alloy Rice law over the dot-to-dot spread of the polar vector and return the spread of the time-reversal-odd component together with the percentage of the wafer below a splitting threshold, without and with that spread.

```python
def wafer_valley_statistics(coeffs: np.ndarray, allowed: np.ndarray, tau_parities: np.ndarray,
                            P_vec: np.ndarray, B_vec: np.ndarray, Fz_MV_per_m: float,
                            strain: np.ndarray, dot_geom: np.ndarray, sigma_ge_ueV: float,
                            sigma_P: float, e_threshold_ueV: float) -> np.ndarray:
    '''Spread of the protected component and the low tail of the wafer with a random polar vector.

    Parameters
    ----------
    coeffs : np.ndarray
        Length-29 coupling constants, as in step 08.
    allowed : np.ndarray
        Output of step 06: length-116 allowed-channel table.
    tau_parities : np.ndarray
        Shape (3,), the time-reversal parity (+1 or -1) of valley operators 1, 2 and 3.
    P_vec : np.ndarray
        Length-3 built-in polar vector; only its in-plane entries are used and randomised.
    B_vec : np.ndarray
        Length-3 magnetic field in tesla.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three strain tensor.
    dot_geom : np.ndarray
        Output of step 07: length-4 real array.
    sigma_ge_ueV : float
        Standard deviation of each real component of the germanium matrix element, in
        microelectron volts; positive.
    sigma_P : float
        Standard deviation of each in-plane component of the random polar vector; not negative.
    e_threshold_ueV : float
        Splitting threshold in microelectron volts; positive.

    Returns
    -------
    result : np.ndarray
        Array of length 3: the standard deviation across the wafer of the valley-magnetic
        component on the time-reversal-odd operator, in microelectron volts; the percentage of
        dots below the threshold with the polar vector fixed at P_vec; and the percentage below
        the threshold with the random polar vector. Percentages are converged to a relative
        precision of 1e-9.

    Raises
    ------
    ValueError
        If sigma_ge_ueV is not positive, if sigma_P is negative, if e_threshold_ueV is not
        positive, if the parities do not single out exactly one time-reversal-odd operator, or if
        the component on that operator is not an affine function of the in-plane polar vector.
    '''
    return result  # placeholder
```

### Step 10

End to end, the fraction of the wafer below threshold

Goal
----
Run the whole pipeline and return the percentage of the wafer whose valley splitting falls below the threshold when both the germanium matrix element and the polar vector vary from dot to dot.

```python
def valley_low_tail_percent(hw_maj_meV: float, hw_min_meV: float, alpha_deg: float,
                            m_t_rel: float, a_ang: float, B_inplane_T: float,
                            theta_B_deg: float, Bz_T: float, P0: float, theta_P_deg: float,
                            Fz_MV_per_m: float, strain: np.ndarray, coeffs: np.ndarray,
                            sigma_ge_ueV: float, sigma_P: float, e_threshold_ueV: float) -> float:
    '''Percentage of the wafer whose valley splitting falls below a threshold.

    Parameters
    ----------
    hw_maj_meV : float
        Confinement energy along the dot major axis, in millielectron volts.
    hw_min_meV : float
        Confinement energy along the dot minor axis, in millielectron volts.
    alpha_deg : float
        Angle of the dot major axis to the crystal x axis, in degrees.
    m_t_rel : float
        In-plane effective mass in units of the free electron mass.
    a_ang : float
        Cubic lattice constant in angstrom.
    B_inplane_T : float
        Magnitude of the in-plane magnetic field, in tesla.
    theta_B_deg : float
        Angle of the in-plane magnetic field to the crystal x axis, in degrees.
    Bz_T : float
        Out-of-plane component of the magnetic field, in tesla.
    P0 : float
        Magnitude of the built-in polar vector.
    theta_P_deg : float
        Angle of the built-in polar vector to the crystal x axis, in degrees.
    Fz_MV_per_m : float
        Vertical electric field in megavolts per metre.
    strain : np.ndarray
        Three-by-three real symmetric strain tensor, dimensionless.
    coeffs : np.ndarray
        Length-29 real array of coupling constants in microelectron volts per unit of the
        corresponding candidate term, indexed as in step 06.
    sigma_ge_ueV : float
        Standard deviation of each real component of the germanium matrix element, in
        microelectron volts.
    sigma_P : float
        Standard deviation of each in-plane component of the random polar vector.
    e_threshold_ueV : float
        Splitting threshold in microelectron volts.

    Returns
    -------
    result : float
        The percentage of the wafer below the threshold, converged to a relative precision of
        1e-9.

    Raises
    ------
    ValueError
        Propagated from the earlier steps for invalid dot, coupling or wafer data.
    '''
    return 0.0  # placeholder
```
