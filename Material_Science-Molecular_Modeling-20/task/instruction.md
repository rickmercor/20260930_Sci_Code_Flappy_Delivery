# Material_Science-Molecular_Modeling-20

## Background

MOFBuilder builds a MOF from its topology, metal nodes, and organic linkers. It checks the connections between building units, assigns the connecting fragments, and adjusts their positions. Periodic images are included when a connection crosses a cell boundary.
After alignment, the node coordinates are propagated using the target connection lengths. The old and new positions are compared in fractional coordinates to evaluate the cell. Missing connections are terminated with cap fragments placed at the vacant sites. Here we use a small example with fixed fragment poses and six proposed cells.

## Problem

Compare six trial geometries using the construction rules in MOFBuilder.
Return the id with the lowest cell loss among candidates that pass every gate.
Use the paper for the topology rules, fragment-search radius, fine-alignment objective, target edge length, and fractional-coordinate cell objective.
The coordinates are synthetic benchmark inputs; the fragment poses and trial cells are fixed.
Derive the fragment-search distances from each candidate's attachment sites and marked anchors; use the resulting assignments for alignment and propagation, then place the rigid caps.
Compare corresponding unwrapped graph nodes under their respective cells, and use cap atoms only for the clearance test.
Report the image shifts, assignments, fine losses, target lengths, cap clearances, rejection reasons, and admissible cell losses, with the relevant paper sections or equations.
Choose the smallest id if losses are within 1e-12 of the global minimum.

Benchmark conventions (specified here, not attributed to the paper)

| Item | Definition |
|---|---|
| Units | Cartesian coordinates and lengths: angstrom; fractional coordinates and cell loss: dimensionless |
| Lattice | Cell vectors are matrix rows; x = f @ T |
| Periodic search | Central cell plus shifts in {-1,0,1}^3; exclude distances <= 1e-8; image ties within 1e-12 of the global minimum use lexicographic shift order |
| Tree | Edges are ordered (parent, child); start at node 0, preserving its Cartesian position; each child occurs once |
| Topology | edge_points are unwrapped E sites; perpendicular distance <= 1e-8 to an open segment; endpoint labels determine ditopic/multitopic acceptance |
| Reassignment | Globally sort eligible (distance, row, column) triples; accept only unused rows and columns; paper radius with +1e-8 tolerance; every edge needs an assignment |
| Fine alignment | One node atom group per edge, paired with its assigned fragment group; gate <= 0.20 + 1e-8 angstrom^2 |
| Propagation | Edge direction comes from its selected image in old_cell; linker length comes from the assigned fragment; const_length and offset_length are nonnegative lengths |
| Capping | Recenter marked atom on each propagated cap node; rotate marked-to-axis direction onto vacancy vector by shortest proper rotation; antiparallel convention in step 07 |
| Cap gate | Every non-marked cap atom must be at least 0.25 - 1e-8 angstrom from every propagated graph node |
| Cell comparison | Paper Eq. (5), graph nodes only; each coordinate set uses its own cell; no wrapping of fractional differences |
| Failure | Invalid topology, missing assignment, failed fine gate, or cap clash excludes a candidate; no surviving candidate raises ValueError |

Data schema

| Field | Meaning |
|---|---|
| frac_points, kinds, edges | Wrapped graph coordinates, endpoint labels (V or EC), ordered tree edges |
| edge_points, linker_type | Unwrapped E sites and ditopic/multitopic label |
| old_cell, new_cell | Original and trial 3x3 row-lattice matrices |
| connection_sites, fragment_sites | Cartesian attachment sites (edge order) and marked fragment anchors (fragment order), in a shared unwrapped assignment frame; use Euclidean site-to-anchor distance with no further periodic wrapping |
| node_groups, fragment_groups | Already oriented atom coordinates in local connection frames; group order follows edges/fragments; these local poses are separate from the assignment frame |
| linker_lengths | One length per fragment, selected by reassignment |
| const_length, offset_length | Bond constant and geometric offset used in target length |
| cap_fragment | One complete rigid template in Cartesian coordinates |
| marked_index, axis_index | Zero-based marked atom and orientation atom indices |
| cap_nodes, vacant_vectors | Graph node indices to cap and their Cartesian vacant directions |
| id | Unique integer candidate identifier |

COMMON contains the fields shared by every candidate; each candidate is an independent copy of COMMON with the listed overrides (whole-field replacement).

```json
{"COMMON":{"kinds":["V","V","V","V","V"],"edges":[[0,1],[1,2],[1,3],[3,4]],"linker_type":"ditopic","const_length":0.22,"offset_length":0.13,"marked_index":0,"axis_index":1,"node_groups":[[[0,0.0,-0.0],[0.8,0.5,1.1]],[[3,0.4,-0.25],[3.8,0.9,0.85]],[[6,0.8,-0.5],[6.8,1.3,0.6]],[[9,1.2,-0.75],[9.8,1.7,0.35]]],"connection_sites":[[0.4,-0.3,0.2],[1.264,0.852,-1.72],[-0.164,0.948,-0.68],[-1.136,2.652,-1.72]]},"CANDIDATE_OVERRIDES":[{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.76,0.84,0.26],[0.17,0.78,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.0,9,0],[1,1.5,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[2,4],"vacant_vectors":[[1.0,2,-1],[-2,1.0,3]],"id":301,"edge_points":[[0.27,1.89,1.2],[-1.775,0.5625,1.8],[0.015,-0.0075,0.2],[2.23,-0.165,-1.6]],"fragment_groups":[[[3.06,0.479,-0.29],[5,1.8,1.45]],[[0.06,0.07,-0.04],[2,1.4,1.7]],[[9.06,1.297,-0.79],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[6.06,0.888,-0.54],[8,2.2,1.2]]],"linker_lengths":[2.937107463786,2.223119577181,4.135186947006,2.5,4.998384690254],"new_cell":[[10.25,0.0,0.0],[2.29,9.225,0.0],[1.025,1.3475,8.2]],"fragment_sites":[[0.904,0.372,-0.92],[0.796,0.228,-0.68],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[1.42,3.06,1.3]]},{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.768,0.84,0.26],[0.17,0.774,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.12,9,0],[1,1.43,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[2,4],"vacant_vectors":[[1.2,2,-1],[-2,1.15,3]],"id":302,"edge_points":[[0.2922,1.8795,1.2],[-1.732,0.54675,1.8],[0.00804,-0.03625,0.2],[2.22544,-0.178,-1.6]],"fragment_groups":[[[3.068,0.479,-0.295],[5,1.8,1.45]],[[6.068,0.888,-0.545],[8,2.2,1.2]],[[9.068,1.297,-0.795],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[0.068,0.07,-0.045],[2,1.4,1.7]]],"linker_lengths":[2.998543828316,5.10868487877,4.319645739728,2.5,2.273010006324],"new_cell":[[10.45,0.0,0.0],[2.0354,9.405,0.0],[1.045,1.77435,8.36]],"fragment_sites":[[0.904,0.372,-0.92],[0.412,1.716,0.04],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[-0.014,-0.852,1.12]]},{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.776,0.84,0.26],[0.17,0.768,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.24,9,0],[1,1.36,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[2,4],"vacant_vectors":[[1.4,2,-1],[-2,1.3,3]],"id":303,"edge_points":[[0.3144,1.869,1.2],[0.00036,-0.065,0.2],[2.22016,-0.191,-1.6]],"fragment_groups":[[[3.076,0.479,-0.3],[5,1.8,1.45]],[[0.076,0.07,-0.05],[2,1.4,1.7]],[[9.076,1.297,-0.8],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[6.076,0.888,-0.55],[8,2.2,1.2]]],"linker_lengths":[2.901107433026,2.194917675198,4.288482533797,2.5,4.969239045448],"new_cell":[[10.2,0.0,0.0],[2.5948,9.18,0.0],[1.02,1.2172,8.16]],"fragment_sites":[[0.904,0.372,-0.92],[0.796,0.228,-0.68],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[1.42,3.06,1.3]]},{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.784,0.84,0.26],[0.17,0.762,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.36,9,0],[1,1.29,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[2,4],"vacant_vectors":[[1.6,2,-1],[-2,1.45,3]],"id":304,"edge_points":[[0.3366,1.8585,1.2],[-1.646,0.51525,1.8],[-0.00804,-0.09375,0.2],[1.03416,-4.704,-1.6]],"fragment_groups":[[[3.084,0.479,-0.305],[5,1.8,1.45]],[[0.084,0.07,-0.055],[2,1.4,1.7]],[[9.084,1.297,-0.805],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[6.65,1.2,-0.15],[8,2.2,1.2]]],"linker_lengths":[3.033387164331,2.301210674389,4.450662257671,2.5,5.191515004727],"new_cell":[[10.6,0.0,0.0],[2.6316,9.54,0.0],[1.06,1.7174,8.48]],"fragment_sites":[[0.904,0.372,-0.92],[0.796,0.228,-0.68],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[1.42,3.06,1.3]]},{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.792,0.84,0.26],[0.17,0.756,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.48,9,0],[1,1.22,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[2,4],"vacant_vectors":[[1.8,2,-1],[-2,1.6,3]],"id":305,"edge_points":[[0.3588,1.848,1.2],[-1.603,0.4995,1.8],[-0.01716,-0.1225,0.2],[0.96744,-4.717,-1.6]],"fragment_groups":[[[3.092,0.479,-0.31],[5,1.8,1.45]],[[6.092,0.888,-0.56],[8,2.2,1.2]],[[9.092,1.297,-0.81],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[0.092,0.07,-0.06],[2,1.4,1.7]]],"linker_lengths":[2.936965688906,5.053015357504,4.265410873815,2.5,2.223372434083],"new_cell":[[10.35,0.0,0.0],[2.6468,9.315,0.0],[1.035,1.2227,8.28]],"fragment_sites":[[0.904,0.372,-0.92],[0.412,1.716,0.04],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[-0.014,-0.852,1.12]]},{"frac_points":[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.8,0.84,0.26],[0.17,0.75,0.86],[0.31,0.25,0.74]],"old_cell":[[10,0,0],[2.6,9,0],[1,1.15,8]],"cap_fragment":[[0.4,-0.2,0.7],[0.4,-0.2,5.458444053138],[1.0,-0.2,0.7],[0.4,0.6,0.7]],"cap_nodes":[4],"vacant_vectors":[[0.02,4.638,0.96]],"id":306,"edge_points":[[0.381,1.8375,1.2],[-1.56,0.48375,1.8],[-0.027,-0.15125,0.2],[0.9,-4.73,-1.6]],"fragment_groups":[[[3.1,0.479,-0.315],[5,1.8,1.45]],[[0.1,0.07,-0.065],[2,1.4,1.7]],[[9.1,1.297,-0.815],[11,2.6,0.95]],[[23,20,20],[24,20,20]],[[6.1,0.888,-0.565],[8,2.2,1.2]]],"linker_lengths":[2.911426511481,2.202236612959,4.178444053138,2.5,5.026221637836],"new_cell":[[10.3,0.0,0.0],[2.698,9.27,0.0],[1.03,1.1945,8.24]],"fragment_sites":[[0.904,0.372,-0.92],[0.796,0.228,-0.68],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[1.42,3.06,1.3]]}]}
```

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one integer candidate id inside <final_answer>...</final_answer>.
Rules:
- The tags are required. Do not omit them or leave them empty.
- Put only that one integer between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, or full candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_validate_topology

Goal
----
Check the candidate connections against the topology rule for the given linker type. Return the indices of connections with the required endpoint labels and an E site inside the segment.

```python
def validate_topology(points: list, kinds: list, edge_points: list, candidate_edges: list, linker_type: str) -> list[int]:
    """Return valid connection indices.

    Parameters
    ----------
    points : list
        N finite Cartesian 3-vectors in Å.
    kinds : list
        N labels, each V or EC.
    edge_points : list
        Finite Cartesian E-site 3-vectors.
    candidate_edges : list
        Distinct oriented index pairs (i, j).
    linker_type : str
        'ditopic' requires V at both ends.
        'multitopic' requires V followed by EC.
        An E site must lie within 1e-8 Å of the segment, with its
        projection strictly between the endpoints.
        Count each edge once, regardless of the number of matching E sites.

    Returns
    -------
    list[int]
        Valid edge indices in input order; [] if none pass.

    Raises
    ------
    ValueError
        Unsupported linker type, inconsistent labels, malformed or
        nonfinite points, duplicate edges, invalid indices, or
        endpoints separated by at most 1e-8 Å.
    """
    return []
```

### Step 2

02_periodic_neighbor

Goal
----
Find the closest positive-distance periodic image of the second point relative to the first. Return its shift and Cartesian distance.

```python
def periodic_neighbor(a_frac: list, b_frac: list, cell: list) -> tuple:
    """Return the closest positive-distance image.

    Parameters
    ----------
    a_frac, b_frac : list
        Finite fractional 3-vectors in [0, 1).
    cell : list
        Finite nonsingular 3x3 row-lattice matrix in Å.
        Search shifts in {-1, 0, 1}^3.
        Ignore distances <= 1e-8 Å.
        Among distances within 1e-12 Å of the global minimum,
        choose the lexicographically smallest shift.

    Returns
    -------
    tuple
        ((sx, sy, sz), distance).

    Raises
    ------
    ValueError
        Invalid coordinates or cell, abs(det(cell)) <= 1e-12 Å^3,
        or no positive-distance image.
    """
    return ((0, 0, 0), 0.0)
```

### Step 3

03_reassign_fragments

Goal
----
Assign compatible fragments to connection slots within the paper’s default search radius. A slot or fragment may be used only once.

```python
def reassign_fragments(distance_matrix: list) -> tuple:
    """Assign fragments without reusing columns.

    Parameters
    ----------
    distance_matrix : list
        Rectangular matrix of finite nonnegative distances in Å.
        Rows are connection slots; columns are compatible fragments.
        The search radius is 4.0 Å; accept distances <= 4.0 + 1e-8 Å.
        Sort eligible pairs by (distance, row, column).
        Accept a pair only if its row and column are unused.

    Returns
    -------
    tuple
        (assigned_columns, total_distance).
        Unassigned rows contain -1.
        Empty input returns ([], 0.0).
        Zero columns return ([-1] * number_of_rows, 0.0).

    Raises
    ------
    ValueError
        Ragged matrix, negative distance, or nonfinite distance.
    """
    return ([], 0.0)
```

### Step 4

04_fine_alignment_loss

Goal
----
Calculate the fine-alignment loss of the supplied connected atom groups using Eq. (3).

```python
def fine_alignment_loss(node_groups: list, linker_groups: list) -> float:
    """Evaluate the fine-alignment objective.

    Parameters
    ----------
    node_groups, linker_groups : list
        Equally long lists of nonempty atom groups.
        Each atom is a finite Cartesian 3-vector in Å.
        Entry e on each side forms one connected edge.
        Coordinates are in a common aligned frame.

    Returns
    -------
    float
        Sum of one minimum squared atom-pair distance per edge, in Å^2.
        Zero edges return 0.0.

    Raises
    ------
    ValueError
        Unequal edge counts, empty group, or malformed/nonfinite coordinates.
    """
    return 0.0
```

### Step 5

05_propagate_coordinate

Goal
----
Calculate the target edge length from Eq. (4) and propagate an endpoint from the supplied starting position.

```python
def propagate_coordinate(
    start: list,
    direction: list,
    linker_length: float,
    const_length: float,
    offset_length: float,
) -> tuple:
    """Return the target length and propagated endpoint.

    Parameters
    ----------
    start, direction : list
        Finite Cartesian 3-vectors.
        Normalize direction before propagation.
    linker_length, const_length, offset_length : float
        Finite nonnegative lengths in Å.
        target = linker_length + 2*const_length + 2*offset_length.
        endpoint = start + target * direction / norm(direction).

    Returns
    -------
    tuple
        (target_length, (end_x, end_y, end_z)), in Å.

    Raises
    ------
    ValueError
        Invalid vectors, direction norm <= 1e-8,
        negative/nonfinite lengths, or nonpositive total target length.
    """
    return (0.0, (0.0, 0.0, 0.0))
```

### Step 6

06_fractional_cell_loss

Goal
----
Calculate the cell loss for corresponding original and propagated graph nodes using Eq. (5).

```python
def fractional_cell_loss(old_cart: list, new_cart: list, old_cell: list, new_cell: list) -> float:
    """Evaluate the fractional-coordinate cell loss.

    Parameters
    ----------
    old_cart, new_cart : list
        Equally long lists of finite Cartesian 3-vectors in Å.
    old_cell, new_cell : list
        Finite nonsingular 3x3 row-lattice matrices in Å.
        Use x = f @ T.
        Compare the supplied unwrapped branches without wrapping
        fractional differences.

    Returns
    -------
    float
        Dimensionless sum of squared fractional-coordinate differences.
        Zero nodes return 0.0 after validating both cells.

    Raises
    ------
    ValueError
        Unequal node counts, malformed/nonfinite data,
        or abs(det(T)) <= 1e-12 Å^3 for either cell.
    """
    return 0.0
```

### Step 7

07_cap_defects

Goal
----
Place and orient the complete cap fragment at each vacant site.

```python
def cap_defects(fragment: list, marked_index: int, axis_index: int, sites: list, vacant_vectors: list) -> list:
    """Place a complete cap fragment at each site.

    Parameters
    ----------
    fragment : list
        Nonempty list of finite Cartesian 3-vectors.
    marked_index, axis_index : int
        Distinct valid atom indices.
        The source axis points from marked_index to axis_index.
    sites, vacant_vectors : list
        Equally long lists of finite Cartesian 3-vectors.
        Use the shortest proper rotation from source to vacancy axis.
        For antiparallel unit axes with cross norm <= 1e-12,
        choose the Cartesian basis least parallel to the source,
        breaking ties in x, y, z order. Normalize source cross basis
        and rotate by pi about that axis.

    Returns
    -------
    list
        One transformed atom-coordinate list per site.
        Preserve atom order. Empty sites return [] after validation.

    Raises
    ------
    ValueError
        Invalid coordinates or atom indices, unequal site/vector counts,
        source-axis norm <= 1e-8, or vacancy norm <= 1e-8.
    """
    return []
```

### Step 8

08_rank_candidates

Goal
----
Evaluate the candidates using the preceding seven steps. Return the id with the lowest cell loss among those that pass all construction checks.

```python
def rank_candidates(candidates: list) -> int:
    """Select the admissible candidate with minimum fractional-coordinate loss.

    Parameters
    ----------
    candidates : list of dict
        Each record has these required keys. Cartesian data are in angstrom.
        id: unique integer.
        frac_points: N wrapped 3-vectors, each component in [0,1), N >= 2.
        kinds: N endpoint labels V or EC.
        edges: N-1 ordered (parent, child) index pairs forming a tree rooted at 0.
        linker_type: 'ditopic' or 'multitopic'.
        edge_points: unwrapped Cartesian E-site vectors.
        old_cell, new_cell: nonsingular finite 3x3 row-lattice matrices, x=f@T.
        connection_sites: one Cartesian attachment-site 3-vector per edge.
        fragment_sites: one Cartesian marked-anchor 3-vector per fragment.
            Both site arrays share one already unwrapped assignment frame.
            Use ordinary Euclidean distances, with no periodic-image search
            on these two arrays. Build a fresh distance matrix per candidate:
            d[e][j] = norm(connection_sites[e] - fragment_sites[j]).
            Assign it with reassign_fragments; radius 4.0 + 1e-8.
        node_groups: one nonempty aligned atom group per edge.
        fragment_groups: one nonempty aligned atom group per fragment.
            These are supplied local alignment poses, distinct from the
            assignment frame. Fragment columns follow fragment_sites order.
        linker_lengths: one nonnegative length per fragment in the same order.
        const_length, offset_length: nonnegative length corrections.
        cap_fragment: rigid template of finite Cartesian atom vectors.
        marked_index, axis_index: indices of its marked and orientation atoms.
        cap_nodes, vacant_vectors: node indices to cap and Cartesian directions.

        Use periodic_neighbor on each graph edge in old_cell, unwrap from
        root 0, and retain the selected Cartesian edge vectors. Validate
        topology with validate_topology. Assign fragments using the freshly
        computed distances. Pair node_groups with assigned fragment_groups
        for fine_alignment_loss. Propagate from the fixed Cartesian root
        along normalized old edge vectors using assigned linker_lengths and
        target = linker_length + 2*const_length + 2*offset_length.
        Place caps at propagated cap_nodes using cap_defects.
        Reject invalid topology, an unassigned edge, fine loss > 0.20 + 1e-8,
        or any non-marked cap atom within 0.25 - 1e-8 of any propagated node.
        For survivors, use fractional_cell_loss on old and propagated nodes
        with their respective cells, without wrapping fractional differences.
        Losses within 1e-12 of the global minimum tie; choose the lowest id.
        Do not modify inputs. Use the seven preceding functions.

    Returns
    -------
    int
        Selected candidate id.

    Raises
    ------
    ValueError
        No surviving candidate, invalid ids/tree, inconsistent array sizes,
        malformed or nonfinite assignment sites, invalid cap node indices,
        or an evaluated earlier function receiving invalid numerical inputs.
        Required keys must exist. Geometric gate failures exclude candidates.
    """
    return 0
```
