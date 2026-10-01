"""
Evaluate the candidates using the preceding seven steps. Return the id with the lowest cell loss among those that pass all construction checks.

Periodic images define the unwrapped graph and propagation directions. Fragment assignments determine the alignment groups and linker lengths. The propagated coordinates enter the cell calculation and define the capping sites. Candidates that fail a construction check are excluded before comparing cell losses.

Returns
-------
The integer id with the lowest admissible cell loss. Resolve ties within 1e-12 of the global minimum using the smallest id. Raise ValueError if no candidate passes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rank_candidates(candidates: list) -> int:
    import math

    ids = [candidate["id"] for candidate in candidates]
    if (
        any(type(i) is not int for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("invalid candidate ids")

    scored = []

    for candidate in candidates:
        frac = candidate["frac_points"]
        cell = candidate["old_cell"]
        edges = candidate["edges"]

        if len(frac) < 2 or len(edges) != len(frac) - 1:
            raise ValueError("tree size mismatch")

        if len(cell) != 3 or any(
            len(row) != 3
            or not all(math.isfinite(x) for x in row)
            for row in cell
        ):
            raise ValueError("invalid old cell")

        if any(
            len(point) != 3
            or any(
                not math.isfinite(x) or not 0 <= x < 1
                for x in point
            )
            for point in frac
        ):
            raise ValueError("invalid fractional coordinate")

        if (
            len(candidate["node_groups"]) != len(edges)
            or len(candidate["connection_sites"]) != len(edges)
            or len(candidate["fragment_sites"])
            != len(candidate["fragment_groups"])
            or len(candidate["linker_lengths"])
            != len(candidate["fragment_groups"])
        ):
            raise ValueError("assignment shape mismatch")

        if any(
            len(point) != 3
            or not all(math.isfinite(x) for x in point)
            for point in (
                candidate["connection_sites"]
                + candidate["fragment_sites"]
            )
        ):
            raise ValueError("invalid assignment geometry")

        distances = [
            [
                math.dist(site, anchor)
                for anchor in candidate["fragment_sites"]
            ]
            for site in candidate["connection_sites"]
        ]

        reached = {0}
        old = [None] * len(frac)
        old[0] = tuple(
            sum(frac[0][k] * cell[k][j] for k in range(3))
            for j in range(3)
        )
        directions = []

        for edge in edges:
            if len(edge) != 2 or any(
                type(i) is not int or i < 0 or i >= len(frac)
                for i in edge
            ):
                raise ValueError("invalid tree edge")

            i, j = edge
            if i not in reached or j in reached:
                raise ValueError("tree is not in propagation order")

            shift, _ = _oracle_periodic_neighbor(
                frac[i], frac[j], cell
            )

            fractional_delta = [
                frac[j][k] + shift[k] - frac[i][k]
                for k in range(3)
            ]
            direction = [
                sum(
                    fractional_delta[k] * cell[k][axis]
                    for k in range(3)
                )
                for axis in range(3)
            ]

            directions.append(direction)
            old[j] = tuple(
                old[i][k] + direction[k]
                for k in range(3)
            )
            reached.add(j)

        valid = _oracle_validate_topology(
            old,
            candidate["kinds"],
            candidate["edge_points"],
            edges,
            candidate["linker_type"],
        )
        if len(valid) != len(edges):
            continue

        assignment, _ = _oracle_reassign_fragments(distances)
        if any(index < 0 for index in assignment):
            continue

        assigned_groups = [
            candidate["fragment_groups"][index]
            for index in assignment
        ]
        fine_loss = _oracle_fine_alignment_loss(
            candidate["node_groups"], assigned_groups
        )
        if fine_loss > 0.20 + 1e-8:
            continue

        new = [None] * len(frac)
        new[0] = old[0]

        for edge_index, (i, j) in enumerate(edges):
            _, new[j] = _oracle_propagate_coordinate(
                new[i],
                directions[edge_index],
                candidate["linker_lengths"][assignment[edge_index]],
                candidate["const_length"],
                candidate["offset_length"],
            )

        if any(
            type(i) is not int or i < 0 or i >= len(new)
            for i in candidate["cap_nodes"]
        ):
            raise ValueError("invalid cap node")

        caps = _oracle_cap_defects(
            candidate["cap_fragment"],
            candidate["marked_index"],
            candidate["axis_index"],
            [new[i] for i in candidate["cap_nodes"]],
            candidate["vacant_vectors"],
        )

        if any(
            math.dist(point, node) < 0.25 - 1e-8
            for cap in caps
            for atom_index, point in enumerate(cap)
            if atom_index != candidate["marked_index"]
            for node in new
        ):
            continue

        cell_loss = _oracle_fractional_cell_loss(
            old, new, cell, candidate["new_cell"]
        )
        scored.append((cell_loss, candidate["id"]))

    if not scored:
        raise ValueError("no admissible candidate")

    minimum = min(loss for loss, candidate_id in scored)

    return min(
        candidate_id
        for loss, candidate_id in scored
        if loss <= minimum + 1e-12
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = "import copy,json\nv0=[[23,20,20],[24,20,20]]\nv1=[[0,1],[1,2],[1,3],[3,4]]\nv2=[[0,0.0,-0.0],[0.8,0.5,1.1]]\nv3=[[6,0.8,-0.5],[6.8,1.3,0.6]]\nv4=[[3,0.4,-0.25],[3.8,0.9,0.85]]\nv5=[[9,1.2,-0.75],[9.8,1.7,0.35]]\nv6=[[0.4,-0.2,0.7],[0.4,-0.2,1.7],[1.0,-0.2,0.7],[0.4,0.6,0.7]]\nv7=[[0.4,-0.3,0.2],[1.264,0.852,-1.72],[-0.164,0.948,-0.68],[-1.136,2.652,-1.72]]\nv8=[[0.904,0.372,-0.92],[0.796,0.228,-0.68],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[1.42,3.06,1.3]]\nv9=[[0.904,0.372,-0.92],[0.412,1.716,0.04],[-1.604,2.028,-0.68],[1.2,34.1,-3.8],[-0.014,-0.852,1.12]]\nv10=[v2,v4,v3,v5]\nPRIMARY=json.loads(json.dumps([{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.76,0.84,0.26],[0.17,0.78,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.0,9,0],[1,1.5,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':v6,'marked_index':0,'axis_index':1,'cap_nodes':[2,4],'vacant_vectors':[[1.0,2,-1],[-2,1.0,3]],'id':301,'edge_points':[[0.27,1.89,1.2],[-1.775,0.5625,1.8],[0.015,-0.0075,0.2],[2.23,-0.165,-1.6]],'node_groups':v10,'fragment_groups':[[[3.06,0.479,-0.29],[5,1.8,1.45]],[[0.06,0.07,-0.04],[2,1.4,1.7]],[[9.06,1.297,-0.79],[11,2.6,0.95]],v0,[[6.06,0.888,-0.54],[8,2.2,1.2]]],'linker_lengths':[2.937107463786,2.223119577181,4.135186947006,2.5,4.998384690254],'new_cell':[[10.25,0.0,0.0],[2.29,9.225,0.0],[1.025,1.3475,8.2]],'connection_sites':v7,'fragment_sites':v8},{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.768,0.84,0.26],[0.17,0.774,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.12,9,0],[1,1.43,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':v6,'marked_index':0,'axis_index':1,'cap_nodes':[2,4],'vacant_vectors':[[1.2,2,-1],[-2,1.15,3]],'id':302,'edge_points':[[0.2922,1.8795,1.2],[-1.732,0.54675,1.8],[0.00804,-0.03625,0.2],[2.22544,-0.178,-1.6]],'node_groups':v10,'fragment_groups':[[[3.068,0.479,-0.295],[5,1.8,1.45]],[[6.068,0.888,-0.545],[8,2.2,1.2]],[[9.068,1.297,-0.795],[11,2.6,0.95]],v0,[[0.068,0.07,-0.045],[2,1.4,1.7]]],'linker_lengths':[2.998543828316,5.10868487877,4.319645739728,2.5,2.273010006324],'new_cell':[[10.45,0.0,0.0],[2.0354,9.405,0.0],[1.045,1.77435,8.36]],'connection_sites':v7,'fragment_sites':v9},{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.776,0.84,0.26],[0.17,0.768,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.24,9,0],[1,1.36,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':v6,'marked_index':0,'axis_index':1,'cap_nodes':[2,4],'vacant_vectors':[[1.4,2,-1],[-2,1.3,3]],'id':303,'edge_points':[[0.3144,1.869,1.2],[0.00036,-0.065,0.2],[2.22016,-0.191,-1.6]],'node_groups':v10,'fragment_groups':[[[3.076,0.479,-0.3],[5,1.8,1.45]],[[0.076,0.07,-0.05],[2,1.4,1.7]],[[9.076,1.297,-0.8],[11,2.6,0.95]],v0,[[6.076,0.888,-0.55],[8,2.2,1.2]]],'linker_lengths':[2.901107433026,2.194917675198,4.288482533797,2.5,4.969239045448],'new_cell':[[10.2,0.0,0.0],[2.5948,9.18,0.0],[1.02,1.2172,8.16]],'connection_sites':v7,'fragment_sites':v8},{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.784,0.84,0.26],[0.17,0.762,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.36,9,0],[1,1.29,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':v6,'marked_index':0,'axis_index':1,'cap_nodes':[2,4],'vacant_vectors':[[1.6,2,-1],[-2,1.45,3]],'id':304,'edge_points':[[0.3366,1.8585,1.2],[-1.646,0.51525,1.8],[-0.00804,-0.09375,0.2],[1.03416,-4.704,-1.6]],'node_groups':v10,'fragment_groups':[[[3.084,0.479,-0.305],[5,1.8,1.45]],[[0.084,0.07,-0.055],[2,1.4,1.7]],[[9.084,1.297,-0.805],[11,2.6,0.95]],v0,[[6.65,1.2,-0.15],[8,2.2,1.2]]],'linker_lengths':[3.033387164331,2.301210674389,4.450662257671,2.5,5.191515004727],'new_cell':[[10.6,0.0,0.0],[2.6316,9.54,0.0],[1.06,1.7174,8.48]],'connection_sites':v7,'fragment_sites':v8},{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.792,0.84,0.26],[0.17,0.756,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.48,9,0],[1,1.22,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':v6,'marked_index':0,'axis_index':1,'cap_nodes':[2,4],'vacant_vectors':[[1.8,2,-1],[-2,1.6,3]],'id':305,'edge_points':[[0.3588,1.848,1.2],[-1.603,0.4995,1.8],[-0.01716,-0.1225,0.2],[0.96744,-4.717,-1.6]],'node_groups':v10,'fragment_groups':[[[3.092,0.479,-0.31],[5,1.8,1.45]],[[6.092,0.888,-0.56],[8,2.2,1.2]],[[9.092,1.297,-0.81],[11,2.6,0.95]],v0,[[0.092,0.07,-0.06],[2,1.4,1.7]]],'linker_lengths':[2.936965688906,5.053015357504,4.265410873815,2.5,2.223372434083],'new_cell':[[10.35,0.0,0.0],[2.6468,9.315,0.0],[1.035,1.2227,8.28]],'connection_sites':v7,'fragment_sites':v9},{'frac_points':[[0.12,0.16,0.11],[0.83,0.21,0.19],[0.8,0.84,0.26],[0.17,0.75,0.86],[0.31,0.25,0.74]],'kinds':['V','V','V','V','V'],'edges':v1,'linker_type':'ditopic','old_cell':[[10,0,0],[2.6,9,0],[1,1.15,8]],'const_length':0.22,'offset_length':0.13,'cap_fragment':[[0.4,-0.2,0.7],[0.4,-0.2,5.458444053138],[1.0,-0.2,0.7],[0.4,0.6,0.7]],'marked_index':0,'axis_index':1,'cap_nodes':[4],'vacant_vectors':[[0.02,4.638,0.96]],'id':306,'edge_points':[[0.381,1.8375,1.2],[-1.56,0.48375,1.8],[-0.027,-0.15125,0.2],[0.9,-4.73,-1.6]],'node_groups':v10,'fragment_groups':[[[3.1,0.479,-0.315],[5,1.8,1.45]],[[0.1,0.07,-0.065],[2,1.4,1.7]],[[9.1,1.297,-0.815],[11,2.6,0.95]],v0,[[6.1,0.888,-0.565],[8,2.2,1.2]]],'linker_lengths':[2.911426511481,2.202236612959,4.178444053138,2.5,5.026221637836],'new_cell':[[10.3,0.0,0.0],[2.698,9.27,0.0],[1.03,1.1945,8.24]],'connection_sites':v7,'fragment_sites':v8}]))\n"
    return [
        {"setup": base + '', "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + '', "call": 'rank_candidates(list(reversed(PRIMARY)))', "gold_call": '_oracle_rank_candidates(list(reversed(PRIMARY)))'},
        {"setup": base + '', "call": 'rank_candidates(PRIMARY[:4])', "gold_call": '_oracle_rank_candidates(PRIMARY[:4])'},
        {"setup": base + "PRIMARY[5]['vacant_vectors']=[[-x for x in PRIMARY[5]['vacant_vectors'][0]]]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['fragment_sites']=[[100+x,100+y,100+z] for x,y,z in PRIMARY[4]['fragment_sites']]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['connection_sites']=[]\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['fragment_sites'][0]=[0,0]\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['fragment_sites'][0][0]=float('nan')\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['fragment_sites'].pop()\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['id']=301\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['edges']=[[1,2],[0,1],[1,3],[3,4]]\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['edge_points']=[]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['fragment_groups']=[[[50,50,50]] for _ in range(5)]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['cap_fragment']=[[0,0,0],[0,0,0.1],[0.5,0,0]]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['new_cell']=[[5,0,0],[0,5,0],[0,0,5]]\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + '\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', "call": 'check_error(lambda: rank_candidates([PRIMARY[2],PRIMARY[3],PRIMARY[5]]))', "gold_call": 'check_error(lambda: _oracle_rank_candidates([PRIMARY[2],PRIMARY[3],PRIMARY[5]]))'},
        {"setup": base + '\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', "call": 'check_error(lambda: rank_candidates([]))', "gold_call": 'check_error(lambda: _oracle_rank_candidates([]))'},
        {"setup": base + "PRIMARY[5]['cap_fragment'][1][2]-=0.13\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[5]['cap_fragment'][1][2]-=0.12999998\n", "call": 'rank_candidates(PRIMARY)', "gold_call": '_oracle_rank_candidates(PRIMARY)'},
        {"setup": base + "PRIMARY[4]['connection_sites'][0][0]=float('inf')\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
        {"setup": base + "PRIMARY[4]['cap_nodes']=[99]\n\ndef check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n", "call": 'check_error(lambda: rank_candidates(PRIMARY))', "gold_call": 'check_error(lambda: _oracle_rank_candidates(PRIMARY))'},
    ]
