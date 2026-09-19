import numpy as np
import networkx as nx
import open3d as o3d
from typing import List, Tuple, Dict, Any
from scipy.spatial import cKDTree
from error_function import calculate_score


def find_best_scored_pair2(pairs, normals):
    top_pairs = []
    top_scores = []

    for i, pair in enumerate(pairs):
        c1 = np.concatenate((np.array(pair[0]), np.array(normals[i][0])))
        c2 = np.concatenate((np.array(pair[1]), np.array(normals[i][1])))
        score = calculate_score(c1, c2)
        top_scores.append(score)
        top_pairs.append(pair)

    return top_pairs, top_scores




def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ap = p - a
    ab = b - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def fit_plane_from_k_nearest_lines(point: np.ndarray,
                                   segments: List[Tuple[int,int,np.ndarray,np.ndarray]],
                                   k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    dists = []
    for i, (u,v,pu,pv) in enumerate(segments):
        d = point_to_segment_distance(point, pu, pv)
        dists.append((d,i))
    dists.sort(key=lambda x: x[0])
    chosen = [segments[i] for _,i in dists[:min(k,len(dists))]]
    if not chosen:
        return None, None
    pts = []
    for seg in chosen:
        pts.append(seg[2])
        pts.append(seg[3])
    pts = np.vstack(pts)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    _, _, vh = np.linalg.svd(pts_centered)
    normal = vh[2]
    return centroid, normal

def make_sphere(center: np.ndarray, radius: float = 0.005, color=[0, 0, 1]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere

def make_connection(p1: np.ndarray, p2: np.ndarray, color=[0, 1, 0]):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([p1, p2])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set

def segment_node_distance(seg1, seg2) -> float:
    _, _, pu, pv = seg1
    _, _, qu, qv = seg2
    return min(
        np.linalg.norm(pu - qu),
        np.linalg.norm(pu - qv),
        np.linalg.norm(pv - qu),
        np.linalg.norm(pv - qv),
    )

def process_pairs_with_graph(
    all_pairs: List[Any],
    G: nx.Graph,
    line_set: o3d.geometry.LineSet,
    pcd_nodes: o3d.geometry.PointCloud,
    masked_pcd: o3d.geometry.PointCloud,
    max_hops: int = 3,
    parallel_threshold: float = 0.9,
    k_lines: int = 10,
    min_seg_dist: float = 0.02,
    extra_depth: int = 5,
    max_segs_per_side: int = 30,
    visualize: bool = False,
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:

    pts = np.asarray(pcd_nodes.points)
    masked_pts = np.asarray(masked_pcd.points)
    masked_normals = np.asarray(masked_pcd.normals)

    pts_tree = cKDTree(pts) if len(pts) else None
    masked_tree = cKDTree(masked_pts) if len(masked_pts) else None

    line_array = np.asarray(line_set.lines, dtype=int)
    segments = []
    for u, v in line_array:
        pu = pts[u].copy()
        pv = pts[v].copy()
        segments.append((int(u), int(v), pu, pv))

    node2segment_idxs: Dict[int, List[int]] = {i: [] for i in range(len(pts))}
    for idx, (u, v, pu, pv) in enumerate(segments):
        node2segment_idxs[u].append(idx)
        node2segment_idxs[v].append(idx)

    seg_plane_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    all_seg_count = len(segments)
    for idx, seg in enumerate(segments):
        u, v, pu, pv = seg
        cand_idxs = set(node2segment_idxs.get(u, []) + node2segment_idxs.get(v, []))
        if not cand_idxs:
            cand_list = [seg]
        else:
            cand_list = [segments[i] for i in cand_idxs]
        mid = (pu + pv) / 2.0
        c, n = fit_plane_from_k_nearest_lines(mid, cand_list, k=k_lines)
        seg_plane_cache[idx] = (c, n)

    def candidate_far_enough(candidate_idx: int, selected_idxs: List[int], min_dist: float) -> bool:
        if not selected_idxs:
            return True
        cand_pu = segments[candidate_idx][2]
        cand_pv = segments[candidate_idx][3]
        selected_endpoints = []
        for sidx in selected_idxs:
            selected_endpoints.append(segments[sidx][2])
            selected_endpoints.append(segments[sidx][3])
        selected_endpoints = np.vstack(selected_endpoints)  # (2*M,3)
        # compute distances from both cand endpoints to all selected endpoints
        d1 = np.linalg.norm(selected_endpoints - cand_pu[None, :], axis=1)
        d2 = np.linalg.norm(selected_endpoints - cand_pv[None, :], axis=1)
        if d1.min() < min_dist or d2.min() < min_dist:
            return False
        return True

    def collect_segments_with_spacing_idx(start_node: int, hop_limit: int, min_dist: float, cap: int):
        visited_nodes = {start_node}
        frontier_nodes = [start_node]
        selected_segment_idxs: List[int] = []
        hops = 0
        while frontier_nodes and hops < hop_limit and len(selected_segment_idxs) < cap:
            next_frontier = []
            candidate_idxs = []
            for node in frontier_nodes:
                candidate_idxs.extend(node2segment_idxs.get(node, []))
            for cidx in candidate_idxs:
                if cidx in selected_segment_idxs:
                    continue
                if candidate_far_enough(cidx, selected_segment_idxs, min_dist):
                    selected_segment_idxs.append(cidx)
                    if len(selected_segment_idxs) >= cap:
                        break
                u, v, pu, pv = segments[cidx]
                other = v if u in frontier_nodes else u
                if other not in visited_nodes:
                    visited_nodes.add(other)
                    next_frontier.append(other)
            frontier_nodes = next_frontier
            hops += 1
        return selected_segment_idxs

    def fast_closest_point(query_pt: np.ndarray, radius: float = 0.02):
        if masked_tree is None or len(masked_pts) == 0:
            return query_pt.copy(), np.array([0.0, 0.0, 1.0])

        _, idx = masked_tree.query(query_pt, k=1)
        idx = int(idx)
        return masked_pts[idx].copy(), masked_normals[idx].copy()

    results = []

    for p1, p2 in all_pairs:
        if pts_tree is None:
            continue
        _, n1_idx = pts_tree.query(p1, k=1)
        _, n2_idx = pts_tree.query(p2, k=1)
        n1_idx = int(n1_idx)
        n2_idx = int(n2_idx)

        d3_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        d3_normals: List[Tuple[np.ndarray, np.ndarray]] = []

        grasp_p1, normal_p1 = fast_closest_point(p1)
        grasp_p2, normal_p2 = fast_closest_point(p2)
        d3_pairs.append((grasp_p1, grasp_p2))
        d3_normals.append((normal_p1, normal_p2))

        if len(segments) != 0:
            seg_idxs1 = collect_segments_with_spacing_idx(n1_idx, max_hops, min_seg_dist, max_segs_per_side)
            if len(seg_idxs1) < 2 and max_hops < extra_depth:
                seg_idxs1 = collect_segments_with_spacing_idx(n1_idx, extra_depth, min_seg_dist, max_segs_per_side)

            seg_idxs2 = collect_segments_with_spacing_idx(n2_idx, max_hops, min_seg_dist, max_segs_per_side)
            if len(seg_idxs2) < 2 and max_hops < extra_depth:
                seg_idxs2 = collect_segments_with_spacing_idx(n2_idx, extra_depth, min_seg_dist, max_segs_per_side)

            if not seg_idxs1 or not seg_idxs2:
                continue

            for si in seg_idxs1:
                c1, n1 = seg_plane_cache.get(si, (None, None))
                if c1 is None:
                    continue
                for sj in seg_idxs2:
                    c2, n2 = seg_plane_cache.get(sj, (None, None))
                    if c2 is None:
                        continue

                    if parallel_threshold is not None and abs(np.dot(n1, n2)) < parallel_threshold:
                        continue
                    grasp1, n1_closest = fast_closest_point(c1)
                    grasp2, n2_closest = fast_closest_point(c2)
                    d3_pairs.append((grasp1, grasp2))
                    d3_normals.append((n1_closest, n2_closest))

        top_points, top_scores = find_best_scored_pair2(d3_pairs, d3_normals)
        results.extend([(tuple(pair), float(score)) for pair, score in zip(top_points, top_scores)])

        if visualize:
            geoms = []
            geoms.append(make_sphere(p1, radius=0.006, color=[1, 0, 0]))
            geoms.append(make_sphere(p2, radius=0.006, color=[1, 0, 0]))
            geoms.append(make_connection(p1, p2, color=[1, 0, 0]))

            for idx in seg_idxs1:
                _, _, pu, pv = segments[idx]
                geoms.append(make_connection(pu, pv, color=[0, 0, 1]))

            for idx in seg_idxs2:
                _, _, pu, pv = segments[idx]
                geoms.append(make_connection(pu, pv, color=[0, 1, 0]))

            for (gp1, gp2) in d3_pairs:
                geoms.append(make_sphere(gp1, radius=0.004, color=[1, 1, 0]))
                geoms.append(make_sphere(gp2, radius=0.004, color=[1, 1, 0]))
                geoms.append(make_connection(gp1, gp2, color=[1, 1, 0]))

            for (gp1, gp2) in top_points:
                geoms.append(make_sphere(gp1, radius=0.005, color=[1, 0, 1]))
                geoms.append(make_sphere(gp2, radius=0.005, color=[1, 0, 1]))
                geoms.append(make_connection(gp1, gp2, color=[1, 0, 1]))

            geoms.append(masked_pcd)
            geoms.append(pcd_nodes)

            o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

    return results




def connect_skel_slices_to_next(
    skels, 
    visualize=False, 
    node_color=[0.0, 0.8, 0.0], 
    line_color=[1.0, 0.0, 0.0], 
    intra_line_color=[0.0, 0.0, 1.0]
):

    processed = []
    for s in skels:
        if isinstance(s, o3d.geometry.PointCloud):
            pts = np.asarray(s.points)
        else:
            pts = np.asarray(s)
        if pts.size == 0:
            pts = pts.reshape((0, 3))
        elif pts.ndim == 1 and pts.size == 3:
            pts = pts.reshape((1, 3))
        processed.append(pts)

    lengths = [len(arr) for arr in processed]
    if sum(lengths) == 0:
        return nx.Graph(), o3d.geometry.LineSet(), o3d.geometry.PointCloud()

    offsets = np.cumsum([0] + lengths[:-1])
    global_points = np.vstack([arr for arr in processed if arr.shape[0] > 0])

    # build graph and add nodes
    G = nx.Graph()
    for slice_idx, arr in enumerate(processed):
        for local_idx, p in enumerate(arr):
            global_idx = int(offsets[slice_idx] + local_idx)
            G.add_node(global_idx, pos=tuple(p), slice=slice_idx)

    edges = []
    colors = []

    for i, arr in enumerate(processed):
        if arr.shape[0] <= 1:
            continue
        from scipy.spatial import distance_matrix
        D = distance_matrix(arr, arr)

        H = nx.Graph()
        for u in range(len(arr)):
            for v in range(u + 1, len(arr)):
                H.add_edge(u, v, weight=float(D[u, v]))

        T = nx.minimum_spanning_tree(H)

        for u, v, d in T.edges(data=True):
            gu = int(offsets[i] + u)
            gv = int(offsets[i] + v)
            G.add_edge(gu, gv, weight=d["weight"])
            edges.append([gu, gv])
            colors.append(intra_line_color)

    for i in range(len(processed) - 1):
        A = processed[i]
        if A.shape[0] == 0:
            continue
        for local_idx, p in enumerate(A):
            best_dist = np.inf
            best_target_global = None

            for j in range(i + 1, min(i + 4, len(processed))):
                B = processed[j]
                if B.shape[0] == 0:
                    continue

                pcd_B = o3d.geometry.PointCloud()
                pcd_B.points = o3d.utility.Vector3dVector(B)
                kdt = o3d.geometry.KDTreeFlann(pcd_B)

                k, idxs, dists = kdt.search_knn_vector_3d(p, 1)
                if k == 0:
                    continue

                target_local_idx = int(idxs[0])
                dist = np.linalg.norm(p - B[target_local_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_target_global = int(offsets[j] + target_local_idx)

            if best_target_global is not None:
                u = int(offsets[i] + local_idx)
                v = best_target_global
                G.add_edge(u, v, weight=float(best_dist))
                edges.append([u, v])
                colors.append(line_color)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(global_points)
    if len(edges) > 0:
        line_set.lines = o3d.utility.Vector2iVector(np.array(edges, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        line_set.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

    pcd_nodes = o3d.geometry.PointCloud()
    pcd_nodes.points = o3d.utility.Vector3dVector(global_points)
    pcd_nodes.paint_uniform_color(node_color)

    if visualize:
        o3d.visualization.draw_geometries([pcd_nodes, line_set])

    return G, line_set, pcd_nodes
