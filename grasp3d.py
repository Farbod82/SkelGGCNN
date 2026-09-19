
import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from collision_detector import ModelFreeCollisionDetector
from g_utils import connect_skel_slices_to_next, process_pairs_with_graph
from graspnetAPI.grasp import GraspGroup
from keypoint_generator import KeypointGeneration
from laplacian_contraction import skeleton3D


def cross_product(a, b):
    return np.cross(a, b)


def rotation_matrix_to_align_vectors(start, end, angle):
    axis = np.subtract(end, start)
    axis_normalized = axis / np.linalg.norm(axis)
    return Rotation.from_rotvec(axis_normalized * np.deg2rad(angle))


def vector_to_quaternion(v, perp_vector):
    v_norm = v / np.linalg.norm(v)
    reference_vector = np.array([-1, 0, 0])
    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    if np.allclose(v_norm, -reference_vector):
        return np.array([0, 0, 0, -1])
    rotation_axis = cross_product(reference_vector, v_norm)
    theta = np.arccos(np.dot(reference_vector, v_norm))
    q1 = Rotation.from_rotvec(theta * rotation_axis)
    if perp_vector is not None:
        perp_vector /= np.linalg.norm(perp_vector)
        pitch_angle = np.arcsin(np.dot(perp_vector, reference_vector))
        q2 = Rotation.from_rotvec(pitch_angle * reference_vector)
        return (q1 * q2).as_quat()
    return q1.as_quat()


def contact_pair_pose(point1: np.ndarray, point2: np.ndarray, gripper_position: np.ndarray):
    midpoint = np.add(point1, point2) / 2
    normal_plane = cross_product(
        np.subtract(point1, gripper_position),
        np.subtract(point2, gripper_position),
    )
    normal_plane = normal_plane / np.linalg.norm(normal_plane)
    normal_plane = rotation_matrix_to_align_vectors(point1, point2, -90).apply(normal_plane)
    return normal_plane * -1, midpoint


@dataclass
class GraspResult:
    contacts: np.ndarray
    scores: np.ndarray
    poses: np.ndarray
    grasps: GraspGroup
    candidates: int
    rejected: int
    rejected_grasps: GraspGroup
    finger_width: float
    base_offset: float
    grasp_height: float
    grasp_depth: float
    rotation_degrees: np.ndarray
    width_expansions: np.ndarray
    selected_colliding: np.ndarray
    orientations_tested: int


def validate_point_cloud(point_cloud):
    """Require object XYZ and oriented unit normals, in metres."""
    cloud = np.asarray(point_cloud, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 6 or len(cloud) == 0:
        raise ValueError("Expected a nonempty (N, 6) array: x, y, z, nx, ny, nz")
    if not np.isfinite(cloud).all():
        raise ValueError("Point cloud contains NaN or infinity")
    lengths = np.linalg.norm(cloud[:, 3:], axis=1)
    if not np.allclose(lengths, 1.0, atol=0.05):
        raise ValueError("Normals must be oriented unit vectors, as in get_point_cloud(sim)")
    return cloud


def load_mesh_point_cloud(path, number_of_points=10000, seed=42):
    """Sample the mesh surface into an Open3D cloud without coordinate transforms."""
    if number_of_points < 31:
        raise ValueError("Surface sampling requires at least 31 points for normal orientation")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise ValueError(f"Could not load a triangle mesh from {path}")
    o3d.utility.random.seed(seed)
    # Triangle normals avoid interpolating cancelled normals in double-sided
    # OBJ files. Consistent orientation handles their opposite face winding.
    geometry = mesh.sample_points_uniformly(
        number_of_points=number_of_points, use_triangle_normal=True,
    )
    if np.any(np.linalg.norm(np.asarray(geometry.normals), axis=1) < 1e-8):
        geometry.estimate_normals()
    geometry.orient_normals_consistent_tangent_plane(30)
    return validate_point_cloud(np.column_stack((geometry.points, geometry.normals)))


def generate_contact_candidates(point_cloud):
    """Generate graph-refined contacts from overlapping point-cloud sections."""
    generator = KeypointGeneration(debugMode=False)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    cloud.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    cloud = cloud.voxel_down_sample(0.005)
    points = np.asarray(cloud.points)
    if len(points) < 4:
        return []

    slices, normals, _ = generator.split_point_cloud_by_height_ovelap(
        points, np.asarray(cloud.normals), step_height=0.025, overlap=0.8
    )
    skeletons = []
    for section, section_normals in zip(slices, normals):
        if len(section) < 41:
            continue
        section_cloud = o3d.geometry.PointCloud()
        section_cloud.points = o3d.utility.Vector3dVector(section)
        section_cloud.normals = o3d.utility.Vector3dVector(section_normals)
        contracted = skeleton3D(section_cloud).extract_skeleton()
        skeleton_cloud = o3d.geometry.PointCloud()
        skeleton_cloud.points = o3d.utility.Vector3dVector(contracted)
        skeletons.append(skeleton_cloud.voxel_down_sample(0.005))

    scored = []
    for section, section_normals in zip(slices, normals):
        if len(section) < 4:
            continue
        section_cloud = o3d.geometry.PointCloud()
        section_cloud.points = o3d.utility.Vector3dVector(section)
        section_cloud.normals = o3d.utility.Vector3dVector(section_normals)
        _, section_heights = generator.split_point_cloud_by_height(
            section, step_height=0.02
        )
        pairs, scores, _ = generator.generate_scored_grasp_poses(
            section_heights, section, section_cloud
        )
        scored.extend((float(score), pair) for pair, score in zip(pairs, scores))

    seeds = heapq.nlargest(10, scored, key=lambda item: item[0])
    if not seeds or not skeletons:
        return []
    graph, lines, nodes = connect_skel_slices_to_next(skeletons, visualize=False)
    refined = process_pairs_with_graph(
        all_pairs=[pair for _, pair in seeds], G=graph, line_set=lines,
        pcd_nodes=nodes, masked_pcd=cloud, max_hops=5, k_lines=5, visualize=False,
    )
    return [(np.asarray(pair, dtype=np.float64), float(score))
            for pair, score in refined if score != 0]


def contacts_to_grasp(pair, score, gripper_position, height, depth,
                      rotation_degrees=0.0, width_expansion=0.0):
    """Create a midpoint/quaternion pose and a detector-frame grasp.

    Detector columns are approach X, closing Y, and height Z. Translation is
    offset from the contact midpoint so X=depth lies at the fingertips. The
    midpoint/quaternion and detector poses use different conventions and are
    intentionally not interchangeable.
    """
    pair = np.asarray(pair, dtype=np.float64)
    axis = pair[1] - pair[0]
    contact_width = np.linalg.norm(axis)
    width = contact_width + width_expansion
    if not np.isfinite(pair).all() or contact_width < 1e-8 or width_expansion < 0:
        return None
    axis /= contact_width
    plane_normal = np.cross(pair[0] - gripper_position, pair[1] - gripper_position)
    if np.linalg.norm(plane_normal) < 1e-8:
        return None
    direction, midpoint = contact_pair_pose(pair[0], pair[1], np.asarray(gripper_position))
    if not np.isfinite(direction).all():
        return None
    direction /= np.linalg.norm(direction)
    height_axis = np.cross(direction, axis)
    if np.linalg.norm(height_axis) < 1e-8:
        return None
    height_axis /= np.linalg.norm(height_axis)
    direction = np.cross(axis, height_axis)
    base_direction = direction.copy()
    base_quaternion = vector_to_quaternion(
        base_direction.copy(), (pair[0] - pair[1]).copy()
    )
    axis_rotation = Rotation.from_rotvec(np.deg2rad(rotation_degrees) * axis)
    direction = axis_rotation.apply(direction)
    height_axis = axis_rotation.apply(height_axis)
    rotation = np.column_stack((direction, axis, height_axis))
    quaternion = (axis_rotation * Rotation.from_quat(base_quaternion)).as_quat()
    row = np.concatenate((
        [score, width, height, depth], rotation.reshape(-1), midpoint - depth * direction, [-1],
    ))
    return row, np.concatenate((midpoint, quaternion))


def filter_candidates(candidates, scene_points, gripper_position, height=0.02,
                      depth=0.03, empty_thresh=10, num_grasps=5, collision_batch_size=32,
                      finger_width=0.02, base_offset=0.012,
                      rotation_step_degrees=15.0, orientation_seed=None,
                      max_width_expansion=0.02, width_step=0.002):
    if not 0 < rotation_step_degrees <= 360:
        raise ValueError("rotation_step_degrees must be in (0, 360]")
    if max_width_expansion < 0 or width_step <= 0:
        raise ValueError("max_width_expansion must be nonnegative and width_step positive")
    rows, poses, contacts, rotations, expansions, pair_indices = [], [], [], [], [], []
    rotation_samples = np.arange(0.0, 360.0, rotation_step_degrees)
    width_samples = np.arange(
        0.0, max_width_expansion + width_step * 0.5, width_step,
    )
    for pair_index, (pair, score) in enumerate(candidates):
        for width_expansion in width_samples:
            for angle in rotation_samples:
                converted = contacts_to_grasp(
                    pair, score, gripper_position, height, depth, angle,
                    width_expansion,
                )
                if converted is None:
                    continue
                row, pose = converted
                rows.append(row)
                poses.append(pose)
                contacts.append(pair)
                rotations.append(angle)
                expansions.append(width_expansion)
                pair_indices.append(pair_index)
    group = GraspGroup(np.asarray(rows, dtype=np.float64).reshape(-1, 17))
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    contacts = np.asarray(contacts, dtype=np.float64).reshape(-1, 2, 3)
    rotations = np.asarray(rotations, dtype=np.float64)
    expansions = np.asarray(expansions, dtype=np.float64)
    pair_indices = np.asarray(pair_indices, dtype=np.int64)
    detector = ModelFreeCollisionDetector(
        scene_points, finger_width=finger_width, depth_base=base_offset
    )
    rejected = np.zeros(len(group), dtype=bool)
    # Bound the detector's (grasps, points, 3) temporary arrays without changing
    # its collision rules or downsampling the collision cloud.
    for start in range(0, len(group), collision_batch_size):
        stop = start + collision_batch_size
        rejected[start:stop] = detector.detect(
            group[start:stop], return_empty_grasp=True, empty_thresh=empty_thresh,
        )
    # Retain exactly one randomly selected orientation for every contact pair.
    # Prefer the collision-free/nonempty pool. If it is empty, deliberately
    # retain one colliding/empty fallback so the contact pair is not discarded.
    rng = np.random.default_rng(orientation_seed)
    selected = []
    for pair_index in range(len(candidates)):
        available = np.flatnonzero(pair_indices == pair_index)
        if not len(available):
            continue
        collision_free = available[~rejected[available]]
        pool = collision_free if len(collision_free) else available
        selected.append(int(rng.choice(pool)))
    indices = np.asarray(selected, dtype=np.int64)
    indices = indices[np.argsort(-group.scores[indices], kind="stable")]
    if num_grasps is not None:
        indices = indices[:num_grasps]
    return GraspResult(
        contacts=contacts[indices], scores=group.scores[indices], poses=poses[indices],
        grasps=group[indices], candidates=len(candidates), rejected=int(rejected.sum()),
        rejected_grasps=group[rejected],
        finger_width=finger_width,
        base_offset=base_offset,
        grasp_height=height,
        grasp_depth=depth,
        rotation_degrees=rotations[indices],
        width_expansions=expansions[indices],
        selected_colliding=rejected[indices],
        orientations_tested=len(group),
    )


def gripper_geometries(grasp_group, color, finger_width=0.02, base_offset=0.012,
                       tail_length=None):
    """Plot solid finger, palm, and rear-tail boxes in Open3D."""
    if tail_length is None:
        tail_length = 4.0 * finger_width
    if tail_length <= 0:
        raise ValueError("tail_length must be positive")
    geometries = []
    for grasp in grasp_group:
        width, height, depth = grasp.width, grasp.height, grasp.depth
        boxes = (
            ((-base_offset, -width / 2 - finger_width, -height / 2),
             (depth + base_offset, finger_width, height)),
            ((-base_offset, width / 2, -height / 2),
             (depth + base_offset, finger_width, height)),
            ((-base_offset - finger_width, -width / 2 - finger_width, -height / 2),
             (finger_width, width + 2 * finger_width, height)),
            ((-base_offset - finger_width - tail_length, -finger_width / 2, -height / 2),
             (tail_length, finger_width, height)),
        )
        transform = np.eye(4)
        transform[:3, :3] = grasp.rotation_matrix
        transform[:3, 3] = grasp.translation
        for origin, dimensions in boxes:
            mesh = o3d.geometry.TriangleMesh.create_box(*dimensions)
            mesh.translate(origin)
            mesh.transform(transform)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            geometries.append(mesh)
    return geometries


def generate_grasps(point_cloud, gripper_position=(0.0, 0.0, 0.4), height=None,
                    depth=None, empty_thresh=10, num_grasps=5, collision_batch_size=32,
                    positive_z_only=True, gripper_scale=1.0,
                    rotation_step_degrees=15.0, orientation_seed=None,
                    max_width_expansion=0.02, width_step=0.002):
    point_cloud = validate_point_cloud(point_cloud)
    scaled_height = 0.02 * gripper_scale if height is None else height
    scaled_depth = 0.03 * gripper_scale if depth is None else depth
    if scaled_height <= 0 or scaled_depth <= 0 or gripper_scale <= 0 or empty_thresh < 0 or collision_batch_size < 1:
        raise ValueError("Grasp dimensions/batch size must be positive and empty threshold nonnegative")
    if num_grasps is not None and num_grasps < 1:
        raise ValueError("num_grasps must be positive or None")
    if not np.isfinite(gripper_position).all() or np.shape(gripper_position) != (3,):
        raise ValueError("Gripper position must contain three finite coordinates")
    # Preserve the existing Z>0 selection for .npy input.
    # Collision checks still use every point supplied.
    object_points = point_cloud[point_cloud[:, 2] > 0] if positive_z_only else point_cloud
    candidates = generate_contact_candidates(object_points) if len(object_points) else []
    finger_width = 0.02 * gripper_scale
    base_offset = 0.012 * gripper_scale
    return filter_candidates(
        candidates, point_cloud[:, :3], gripper_position, scaled_height, scaled_depth,
        empty_thresh, num_grasps, collision_batch_size, finger_width, base_offset,
        rotation_step_degrees, orientation_seed, max_width_expansion, width_step,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--point-cloud", type=Path,
                        help=".npy array (N,6): XYZ metres and oriented normals")
    inputs.add_argument("--mesh", type=Path,
                        help="Sample a mesh surface into an Open3D point cloud")
    parser.add_argument("--mesh-points", type=int, default=10000)
    parser.add_argument("--mesh-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gripper-position", type=float, nargs=3,
                        default=(0.0, 0.0, 0.4))
    parser.add_argument("--grasp-height", type=float,
                        help="Metres; mesh default 0.002, point-cloud default 0.02")
    parser.add_argument("--grasp-depth", type=float,
                        help="Metres; mesh default 0.025, point-cloud default 0.03")
    parser.add_argument("--gripper-scale", type=float,
                        help="Default 0.1 for mesh and 1.0 for point cloud")
    parser.add_argument("--empty-thresh", type=int, default=10)
    parser.add_argument("--num-grasps", type=int,
                        help="Limit selected pairs; omitted to keep one grasp per pair")
    parser.add_argument("--collision-batch-size", type=int, default=32)
    parser.add_argument("--rotation-step-degrees", type=float, default=15.0)
    parser.add_argument("--orientation-seed", type=int,
                        help="Make random width/orientation selection reproducible")
    parser.add_argument("--max-width-expansion", type=float, default=0.02)
    parser.add_argument("--width-step", type=float, default=0.002)
    parser.add_argument("--visualize", action="store_true",
                        help="Show the point cloud and selected solid grippers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gripper_scale = args.gripper_scale
    if gripper_scale is None:
        gripper_scale = 0.1 if args.mesh is not None else 1.0
    if gripper_scale <= 0:
        parser.error("--gripper-scale must be positive")
    if args.max_width_expansion < 0 or args.width_step <= 0:
        parser.error("--max-width-expansion must be nonnegative and --width-step positive")
    if args.mesh_points < 31:
        parser.error("--mesh-points must be at least 31")
    if args.point_cloud is not None and args.point_cloud.suffix.lower() != ".npy":
        parser.error("--point-cloud must be a .npy array")
    input_path = args.point_cloud if args.point_cloud is not None else args.mesh
    filenames = ("contacts.npy", "scores.npy", "poses.npy", "grasps.npy",
                 "rotation_degrees.npy", "width_expansions.npy",
                 "selected_colliding.npy", "rejected_grasps.npy", "summary.json")
    if any((args.output_dir / name).resolve() == input_path.resolve()
           for name in filenames):
        parser.error("Output files must not overwrite the input")
    if not args.overwrite and any((args.output_dir / name).exists()
                                  for name in filenames):
        parser.error("Output already exists; use --overwrite or a new output directory")

    if args.mesh is not None:
        cloud = load_mesh_point_cloud(args.mesh, args.mesh_points, args.mesh_seed)
        print(f"Loaded mesh surface cloud: {len(cloud)} points.")
    else:
        cloud = validate_point_cloud(np.load(args.point_cloud, allow_pickle=False))
    grasp_height = args.grasp_height
    grasp_depth = args.grasp_depth
    if args.mesh is not None:
        if grasp_height is None:
            grasp_height = 0.002
        if grasp_depth is None:
            grasp_depth = 0.025
    result = generate_grasps(
        cloud, args.gripper_position, grasp_height, grasp_depth,
        args.empty_thresh, args.num_grasps, args.collision_batch_size,
        positive_z_only=args.mesh is None, gripper_scale=gripper_scale,
        rotation_step_degrees=args.rotation_step_degrees,
        orientation_seed=args.orientation_seed,
        max_width_expansion=args.max_width_expansion, width_step=args.width_step,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "contacts.npy", result.contacts)
    np.save(args.output_dir / "scores.npy", result.scores)
    np.save(args.output_dir / "poses.npy", result.poses)
    np.save(args.output_dir / "rotation_degrees.npy", result.rotation_degrees)
    np.save(args.output_dir / "width_expansions.npy", result.width_expansions)
    np.save(args.output_dir / "selected_colliding.npy", result.selected_colliding)
    result.grasps.save_npy(str(args.output_dir / "grasps.npy"))
    result.rejected_grasps.save_npy(str(args.output_dir / "rejected_grasps.npy"))
    summary = dict(
        input_path=str(input_path.resolve()),
        input_kind="mesh surface cloud" if args.mesh else "point cloud",
        mesh_points=args.mesh_points if args.mesh else None,
        mesh_seed=args.mesh_seed if args.mesh else None,
        frame="original mesh coordinates" if args.mesh else "input point-cloud coordinates",
        units="metres", positive_z_only=args.mesh is None,
        gripper_position=args.gripper_position, gripper_scale=gripper_scale,
        grasp_height=result.grasp_height, grasp_depth=result.grasp_depth,
        finger_width=result.finger_width, base_offset=result.base_offset,
        empty_thresh=args.empty_thresh, num_grasps=args.num_grasps,
        contact_pairs=result.candidates,
        rotation_step_degrees=args.rotation_step_degrees,
        orientation_seed=args.orientation_seed,
        max_width_expansion=args.max_width_expansion, width_step=args.width_step,
        width_rotation_configurations_tested=result.orientations_tested,
        collision_or_empty_rejected=result.rejected,
        selected_collision_free=int((~result.selected_colliding).sum()),
        selected_collision_fallbacks=int(result.selected_colliding.sum()),
        saved=len(result.grasps),
    )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Contact pairs: {result.candidates}; selected: {len(result.grasps)} "
          f"({int(result.selected_colliding.sum())} collision fallbacks).")
    print(f"Output: {args.output_dir.resolve()}")

    if args.visualize:
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(cloud[:, :3])
        geometry.normals = o3d.utility.Vector3dVector(cloud[:, 3:])
        geometry.paint_uniform_color([0.65, 0.65, 0.65])
        valid = result.grasps[~result.selected_colliding]
        fallback = result.grasps[result.selected_colliding]
        geometries = [geometry]
        geometries += gripper_geometries(
            valid, [0.1, 0.9, 0.1], result.finger_width, result.base_offset
        )
        geometries += gripper_geometries(
            fallback, [0.9, 0.1, 0.1], result.finger_width, result.base_offset
        )
        print("Viewer: green collision-free grasps; red all-colliding fallbacks.")
        o3d.visualization.draw_geometries(
            geometries, window_name=f"Point cloud + {len(result.grasps)} grasps"
        )


if __name__ == "__main__":
    main()
