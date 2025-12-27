import open3d as o3d
import numpy as np
import json
import os
import re
import csv

def create_dense_box_points(corners, points_per_edge=100):
    edges = [
        (0,1), (1,3), (3,2), (2,0),  # bottom face
        (4,5), (5,7), (7,6), (6,4),  # top face
        (0,5), (1,4), (2,7), (3,6)   # vertical edges
    ]
    box_points = []
    for start_idx, end_idx in edges:
        start = corners[start_idx]
        end = corners[end_idx]
        points = np.linspace(start, end, points_per_edge)
        box_points.extend(points)
    return np.array(box_points)

def transform_corners_cam_to_lidar(corners, T_cam_lidar):
    corners_hom = np.hstack((corners, np.ones((8, 1))))
    corners_lidar = (T_cam_lidar @ corners_hom.T).T[:, :3]
    return corners_lidar

def approximate_color_name(rgb):
    known_colors = {
        (1.0, 0.0, 0.0): "Red",
        (0.0, 1.0, 0.0): "Green",
        (0.0, 0.0, 1.0): "Blue",
        (1.0, 1.0, 0.0): "Yellow",
        (1.0, 0.5, 0.0): "Orange",
        (0.5, 0.0, 0.5): "Purple",
        (0.0, 1.0, 1.0): "Cyan",
        (1.0, 0.0, 1.0): "Magenta",
        (0.5, 0.5, 0.5): "Gray",
        (0.8, 0.8, 0.8): "Light Gray",
    }
    return known_colors.get(rgb, "Unknown")

def process_and_save(ply_path, json_path, output_path, csv_writer, image_number):
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        original_points = np.asarray(pcd.points)
        original_colors = np.asarray(pcd.colors)

        if original_colors.shape[0] != original_points.shape[0]:
            light_gray_rgb = np.array([200/255, 200/255, 200/255])
            original_colors = np.tile(light_gray_rgb, (original_points.shape[0], 1))

        with open(json_path) as f:
            boxes = json.load(f)

        T_cam_lidar = np.array([
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ])

        all_box_points = []
        all_box_colors = []
        box_corners_list = []
        for box in boxes:
            corners_cam = np.array(box["corners_cam0"])
            corners_lidar = transform_corners_cam_to_lidar(corners_cam, T_cam_lidar)
            dense_points = create_dense_box_points(corners_lidar)
            all_box_points.append(dense_points)
            all_box_colors.append(np.tile([[0.0, 0.0, 0.0]], (len(dense_points), 1)))
            box_corners_list.append(corners_lidar)

        if all_box_points:
            box_points = np.vstack(all_box_points)
            box_colors = np.vstack(all_box_colors)
            combined_points = np.vstack([original_points, box_points])
            combined_colors = np.vstack([original_colors, box_colors])
        else:
            combined_points = original_points
            combined_colors = original_colors

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

        o3d.io.write_point_cloud(output_path, combined_pcd)
        print(f" Saved: {output_path}")

        print("\n Analyzing color-to-box assignment...\n")

        lidar_points = combined_points[:original_points.shape[0]]
        lidar_colors = combined_colors[:original_points.shape[0]]

        color_labels = {}
        for idx, color in enumerate(lidar_colors):
            color_key = tuple(np.round(color, 3))
            if np.allclose(color_key, [200/255]*3, atol=1e-2) or np.allclose(color_key, [0.0, 0.0, 0.0], atol=1e-3):
                continue
            if color_key not in color_labels:
                color_labels[color_key] = []
            color_labels[color_key].append(idx)

        # ---- New logic starts here ----
        color_to_box = {}  # Map each color to exactly one box idx
        box_has_color = set()  # Track boxes assigned to any color

        for color_key, indices in color_labels.items():
            color_points = lidar_points[indices]
            box_vote_counts = []
            box_point_indices_inside = []

            for box_corners in box_corners_list:
                box_min = np.min(box_corners, axis=0)
                box_max = np.max(box_corners, axis=0)

                inside_mask = np.all((color_points >= box_min) & (color_points <= box_max), axis=1)
                count_inside = np.sum(inside_mask)

                box_vote_counts.append(count_inside)
                box_point_indices_inside.append(np.array(indices)[inside_mask])

            best_box_idx = np.argmax(box_vote_counts)
            best_vote = box_vote_counts[best_box_idx]

            if best_vote == 0:
                print(f" Color {color_key}  No matching box.")
                continue

            # If tie in top two votes
            top_two = sorted(box_vote_counts, reverse=True)[:2]
            if len(top_two) > 1 and top_two[0] == top_two[1]:
                print(f" Color {color_key}  Tie between multiple boxes.")
                continue

            # Assign color uniquely to the box with max points
            color_to_box[color_key] = best_box_idx
            box_has_color.add(best_box_idx)

            inside_indices = box_point_indices_inside[best_box_idx]
            all_indices = np.array(indices)
            outside_indices = np.setdiff1d(all_indices, inside_indices)

            rgb_string = f"RGB: {tuple(round(c * 255) for c in color_key)}"
            color_name = approximate_color_name(color_key)

            distances_from_lidar = np.linalg.norm(lidar_points[inside_indices], axis=1)
            min_distance = np.min(distances_from_lidar)

            # New part: count points outside box but within 0.5m from any inside point of this color
            points_inside = lidar_points[inside_indices]
            points_outside = lidar_points[outside_indices]

            # If there are outside points and inside points
            count_outside_within_05m = 0
            if len(points_outside) > 0 and len(points_inside) > 0:
                # For efficiency, build a KD-tree from points_inside
                import scipy.spatial
                kdtree = scipy.spatial.cKDTree(points_inside)
                # Query all outside points, get distances to nearest inside point
                distances, _ = kdtree.query(points_outside, distance_upper_bound=0.5)
                # distances with inf means no inside point within 0.5
                count_outside_within_05m = np.sum(distances != np.inf)

            print(f" Color {color_key} ({color_name}, {rgb_string}): Box #{best_box_idx} owns it.")
            print(f"   → Points inside: {len(inside_indices)}")
            print(f"   → Points outside: {len(outside_indices)}")  # Points outside the assigned box
            print(f"   → Points outside but within 0.5m: {count_outside_within_05m}")
            print(f"   → Nearest point distance to box center: {min_distance:.2f} meters\n")

            # Write a CSV row: image_number, color RGB, inside points, outside points, outside within 0.5m, min distance
            rgb_str = f"{int(color_key[0]*255)}-{int(color_key[1]*255)}-{int(color_key[2]*255)}"


            csv_writer.writerow([
                image_number,
                rgb_str,
                len(inside_indices),
                len(outside_indices),
                count_outside_within_05m,
                f"{min_distance:.2f}"
            ])


        # Remove boxes that have no assigned colors
        print("\n Cleaning boxes with no assigned colors...")

        final_box_points = []
        final_box_colors = []

        for idx, (box_corners, dense_points) in enumerate(zip(box_corners_list, all_box_points)):
            if idx in box_has_color:
                final_box_points.append(dense_points)
                final_box_colors.append(np.tile([[0.0, 0.0, 0.0]], (len(dense_points), 1)))
            else:
                print(f" Box #{idx} removed due to no assigned color points.")

        # Merge remaining box points with original LiDAR
        if final_box_points:
            box_points = np.vstack(final_box_points)
            box_colors = np.vstack(final_box_colors)
            combined_points = np.vstack([original_points, box_points])
            combined_colors = np.vstack([original_colors, box_colors])
        else:
            combined_points = original_points
            combined_colors = original_colors

        # Create final point cloud
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

        # Save to .ply
        o3d.io.write_point_cloud(output_path, combined_pcd)
        print(f" Saved: {output_path}")

    except Exception as e:
        print(f" Error processing {ply_path}: {e}")

if __name__ == "__main__":
    ply_path = r"D:\\Lira2\\colored_pointclouds_output"
    json_path = r"D:\\Lira2\\KITTI-360_sample\\bboxes_3D_cam0"
    output_folder = r"D:\\Lira2\\final_op2"
    csv_folder = r"D:\\Lira2\\csvfile"

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    csv_file_path = os.path.join(csv_folder, "baredata2.csv")

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Image_Number",
            "Car_RGB",
            "Points_Inside",
            "Points_Outside",
            "Points_Outside_Within_0.5m",
            "Nearest_Point_Distance"
        ])

        ply_files = sorted([f for f in os.listdir(ply_path) if f.endswith(".ply")])
        count = 1
        for ply_file in ply_files:
            match = re.search(r"(\d+)", ply_file)
            if not match:
                print(f" Could not extract frame number from: {ply_file}")
                continue

            frame_number = str(int(match.group(1)))
            json_filename = f"BBoxes_{frame_number}.json"
            json_file_path = os.path.join(json_path, json_filename)

            if not os.path.exists(json_file_path):
                print(f" JSON not found for {ply_file} → Expected: {json_filename}")
                continue

            ply_file_path = os.path.join(ply_path, ply_file)
            output_file_path = os.path.join(output_folder, ply_file)

            print(f" Matching {ply_file} → {json_filename}")
            print(f"image count: {count}")

            process_and_save(ply_file_path, json_file_path, output_file_path, csv_writer, count)
            count += 1
