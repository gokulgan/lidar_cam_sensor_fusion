import numpy as np
import cv2
import os

def load_calibration():
    R_rect_00 = np.array([
        [0.999974, -0.007141, -0.000089],
        [0.007141, 0.999969, -0.003247],
        [0.000112, 0.003247, 0.999995]
    ])
    P_rect_00 = np.array([
        [552.554261, 0.000000, 682.049453, 0.000000],
        [0.000000, 552.554261, 238.769549, 0.000000],
        [0.000000, 0.000000, 1.000000, 0.000000]
    ])
    T_cam_lidar = np.array([
        [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
        [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
        [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
        [0, 0, 0, 1]
    ])
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    return R_rect_00, P_rect_00, T_lidar_cam

def project_lidar_to_image(points_3d, T_lidar_cam, R_rect, P_rect):
    points_hom = np.column_stack([points_3d, np.ones(len(points_3d))])
    points_cam = (T_lidar_cam @ points_hom.T).T[:, :3]
    points_rect = (R_rect @ points_cam.T).T
    points_img = (P_rect @ np.column_stack([points_rect, np.ones(len(points_rect))]).T).T
    points_2d = points_img[:, :2] / points_img[:, [2]]
    valid = points_img[:, 2] > 0
    return points_2d, valid

def save_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[2]} {c[1]} {c[0]}\n")  # BGR->RGB swap

# def color_lidar_points(bin_path, image_path, output_ply_path):
#     R_rect_00, P_rect_00, T_lidar_cam = load_calibration()

#     points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Failed to load image: {image_path}")
#         return

#     points_2d, valid = project_lidar_to_image(points, T_lidar_cam, R_rect_00, P_rect_00)
#     points_2d_int = np.round(points_2d).astype(int)

#     colors = np.zeros_like(points, dtype=np.uint8)
#     # colors = np.full((len(points), 3), [200, 200, 200], dtype=np.uint8)
#     # colors = np.full(points.shape, [200, 200, 200], dtype=np.uint8)



#     h, w = img.shape[:2]

#     for i, (valid_flag, (u, v)) in enumerate(zip(valid, points_2d_int)):
#         if valid_flag and 0 <= u < w and 0 <= v < h:
#             color = img[v, u]
#             if not np.all(color == 0):
#                 colors[i] = color
#             elif np.all(color==0):
#                 # Background point → light gray
#                 colors[i] = [200, 200, 200]
#         else:
#             # Invalid projection → also light gray
#             colors[i] = [200, 200, 200]

#     save_ply(output_ply_path, points, colors)
#     print(f"Saved colored point cloud to {output_ply_path}")



def color_lidar_points(bin_path, image_path, output_ply_path):
    R_rect_00, P_rect_00, T_lidar_cam = load_calibration()

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    points_2d, valid = project_lidar_to_image(points, T_lidar_cam, R_rect_00, P_rect_00)
    points_2d_int = np.round(points_2d).astype(int)

    # Initialize ALL points to light grey (BGR: [200, 200, 200])
    colors = np.full((len(points), 3), [200, 200, 200], dtype=np.uint8)
    
    h, w = img.shape[:2]

    for i, (valid_flag, (u, v)) in enumerate(zip(valid, points_2d_int)):
        if valid_flag and 0 <= u < w and 0 <= v < h:
            color = img[v, u]
            if np.all(color < 10):  # Ignore very dark/black pixels
                colors[i] = [200, 200, 200]
            else:
                colors[i] = color
        else:
            colors[i] = [200, 200, 200]
            


    
    # Final check: Ensure no black points remain
    black_points = np.all(colors == [0, 0, 0], axis=1)
    if np.any(black_points):
        print(f"Warning: {np.sum(black_points)} points are still black!")
        # Force them to grey
        colors[black_points] = [200, 200, 200]

    save_ply(output_ply_path, points, colors)
    print(f"Saved colored point cloud to {output_ply_path}")

def batch_process(bin_folder, image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    bin_files = sorted([f for f in os.listdir(bin_folder) if f.endswith('.bin')])
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if len(bin_files) != len(image_files):
        print("Warning: Number of .bin files and images do not match!")

    for bin_file, image_file in zip(bin_files, image_files):
        bin_path = os.path.join(bin_folder, bin_file)
        image_path = os.path.join(image_folder, image_file)
        output_ply_path = os.path.join(output_folder, f"{os.path.splitext(bin_file)[0]}_colored.ply")

        print(f"Processing {bin_file} and {image_file}...")
        color_lidar_points(bin_path, image_path, output_ply_path)

if __name__ == "__main__":
    bin_folder = r"D:\\Lira2\\KITTI-360_sample\\data_3d_raw\\2013_05_28_drive_0000_sync\\velodyne_points\\data"
    image_folder = r"D:\\Lira2\\opim\\22ndtryimgnpy"  
    output_folder = r"D:\\Lira2\\colored_pointclouds_output"

    batch_process(bin_folder, image_folder, output_folder)





