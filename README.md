# lidar_cam_sensor_fusion


Overview
This project performs sensor fusion between 3D LiDAR point clouds and 2D instance segmentation to detect and analyze vehicles in a scene.


Project Pipeline
2D Instance Segmentation
Vehicles are segmented from RGB images.
Each detected car is assigned a unique mask and color.
3D → 2D Projection
LiDAR point clouds are transformed into the camera frame.
Projection uses calibrated intrinsic and extrinsic matrices.
Point Cloud Colorization
Projected LiDAR points inherit colors from segmented vehicle regions.
Points overlapping a vehicle mask are labeled accordingly.
3D Bounding Box Fusion
Colored LiDAR clusters are used to generate 3D bounding boxes.
Detection Evaluation
Detections are validated using Point Containment Ratio.




File Descriptions
1. segmentation_of_2dImage.py
Performs instance segmentation on 2D images.
Functionality:
Runs YOLOv8/YOLOv9 instance segmentation.
Detects vehicles (cars).
Generates:
Segmentation masks
Bounding boxes
Unique color labels for each vehicle instance
Output:
Segmented images
Mask data for each detected car

2. color_3d_pointcloud.py
Projects and colors LiDAR point clouds using 2D segmentation results.
Method:
Transforms 3D LiDAR points into the camera coordinate frame.
Uses intrinsic and extrinsic calibration matrices.
Projects 3D points onto the 2D image plane.
Checks which projected points fall inside segmented vehicle masks.
Assigns the corresponding vehicle color to those 3D points.
Result:
Colorized 3D point cloud
Vehicle-specific LiDAR clusters

3. complete_sensor_fusion(Detection).py
Performs full fusion and detection evaluation.
Key Steps:
Aggregates colored LiDAR clusters.
Generates 3D bounding boxes (3D BB) around vehicles.
Associates 3D detections with 2D segmentation results.
Computes detection validity using Point Containment Ratio.

Detection Evaluation Metric
Point Containment Ratio (PCR)
A quantitative metric to evaluate detection accuracy.
Definition:
[
PCR = \frac{\text{Number of LiDAR points inside a bounding box}}{\text{Total LiDAR points for that object}}
]
Classification Rule:
A detection is considered a True Positive if:
PCR > 0.6

This ensures that the majority of LiDAR points corresponding to a vehicle lie within its detected bounding box.

