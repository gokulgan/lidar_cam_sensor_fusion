# lidar_cam_sensor_fusion

## Project Overview

This project performs **multi-sensor fusion** by combining:

- 3D LiDAR point clouds  
- 2D YOLOv8 instance segmentation  

The goal is to detect vehicles in 3D space by projecting LiDAR data onto segmented 2D images and evaluating detection accuracy using a custom quantitative metric called **Point Containment Ratio (PCR)**.

---

## Methodology

The pipeline consists of the following stages:

1. **2D Instance Segmentation**
   - Vehicles are segmented from RGB images.
   - Each detected car is assigned a unique mask and color.

2. **3D → 2D Projection**
   - LiDAR points are transformed into the camera frame.
   - Projection is done using calibration matrices.

3. **Point Cloud Colorization**
   - Projected points inherit colors from segmented vehicles.

4. **3D Bounding Box Generation**
   - Colored clusters are used to form 3D bounding boxes.

5. **Detection Evaluation**
   - Bounding boxes are validated using Point Containment Ratio.

---


## File Descriptions

### segmentation_of_2dImage.py

Performs **YOLOv8/YOLOv9 instance segmentation** on 2D images.

**Key Functions:**

- Detects vehicles (cars)
- Generates segmentation masks
- Produces bounding boxes
- Assigns unique colors to each car instance

**Outputs:**

- Segmented images  
- Vehicle masks  
- Detection metadata  

---

### color_3d_pointcloud.py

Colors LiDAR point clouds using segmentation results.

**Process:**

1. Transform LiDAR points to camera coordinates.
2. Use **intrinsic** and **extrinsic** calibration matrices.
3. Project 3D points onto the 2D image plane.
4. Check which points fall inside vehicle masks.
5. Assign the corresponding vehicle color to those points.

**Result:**

- Colorized 3D point cloud  
- Vehicle-specific LiDAR clusters  

---

### complete_sensor_fusion(Detection).py

Performs full fusion and detection evaluation.

**Functions:**

- Generates 3D bounding boxes (3D BB)
- Associates LiDAR clusters with 2D detections
- Computes detection validity
- Classifies True Positives

---

## Detection Evaluation Metric

### Point Containment Ratio (PCR)

A custom metric used to evaluate detection accuracy.

#### Formula

\[
PCR = \frac{N_{inside}}{N_{total}}
\]

Where:

- \(N_{inside}\) = Number of LiDAR points inside the bounding box  
- \(N_{total}\) = Total LiDAR points belonging to that object  

---

### True Positive Condition

A detection is classified as **True Positive** if:

\[
PCR > 0.6
\]

This ensures that most LiDAR points of a vehicle lie within its detected 3D bounding box.





