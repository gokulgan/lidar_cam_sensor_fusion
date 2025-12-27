
import os
import cv2
import numpy as np
from ultralytics import YOLO
import random

def get_random_unique_color(used_colors):
    max_attempts = 1000
    for _ in range(max_attempts):
        color = tuple(random.randint(50, 255) for _ in range(3))
        if color not in used_colors:
            used_colors.add(color)
            return color
    raise RuntimeError("Failed to generate a unique color after many attempts.")

def process_images():
    image_folder = r"D:\\lira_data\\KITTI-360_sample\\data_2d_raw\\2013_05_28_drive_0000_sync\\image_00\data_rect"
    output_folder = r"D:\\Lira2\\opim\\2ndtryimgnpy"
    output_folder2 = r"D:\\Lira2\\opim\\22ndtryimgnpy"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    model = YOLO('yolov8l-seg.pt')  # Segmentation model

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        output_path2 = os.path.join(output_folder2, image_file)

        img = cv2.imread(image_path)
        results = model.predict(image_path)

        colored_mask = np.zeros_like(img, dtype=np.uint8)
        cars_info = {}
        car_count = 1
        used_colors = set()

        for result in results:
            if result.masks is not None:
                for cls, mask in zip(result.boxes.cls, result.masks.data):
                    if int(cls) == 2:  # Class 2 = car
                        color = get_random_unique_color(used_colors)

                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        mask_np = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                        colored_mask[mask_np > 0] = color
                        cars_info[car_count] = {
                            'color': color,
                            'mask': mask_np
                        }
                        car_count += 1

        alpha = 0.5
        segmented_img = cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0)
        segmented_img2 = colored_mask

        cv2.imwrite(output_path, segmented_img)
        cv2.imwrite(output_path2, segmented_img2)

        info_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_cars.npy")
        np.save(info_path, cars_info)

        print(f"Processed {image_file} with {car_count - 1} cars.")

if __name__ == "__main__":
    process_images()
