import numpy as np
import random
import cv2
import albumentations as A
from glob import glob
import os
import shutil
from config import IMG_OUT_DIR

# Define your augmentation pipeline
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=40, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussianBlur(p=0.1),
    ]
)


def augment_image(image):
    augmented = augmentation_pipeline(image=image)
    return augmented["image"]


# Define class directories and their counts
class_dirs = {
    "back": 607,
    "exchange": 425,
    "movement": 383,
    "normal": 689,
    "peek": 348,
    # "Phone": 314,
    # "Talking": 158
}

# Define the target number of images per class (e.g., maximum count)
target_count = 1500

# Balance the dataset
for class_name, count in class_dirs.items():
    class_path = f"{IMG_OUT_DIR}/{class_name}"  # Update with your dataset path
    images = glob(
        os.path.join(class_path, "*.jpg")
    )  # Assuming images are in .jpg format

    if count < target_count:
        # Oversample by augmentation
        while len(images) < target_count:
            image_path = random.choice(images)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_image = augment_image(image)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            augmented_image_path = os.path.join(class_path, f"aug_{len(images)}.jpg")
            cv2.imwrite(augmented_image_path, augmented_image)
            images.append(augmented_image_path)
    elif count > target_count:
        # Undersample by random removal
        images_to_remove = random.sample(images, count - target_count)
        for image_path in images_to_remove:
            os.remove(image_path)

print("Dataset balancing complete.")
