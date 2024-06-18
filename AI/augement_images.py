import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def augment_image(image_path):
    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=40, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussianBlur(p=0.1),
        A.RandomSizedCrop(min_max_height=(200, 256), height=256, width=256, p=0.5),
        A.CoarseDropout(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Read the image from the file
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Apply the augmentations
    augmented = transform(image=image)
    augmented_image = augmented['image']

    return augmented_image

# Example usage:
augmented_image = augment_image("path_to_your_image.jpg")
