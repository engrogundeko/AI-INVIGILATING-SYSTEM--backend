import os
from glob import glob
from pathlib import Path
import pillow_heif
from PIL import Image

img_path = r"C:\Users\Admin\Downloads\heic_files"
dst = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\heic"

def convert_heic_to_jpg(heic_path, jpg_path):
    pillow_heif.register_heif_opener()

    image = Image.open(heic_path)
    # image.show('IMAGE')
    image.save(jpg_path, "JPEG")

def look_for_heif_files():
    path = r"C:\Users\Admin\Downloads"
    paths = glob(os.path.join(path, "*.heic"))
    for path in paths:
        convert_heic_to_jpg(path, dst)

paths = os.listdir(dst)
for path in paths:
    if not os.path.basename(path).split(".")[1] == "MOV":
        name_path = os.path.join(dst, os.path.basename(path).removesuffix(".HEIC"))
        files_path = os.path.join(dst, path)
        convert_heic_to_jpg(files_path, f"{name_path}.jpg")

