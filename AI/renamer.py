import os
import shutil
from glob import glob

dst = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\all"
src = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\frame_classifier\image"


def rename_and_copy_file(file, s, dst):
    base = os.path.basename(file)
    new_name = s + "_" + base
    new_file_path = os.path.join(dst, new_name)
    shutil.copyfile(file, new_file_path)
    print(f"Copied file to the location: {new_file_path}")


def main():
    os.makedirs(dst, exist_ok=True)
    paths = os.listdir(src)
    for path in paths:
        folder = os.path.join(src, path)
        if os.path.isdir(folder):
            files = glob(os.path.join(folder, "*.jpg"))
            for file in files:
                rename_and_copy_file(file, path, dst)


if __name__ == "__main__":
    main()
