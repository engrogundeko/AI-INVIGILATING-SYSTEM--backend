from glob import glob
import os
from ai import main

path = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\converted"
output_base_dir = (
    r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\recordings\video"
)
json_base_dir = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\recordings\json"

video_files = glob(os.path.join(path, "*.mp4"))

for video in video_files:
    base = os.path.basename(video).split(".")[0]
    pth = video
    output_dir = os.path.join(output_base_dir, f"{base}_s.avi")
    # print(output_dir)
    json_dump_file = os.path.join(json_base_dir, f"{base}_s.json")
    main(pth, output_dir, json_dump_file)
