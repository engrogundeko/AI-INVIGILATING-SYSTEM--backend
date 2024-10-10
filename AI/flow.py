import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler


path = r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\media\datasets\json\normal"
video_files = os.listdir(path)
video_paths = [os.path.join(path, file) for file in video_files]

scaler = StandardScaler()


normalized_flows = []
normal = []
for path in video_paths:
    with open(path, "r") as json_file:
        data = json.load(json_file)
        
        for cheat in data:
            # print(cheat["coordinate"])
            x1, y1, x2, y2 = map(int, cheat.get("coordinate"))
            object_size = (x2 - x1) * (y2 - y1)
            for flow in cheat["flows"]:
                normal.append(flow)
                normalized_flow = flow / object_size
                normalized_flows.append(normalized_flow)
                

normalized_flow_array = np.array(normalized_flows).reshape(-1, 1)


scaled_values = scaler.fit_transform(normalized_flow_array)
new = []
for scale in scaled_values:
    new.append(list(scale).pop())
    

# all = dict(normalized_flow=new, normal_values=normal)
print(new[:10])

with open(f"normalized2.json", "w") as json_file:
    json.dump(new, json_file, indent=4)
