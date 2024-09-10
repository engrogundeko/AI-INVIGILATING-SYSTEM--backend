import cv2

def list_cameras(max_devices=10):
    available_cameras = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)  # Try to open the camera
        if cap.isOpened():
            print(f"Camera {i} is available.")
            available_cameras.append(i)
            cap.release()  # Release the camera
        else:
            print(f"Camera {i} is not available.")
    
    return available_cameras

if __name__ == "__main__":
    cameras = list_cameras()
    print(f"Available cameras: {cameras}")
