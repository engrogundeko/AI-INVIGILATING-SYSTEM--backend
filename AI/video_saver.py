import cv2
def save_video_clip(video_path, start_frame, end_frame, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame):
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
    
    cap.release()
    out.release()

# Example usage
save_video_clip('path_to_video.mp4', 100, 200, 'output_clip.avi')
