import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def extract_frames(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        if count % frame_rate == 0 and success:
            frames.append(frame)
        count += 1
    
    cap.release()
    return frames

def resize_frames(frames, size=(64, 64)):
    resized_frames = [cv2.resize(frame, size) for frame in frames]
    return resized_frames

def normalize_frames(frames):
    normalized_frames = [frame / 255.0 for frame in frames]
    return normalized_frames

def create_sequences(frames, sequence_length=10):
    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        sequence = frames[i:i + sequence_length]
        sequences.append(sequence)
    return sequences

def prepare_data(video_paths, labels, frame_rate=5, frame_size=(64, 64), sequence_length=10):
    all_sequences = []
    all_labels = []
    
    for video_path, label in zip(video_paths, labels):
        frames = extract_frames(video_path, frame_rate=frame_rate)
        resized_frames = resize_frames(frames, size=frame_size)
        normalized_frames = normalize_frames(resized_frames)
        sequences = create_sequences(normalized_frames, sequence_length=sequence_length)
        
        all_sequences.extend(sequences)
        all_labels.extend([label] * len(sequences))
    
    X = np.array(all_sequences)
    y = np.array(all_labels)
    
    return X, y

# List of video paths and corresponding labels
video_paths = ['path_to_video1.mp4', 'path_to_video2.mp4']
labels = [0, 1]  # Example labels

X, y = prepare_data(video_paths, labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Now X_train, X_val, y_train, and y_val can be used for training the model
