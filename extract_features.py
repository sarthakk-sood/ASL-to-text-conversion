import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from tqdm import tqdm

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

landmarker = HandLandmarker.create_from_options(options)

def extract_landmarks(image_path):
    try:
        mp_image = mp.Image.create_from_file(image_path)
    except Exception:
        return None

    results = landmarker.detect(mp_image)
    
    if results.hand_landmarks:
        # Get the first hand detected
        hand_landmarks = results.hand_landmarks[0]
        # Use relative coordinates for translation invariance
        base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
        return landmarks
    return None

def main():
    dataset_dir = r"d:\AIProject\asl_alphabet_train\asl_alphabet_train"
    data = []
    
    # Iterate through each folder (A-Z, space, del, nothing)
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for cls in classes:
        print(f"Processing class {cls}...")
        class_dir = os.path.join(dataset_dir, cls)
        # Limit to first 100 images per class for faster extraction during setup
        images = os.listdir(class_dir)[:200]
        for img_name in tqdm(images):
            img_path = os.path.join(class_dir, img_name)
            landmarks = extract_landmarks(img_path)
            if landmarks is not None:
                landmarks.append(cls)  # target label
                data.append(landmarks)
                
    # Create DataFrame and save
    columns = [f"lm_{i}" for i in range(63)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("asl_landmarks.csv", index=False)
    print("Feature extraction complete. Data saved to asl_landmarks.csv")

if __name__ == "__main__":
    main()
