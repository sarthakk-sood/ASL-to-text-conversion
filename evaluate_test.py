import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    test_dir = r"d:\AIProject\asl_alphabet_test\asl_alphabet_test"
    
    # Load Model
    try:
        with open("mlp_asl_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model not found! Run train_mlp.py first.")
        return

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1)

    landmarker = HandLandmarker.create_from_options(options)

    y_true = []
    y_pred = []

    print(f"Evaluating images in {test_dir}...")
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found {test_dir}")
        return

    for img_name in os.listdir(test_dir):
        if not img_name.endswith('.jpg'):
            continue
            
        # Parse label from filename (e.g. A_test.jpg -> A)
        label = img_name.replace("_test.jpg", "")
        img_path = os.path.join(test_dir, img_name)
        
        try:
            mp_image = mp.Image.create_from_file(img_path)
        except Exception:
            print(f"Skipping {img_name}: could not load image.")
            continue
            
        results = landmarker.detect(mp_image)
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Relative coordinates
            base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
            pred = model.predict([landmarks])[0]
            
            y_true.append(label)
            y_pred.append(pred)
            print(f"File: {img_name:<15} | True: {label:<8} | Pred: {pred}")
        else:
            print(f"File: {img_name:<15} | True: {label:<8} | No hand detected")

    if len(y_true) > 0:
        print("\n--- Evaluation Results ---")
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc*100:.2f}%\n")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("No valid predictions were made.")

if __name__ == "__main__":
    main()
