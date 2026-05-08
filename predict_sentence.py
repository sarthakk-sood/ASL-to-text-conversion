import cv2
import mediapipe as mp
import pickle
import numpy as np
from textblob import TextBlob
import collections

# Load Model
try:
    with open("mlp_asl_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model not found! Run train_mlp.py first.")
    exit()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1)

landmarker = HandLandmarker.create_from_options(options)
mp_drawing = mp.tasks.vision.drawing_utils
# we need to import pose/hand landmarks connections manually
try:
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
except AttributeError:
    connections = None

# Default built-in webcam index is usually 0.
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
    exit()

# Prediction stabilization
prediction_buffer = collections.deque(maxlen=15)
current_word = ""
sentence = ""

frames_held = 0
last_stable_pred = None

def get_most_frequent(buffer):
    if not buffer: return None
    return max(set(buffer), key=buffer.count)

print("Starting Camera... Hold a sign to type. Sign 'space' to start a new word.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)

    if results.hand_landmarks:
        for hand_landmarks, hand_world_landmarks in zip(results.hand_landmarks, results.hand_world_landmarks):
            
            # Use relative coordinates (same as training)
            base_x, base_y, base_z = hand_landmarks[0].x, hand_landmarks[0].y, hand_landmarks[0].z
            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
            # Predict
            pred = model.predict([landmarks])[0]
            prediction_buffer.append(pred)
            
            stable_pred = get_most_frequent(prediction_buffer)

            # Auto-type logic: if gesture is held consistently for ~40 frames (2x original)
            if stable_pred == last_stable_pred:
                frames_held += 1
            else:
                frames_held = 0
                last_stable_pred = stable_pred

            # Progress bar for visual feedback
            cv2.rectangle(frame, (10, 60), (10 + (frames_held * 5), 80), (0, 255, 0), -1)

            if frames_held >= 40: 
                # Trigger action
                if stable_pred == 'space':
                    if current_word:
                        corrected_word = str(TextBlob(current_word).correct())
                        sentence += corrected_word + " "
                        current_word = ""
                elif stable_pred == 'del':
                    if len(current_word) > 0:
                        current_word = current_word[:-1]
                elif stable_pred != 'nothing':
                    # Add character
                    current_word += stable_pred
                
                prediction_buffer.clear()
                frames_held = -10 # Cooldown to avoid rapid double-typing
                
            cv2.putText(frame, f"Prediction: {stable_pred}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    cv2.putText(frame, f"Current Word: {current_word}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASL Predictor", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
