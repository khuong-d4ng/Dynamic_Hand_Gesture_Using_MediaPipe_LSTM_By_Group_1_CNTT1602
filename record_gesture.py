import cv2
import mediapipe as mp
import numpy as np
import os
import datetime

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

recording = False
base_output_path = "gesture_videos/"
os.makedirs(base_output_path, exist_ok=True)  

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = None  
sequence = []  

gesture_name = input("Nhap ten cu chi (e.g., 'wave'): ").strip()
start_number = int(input("Nhap so bat dau (e.g., 10 de bat dau tu #11): ").strip())

gesture_path = os.path.join(base_output_path, gesture_name)
os.makedirs(gesture_path, exist_ok=True) 

video_path = os.path.join(gesture_path, "videos")
landmark_path = os.path.join(gesture_path, "landmarks")
os.makedirs(video_path, exist_ok=True) 
os.makedirs(landmark_path, exist_ok=True)

def get_next_count(gesture_name, folder_path, start_number):
    existing_files = [f for f in os.listdir(folder_path) if f.startswith(gesture_name) and f.endswith(('.mp4', '_landmarks.npy'))]
    max_count = start_number - 1  
    for file in existing_files:
        try:
            count = int(''.join(filter(str.isdigit, file.split(gesture_name)[1].split('.')[0])))
            max_count = max(max_count, count)
        except (IndexError, ValueError):
            continue
    return max_count + 1

gesture_count = {gesture_name: get_next_count(gesture_name, video_path, start_number)}

frame_count = 0
max_frames = 150 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if recording:
                sequence.append(landmarks)

    if recording:
        cv2.putText(frame, "Dang quay video, bam 's' de ket thuc", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Bam 'r' de bat dau quay, 'q' de thoat", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and not recording:
        recording = True
        frame_count = 0
        sequence = [] 
        count = gesture_count[gesture_name]
        out = cv2.VideoWriter(os.path.join(video_path, f"{gesture_name}{count}.mp4"), 
                              fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        gesture_count[gesture_name] += 1 
    elif key == ord('s') and recording:
        recording = False
        out.release() 
        np.save(os.path.join(landmark_path, f"{gesture_name}{count}_landmarks.npy"), np.array(sequence))
        print(f"Saved {len(sequence)} frames of landmarks for {gesture_name}#{count}")
    elif key == ord('q'):
        break

    if recording:
        out.write(frame) 
        frame_count += 1
        if frame_count >= max_frames:  
            recording = False
            out.release()
            np.save(os.path.join(landmark_path, f"{gesture_name}{count}_landmarks.npy"), np.array(sequence))
            print(f"Saved {len(sequence)} frames of landmarks for {gesture_name}#{count}")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()