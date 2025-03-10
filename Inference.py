import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

model = load_model('model.h5') #Đổi thành path tới model vừa lưu


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
 
buffer = deque(maxlen=50)

gesture_labels = ["fist", "swipe_left", "wave", "swipe_up"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    buffer.append(landmarks if landmarks else [0.0] * 63)

    if len(buffer) >= 1:
        sequence = list(buffer)
        if len(sequence) < 150:
            sequence += [[0.0] * 63] * (150 - len(sequence))
        
        input_data = np.array(sequence, dtype=np.float32).reshape(1, 150, 63)
        
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        gesture_name = gesture_labels[predicted_class]

        cv2.putText(frame, f"{gesture_name} ({confidence:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()