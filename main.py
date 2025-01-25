import cv2
import mediapipe as mp
import pyautogui  # For volume control
import time
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables for control
mode = "menu"  # Initial mode
prev_time = time.time()
selected_option = None
canvas = None

# Open Webcam
cap = cv2.VideoCapture(0)
canvas_initialized = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand Detection
    result = hands.process(rgb_frame)

    # Menu
    if mode == "menu":
        cv2.putText(frame, "1. Volume Control", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "2. Drawing", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]

                height, width, _ = frame.shape
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

                # Draw circle on index finger tip
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)

                # Check if index finger tip is near the options
                if 40 < index_x < 300 and 70 < index_y < 120:  # Near "Volume Control"
                    selected_option = "volume"
                    mode = "volume"
                elif 40 < index_x < 300 and 130 < index_y < 180:  # Near "Drawing"
                    selected_option = "drawing"
                    mode = "drawing"

    # Volume Control Mode
    elif mode == "volume":
        cv2.putText(frame, "Volume Control Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                index_base = hand_landmarks.landmark[5]

                height, width, _ = frame.shape
                index_tip_y = int(index_tip.y * height)
                index_base_y = int(index_base.y * height)

                current_time = time.time()
                if current_time - prev_time > 0.2:  # Delay for smooth changes
                    if index_tip_y < index_base_y:  # Index finger pointing up
                        pyautogui.press("volumeup")
                    elif index_tip_y > index_base_y:  # Index finger pointing down
                        pyautogui.press("volumedown")
                    prev_time = current_time

    # Drawing Mode
    elif mode == "drawing":
        if not canvas_initialized:
            canvas = np.zeros_like(frame)
            canvas_initialized = True

        cv2.putText(frame, "Drawing Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]

                height, width, _ = frame.shape
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

                # Draw circle where index finger is
                cv2.circle(canvas, (index_x, index_y), 5, (0, 255, 0), -1)

        # Blend canvas with original frame
        frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Display Frame
    cv2.imshow('Hand Gesture AI', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
