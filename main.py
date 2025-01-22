import cv2
import mediapipe as mp
import pyautogui  # For volume and brightness control
import time

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables for control
prev_time = time.time()
right_hand_mode = 'volume'  # Right hand controls volume
left_hand_mode = 'brightness'  # Left hand controls brightness

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand Detection
    result = hands.process(rgb_frame)

    # Draw Landmarks and Control Volume/Brightness
    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine if the hand is left or right
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            # Get landmarks for tips of all fingers
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Get landmarks for bases of fingers
            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]

            # Check if all other fingers are closed
            other_closed = (thumb_tip.y > index_base.y and
                            middle_tip.y > middle_base.y and
                            ring_tip.y > middle_base.y and
                            pinky_tip.y > middle_base.y)

            # VOLUME CONTROL with Right Hand
            if hand_label == 'Right' and right_hand_mode == 'volume':
                current_time = time.time()
                if current_time - prev_time > 0.2:  # Delay for smooth changes
                    # Check if index finger is pointing up (increase volume) and others are closed
                    if other_closed and index_tip.y < index_base.y:  # Pointing up
                        pyautogui.press('volumeup', presses=1)
                    # Check if index finger is pointing down (decrease volume) and others are closed
                    elif other_closed and index_tip.y > index_base.y:  # Pointing down
                        pyautogui.press('volumedown', presses=1)
                    prev_time = current_time

            # BRIGHTNESS CONTROL with Left Hand
            if hand_label == 'Left' and left_hand_mode == 'brightness':
                current_time = time.time()
                if current_time - prev_time > 0.2:  # Delay for smooth changes
                    # Check if index finger is pointing up (increase brightness) and others are closed
                    if other_closed and index_tip.y < index_base.y:  # Pointing up
                        pyautogui.press('brightnessup', presses=1)
                    # Check if index finger is pointing down (decrease brightness) and others are closed
                    elif other_closed and index_tip.y > index_base.y:  # Pointing down
                        pyautogui.press('brightnessdown', presses=1)
                    prev_time = current_time

            # Check for Korean Heart Gesture (Thumb and Index Pinch)
            thumb_to_index_dist = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            if thumb_to_index_dist < 0.03:  # Adjust threshold for pinch detection
                cv2.putText(frame, '❤️', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display Frame
    cv2.imshow('Hand Gesture AI', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
