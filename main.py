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
hold_start_time = None
menu_options = ["Volume Control", "Drawing"]
canvas_initialized = False
canvas = None

# Open Webcam
cap = cv2.VideoCapture(0)

def check_hover_and_hold(index_x, index_y, x1, y1, x2, y2, hold_start_time):
    """Check if the index finger is hovering over a button and handle holding logic."""
    if x1 < index_x < x2 and y1 < index_y < y2:
        if hold_start_time is None:
            return time.time()
        elif time.time() - hold_start_time > 1:  # Hold for 3 seconds
            return "selected"
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip image for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Hand Detection
    result = hands.process(rgb_frame)

    height, width, _ = frame.shape

    # Menu
    if mode == "menu":
        for i, option in enumerate(menu_options):
            y_position = 100 + i * 50
            cv2.putText(frame, f"{i + 1}. {option}", (50, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

                # Draw circle on index finger tip
                cv2.circle(frame, (index_x, index_y), 10, (255, 0, 0), -1)

                for i, option in enumerate(menu_options):
                    y_position = 100 + i * 50
                    result = check_hover_and_hold(index_x, index_y, 50, y_position - 30, 300, y_position + 10, hold_start_time)
                    if result == "selected":
                        mode = option.lower().replace(" ", "_")
                        hold_start_time = None
                        break
                    elif result:
                        hold_start_time = result

    # Volume Control Mode
    elif mode == "volume_control":
        cv2.putText(frame, "Volume Control Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (500, 20), (650, 70), (0, 255, 0), -1)
        cv2.putText(frame, "Back", (510, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                index_base = hand_landmarks.landmark[5]
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
                index_tip_y = int(index_tip.y * height)
                index_base_y = int(index_base.y * height)

                # Check for "Back" button hover
                result = check_hover_and_hold(index_x, index_y, 500, 20, 650, 70, hold_start_time)
                if result == "selected":
                    mode = "menu"
                    hold_start_time = None
                    break
                elif result:
                    hold_start_time = result

                # Volume control logic
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
        cv2.rectangle(frame, (500, 20), (650, 70), (0, 255, 0), -1)
        cv2.putText(frame, "Erase All", (510, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (500, 90), (650, 140), (0, 255, 0), -1)
        cv2.putText(frame, "Back", (510, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)

                # Draw circle where index finger is
                cv2.circle(canvas, (index_x, index_y), 5, (0, 255, 0), -1)

                # Check for "Erase All" button hover
                result = check_hover_and_hold(index_x, index_y, 500, 20, 650, 70, hold_start_time)
                if result == "selected":
                    canvas = np.zeros_like(frame)
                    hold_start_time = None

                # Check for "Back" button hover
                result = check_hover_and_hold(index_x, index_y, 500, 90, 650, 140, hold_start_time)
                if result == "selected":
                    mode = "menu"
                    hold_start_time = None
                elif result:
                    hold_start_time = result

        # Blend canvas with original frame
        frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

    # Display Frame
    cv2.imshow('Hand Gesture AI', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
