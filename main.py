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
menu_options = ["Volume Control", "Drawing", "Puppet"]
canvas_initialized = False
canvas = None

# Puppet properties
puppet_positions = {
    "head": (300, 200),
    "left_hand": (250, 300),
    "right_hand": (350, 300),
    "left_leg": (270, 400),
    "right_leg": (330, 400),
}

# Open Webcam
cap = cv2.VideoCapture(0)

def check_hover_and_hold(index_x, index_y, x1, y1, x2, y2, hold_start_time):
    """Check if the index finger is hovering over a button and handle holding logic."""
    if x1 < index_x < x2 and y1 < index_y < y2:
        if hold_start_time is None:
            return time.time()
        elif time.time() - hold_start_time > 1:  # Hold for 1 second
            return "selected"
    return None

def draw_puppet(frame, puppet_positions, index_pos, thumb_pos):
    """Draw the puppet on the frame based on joint positions."""
    # Draw strings (ropes)
    cv2.line(frame, index_pos, puppet_positions["head"], (41.246, 21.267, 1.933), 2)  # Head rope
    cv2.line(frame, thumb_pos, puppet_positions["left_hand"], (41.246, 21.267, 1.933), 2)  # Left hand rope
    cv2.line(frame, puppet_positions["head"], puppet_positions["right_hand"], (41.246, 21.267, 1.933), 2)  # Right hand rope

    # Draw body parts
    cv2.circle(frame, puppet_positions["head"], 30, (0, 255, 255), -1)

    # Add smiley face to head
    center = puppet_positions["head"]
    cv2.circle(frame, (center[0] - 10, center[1] - 10), 5, (0, 0, 0), -1)  # Left eye
    cv2.circle(frame, (center[0] + 10, center[1] - 10), 5, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(frame, (center[0], center[1] + 10), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Smiling mouth

    # Draw limbs and connections
    cv2.line(frame, puppet_positions["head"], puppet_positions["left_hand"], (255, 255, 255), 4)
    cv2.line(frame, puppet_positions["head"], puppet_positions["right_hand"], (255, 255, 255), 4)
    cv2.line(frame, puppet_positions["head"], ((puppet_positions["left_leg"][0] + puppet_positions["right_leg"][0]) // 2, puppet_positions["left_leg"][1] - 30), (255, 255, 255), 4)
    cv2.line(frame, ((puppet_positions["left_leg"][0] + puppet_positions["right_leg"][0]) // 2, puppet_positions["left_leg"][1] - 30), puppet_positions["left_leg"], (255, 255, 255), 4)
    cv2.line(frame, ((puppet_positions["left_leg"][0] + puppet_positions["right_leg"][0]) // 2, puppet_positions["left_leg"][1] - 30), puppet_positions["right_leg"], (255, 255, 255), 4)
    cv2.circle(frame, puppet_positions["left_hand"], 15, (0, 255, 0), -1)
    cv2.circle(frame, puppet_positions["right_hand"], 15, (0, 255, 0), -1)
    cv2.circle(frame, puppet_positions["left_leg"], 15, (255, 0, 0), -1)
    cv2.circle(frame, puppet_positions["right_leg"], 15, (255, 0, 0), -1)

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
                cv2.circle(canvas, (index_x, index_y), 15, (0, 255, 255), -1)

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

        # Combine canvas with frame
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Puppet Mode
    elif mode == "puppet":
        cv2.putText(frame, "Puppet Mode", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (500, 20), (650, 70), (0, 255, 0), -1)
        cv2.putText(frame, "Back", (510, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                index_x, index_y = int(index_tip.x * width), int(index_tip.y * height)
                thumb_x, thumb_y = int(thumb_tip.x * width), int(thumb_tip.y * height)

                string_length = 70

                puppet_positions["head"] = (index_x, index_y + string_length)
                puppet_positions["left_hand"] = (thumb_x - 30, thumb_y + 30)
                puppet_positions["right_hand"] = (puppet_positions["head"][0] + 50, puppet_positions["head"][1] + 50)
                puppet_positions["left_leg"] = (
                puppet_positions["head"][0] - 30, puppet_positions["head"][1] + 100 + int(10 * np.sin(time.time())))
                puppet_positions["right_leg"] = (
                puppet_positions["head"][0] + 30, puppet_positions["head"][1] + 100 - int(10 * np.sin(time.time())))

                # Draw the updated puppet
                draw_puppet(frame, puppet_positions, (index_x, index_y), (thumb_x, thumb_y))

                # Check for "Back" button hover
                result = check_hover_and_hold(index_x, index_y, 500, 20, 650, 70, hold_start_time)
                if result == "selected":
                    mode = "menu"
                    hold_start_time = None
                elif result:
                    hold_start_time = result

    cv2.imshow("Hand Gesture AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
