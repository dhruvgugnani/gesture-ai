import cv2
import mediapipe as mp
import numpy as np
import time
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
COLOR_THEME = {
    "primary": (0, 150, 255),
    "secondary": (255, 50, 50),
    "accent": (50, 255, 50),
    "text": (240, 240, 240)
}

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()

# Global state with realistic proportions
app_state = {
    "mode": "menu",
    "prev_mode": "menu",
    "hold_start": None,
    "puppet": {
        "joints": {
            "head": {"pos": (SCREEN_WIDTH // 2, 100), "vel": (0, 0)},
            "neck": {"pos": (SCREEN_WIDTH // 2, 130), "vel": (0, 0)},
            "left_shoulder": {"pos": (SCREEN_WIDTH // 2 - 50, 150), "vel": (0, 0)},
            "right_shoulder": {"pos": (SCREEN_WIDTH // 2 + 50, 150), "vel": (0, 0)},
            "left_elbow": {"pos": (SCREEN_WIDTH // 2 - 70, 220), "vel": (0, 0)},
            "right_elbow": {"pos": (SCREEN_WIDTH // 2 + 70, 220), "vel": (0, 0)},
            "left_wrist": {"pos": (SCREEN_WIDTH // 2 - 80, 300), "vel": (0, 0)},
            "right_wrist": {"pos": (SCREEN_WIDTH // 2 + 80, 300), "vel": (0, 0)},
            "pelvis": {"pos": (SCREEN_WIDTH // 2, 180), "vel": (0, 0)},
            "left_hip": {"pos": (SCREEN_WIDTH // 2 - 30, 200), "vel": (0, 0)},
            "right_hip": {"pos": (SCREEN_WIDTH // 2 + 30, 200), "vel": (0, 0)},
            "left_knee": {"pos": (SCREEN_WIDTH // 2 - 30, 300), "vel": (0, 0)},
            "right_knee": {"pos": (SCREEN_WIDTH // 2 + 30, 300), "vel": (0, 0)},
            "left_ankle": {"pos": (SCREEN_WIDTH // 2 - 30, 400), "vel": (0, 0)},
            "right_ankle": {"pos": (SCREEN_WIDTH // 2 + 30, 400), "vel": (0, 0)}
        },
        "attraction_points": []
    },
    "draw": {
        "canvas": np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8),
        "color": (0, 255, 255),
        "brush_size": 10,
        "prev_point": None,
        "erase_progress": 0
    }
}

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

def draw_button(frame, text, pos, size, hover=False):
    """Draw interactive button with hover effect"""
    color = COLOR_THEME["accent"] if hover else COLOR_THEME["primary"]
    cv2.rectangle(frame, pos, (pos[0] + size[0], pos[1] + size[1]), color, -1)
    cv2.rectangle(frame, pos, (pos[0] + size[0], pos[1] + size[1]), (255, 255, 255), 2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = pos[0] + (size[0] - text_size[0]) // 2
    text_y = pos[1] + (size[1] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_THEME["text"], 2)

def update_volume(hand_landmarks):
    """Volume control using index finger direction"""
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Check if other fingers are closed
    fingers_closed = (middle_tip.y > hand_landmarks.landmark[9].y and
                     ring_tip.y > hand_landmarks.landmark[13].y and
                     pinky_tip.y > hand_landmarks.landmark[17].y)

    if not fingers_closed:
        return None

    # Calculate vertical direction
    vol_change = 0
    if index_tip.y < wrist.y - 0.15:  # Finger up
        vol_change = 0.02
    elif index_tip.y > wrist.y + 0.15:  # Finger down
        vol_change = -0.02

    if vol_change != 0:
        current_vol = volume.GetMasterVolumeLevelScalar()
        new_vol = np.clip(current_vol + vol_change, 0, 1)
        volume.SetMasterVolumeLevelScalar(new_vol, None)
        return new_vol
    return None

def smooth_drawing(current_point, prev_point):
    """Create smooth lines with linear interpolation"""
    if prev_point is None:
        return

    x1, y1 = prev_point
    x2, y2 = current_point
    distance = hypot(x2 - x1, y2 - y1)

    if distance > 0:
        steps = int(distance / 5) + 1
        for t in np.linspace(0, 1, steps):
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            cv2.circle(app_state["draw"]["canvas"], (x, y),
                       app_state["draw"]["brush_size"],
                       app_state["draw"]["color"], -1)

def update_puppet(hand_landmarks):
    """Enhanced puppet physics with realistic proportions"""
    GRAVITY = (0, 0.2)
    STIFFNESS = 2.0
    DAMPING = 0.9
    DT = 0.5
    FLOOR = SCREEN_HEIGHT - 50

    # Update control points
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]
    app_state["puppet"]["attraction_points"] = [
        (int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT)),
        (int(thumb_tip.x * SCREEN_WIDTH), int(thumb_tip.y * SCREEN_HEIGHT))
    ]

    for _ in range(3):
        # Apply physics
        for joint in app_state["puppet"]["joints"].values():
            joint["vel"] = (
                joint["vel"][0] * DAMPING,
                joint["vel"][1] * DAMPING + GRAVITY[1]
            )
            joint["pos"] = (
                joint["pos"][0] + joint["vel"][0] * DT,
                joint["pos"][1] + joint["vel"][1] * DT
            )

            # Floor collision
            if joint["pos"][1] > FLOOR:
                joint["pos"] = (joint["pos"][0], FLOOR)
                joint["vel"] = (joint["vel"][0], -joint["vel"][1] * 0.5)

        apply_body_constraints()
        apply_attraction_forces()

def apply_body_constraints():
    """Realistic body constraints"""
    joints = app_state["puppet"]["joints"]
    constraints = [
        ("head", "neck", 25),
        ("neck", "left_shoulder", 40),
        ("neck", "right_shoulder", 40),
        ("left_shoulder", "left_elbow", 60),
        ("left_elbow", "left_wrist", 70),
        ("right_shoulder", "right_elbow", 60),
        ("right_elbow", "right_wrist", 70),
        ("neck", "pelvis", 80),
        ("pelvis", "left_hip", 30),
        ("left_hip", "left_knee", 80),
        ("left_knee", "left_ankle", 90),
        ("pelvis", "right_hip", 30),
        ("right_hip", "right_knee", 80),
        ("right_knee", "right_ankle", 90)
    ]

    for a, b, dist in constraints:
        dx = joints[b]["pos"][0] - joints[a]["pos"][0]
        dy = joints[b]["pos"][1] - joints[a]["pos"][1]
        current_dist = hypot(dx, dy)
        if current_dist == 0:
            continue

        correction = (dist - current_dist) / (current_dist * 2)
        joints[a]["pos"] = (
            joints[a]["pos"][0] - dx * correction,
            joints[a]["pos"][1] - dy * correction
        )
        joints[b]["pos"] = (
            joints[b]["pos"][0] + dx * correction,
            joints[b]["pos"][1] + dy * correction
        )

def apply_attraction_forces():
    """Apply smooth attraction to control points"""
    head = app_state["puppet"]["joints"]["head"]
    left_wrist = app_state["puppet"]["joints"]["left_wrist"]
    points = app_state["puppet"]["attraction_points"]

    if len(points) >= 2:
        # Head follows index finger
        dx = points[0][0] - head["pos"][0]
        dy = points[0][1] - head["pos"][1]
        head["vel"] = (
            head["vel"][0] + dx * 0.12,
            head["vel"][1] + dy * 0.12
        )

        # Left wrist follows thumb
        dx = points[1][0] - left_wrist["pos"][0]
        dy = points[1][1] - left_wrist["pos"][1]
        left_wrist["vel"] = (
            left_wrist["vel"][0] + dx * 0.08,
            left_wrist["vel"][1] + dy * 0.08
        )

def draw_puppet(frame):
    """Draw realistic stick figure with joints"""
    joints = app_state["puppet"]["joints"]
    connections = [
        ("head", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("neck", "pelvis"), ("pelvis", "left_hip"), ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"), ("pelvis", "right_hip"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]

    # Draw bones
    for a, b in connections:
        cv2.line(frame,
                 tuple(map(int, joints[a]["pos"])),
                 tuple(map(int, joints[b]["pos"])),
                 (255, 255, 255), 3)

    # Draw joints
    for joint in joints.values():
        pos = tuple(map(int, joint["pos"]))
        cv2.circle(frame, pos, 6, (0, 150, 255), -1)
        cv2.circle(frame, pos, 8, (255, 255, 255), 2)

    # Draw control strings
    if app_state["puppet"]["attraction_points"]:
        cv2.line(frame, tuple(map(int, joints["head"]["pos"])),
                 app_state["puppet"]["attraction_points"][0],
                 (100, 100, 100), 2)

def draw_tools_palette(frame):
    """Vertical tool panels on edges"""
    # Color palette (left)
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255)]
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (20, 100 + i * 80), (70, 150 + i * 80), color, -1)

    # Brush sizes (right)
    sizes = [5, 10, 20, 30]
    for i, size in enumerate(sizes):
        cv2.circle(frame, (SCREEN_WIDTH - 50, 150 + i * 80), size, (255, 255, 255), -1)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Flip and resize frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    results = hands.process(rgb_frame)

    # Clear canvas when switching modes
    if app_state["prev_mode"] == "draw" and app_state["mode"] != "draw":
        app_state["draw"]["canvas"] = np.zeros_like(app_state["draw"]["canvas"])
    app_state["prev_mode"] = app_state["mode"]

    # Prepare fresh frame
    display_frame = frame.copy()

    # Draw persistent canvas
    if app_state["mode"] == "draw":
        display_frame = cv2.add(display_frame, app_state["draw"]["canvas"])

    # UI Elements
    modes = ["Menu", "Volume", "Draw", "Puppet"]
    for i, mode in enumerate(modes):
        btn_pos = (40 + i * 200, 40)
        active = mode.lower() == app_state["mode"]
        draw_button(display_frame, mode, btn_pos, (160, 50), active)

    # Draw tools panel
    if app_state["mode"] == "draw":
        draw_tools_palette(display_frame)
        draw_button(display_frame, "Erase", (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 150), (150, 50))

    if results.multi_hand_landmarks:
        for hand_index, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                display_frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=COLOR_THEME["primary"], thickness=2),
                mp_drawing.DrawingSpec(color=COLOR_THEME["accent"], thickness=2)
            )

            index_tip = hand.landmark[8]
            ix, iy = int(index_tip.x * SCREEN_WIDTH), int(index_tip.y * SCREEN_HEIGHT)

            # Mode switching
            for i, mode in enumerate(modes):
                btn_pos = (40 + i * 200, 40)
                if btn_pos[0] < ix < btn_pos[0] + 160 and btn_pos[1] < iy < btn_pos[1] + 50:
                    if app_state["hold_start"] is None:
                        app_state["hold_start"] = time.time()
                    elif time.time() - app_state["hold_start"] > 1:
                        app_state["mode"] = mode.lower()
                        app_state["hold_start"] = None
                    break
            else:
                app_state["hold_start"] = None

            # Mode functionality
            if app_state["mode"] == "volume" and hand_index == 0:
                vol_level = update_volume(hand)
                if vol_level is not None:
                    cv2.rectangle(display_frame, (100, 100), (300, 130), COLOR_THEME["primary"], 2)
                    cv2.rectangle(display_frame, (100, 100),
                                  (100 + int(200 * vol_level), 130),
                                  COLOR_THEME["accent"], -1)

            elif app_state["mode"] == "draw":
                # Tool selection
                colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255)]
                for i, color in enumerate(colors):
                    if 20 < ix < 70 and 100 + i * 80 < iy < 150 + i * 80:
                        app_state["draw"]["color"] = color

                sizes = [5, 10, 20, 30]
                for i, size in enumerate(sizes):
                    if SCREEN_WIDTH - 50 - size < ix < SCREEN_WIDTH - 50 + size and 150 + i * 80 - size < iy < 150 + i * 80 + size:
                        app_state["draw"]["brush_size"] = size

                # Drawing
                smooth_drawing((ix, iy), app_state["draw"]["prev_point"])
                app_state["draw"]["prev_point"] = (ix, iy)

                # Erase functionality
                if SCREEN_WIDTH - 200 < ix < SCREEN_WIDTH - 50 and SCREEN_HEIGHT - 150 < iy < SCREEN_HEIGHT - 100:
                    app_state["draw"]["erase_progress"] += 1 / 30
                    if app_state["draw"]["erase_progress"] >= 1:
                        app_state["draw"]["canvas"] = np.zeros_like(app_state["draw"]["canvas"])
                        app_state["draw"]["erase_progress"] = 0
                    cv2.rectangle(display_frame, (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 150),
                                  (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 100), COLOR_THEME["secondary"], 2)
                    cv2.rectangle(display_frame, (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 150),
                                  (SCREEN_WIDTH - 200 + int(150 * app_state["draw"]["erase_progress"]),
                                   SCREEN_HEIGHT - 100),
                                  COLOR_THEME["secondary"], -1)
                else:
                    app_state["draw"]["erase_progress"] = 0

            elif app_state["mode"] == "puppet" and hand_index == 1:
                update_puppet(hand)
                draw_puppet(display_frame)

    # Show hover progress
    if app_state["hold_start"] is not None and results.multi_hand_landmarks:
        progress = (time.time() - app_state["hold_start"]) / 1
        cv2.circle(display_frame, (ix, iy), int(30 * progress), COLOR_THEME["accent"], 2)

    cv2.imshow("Gesture Control Suite", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
