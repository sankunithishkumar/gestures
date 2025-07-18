import cv2
import dlib
import pyautogui
import math
import time
import webbrowser
from imutils import face_utils
import mediapipe as mp

# === Setup ===
predictor_path = r"C:\Users\siddartha\OneDrive\Apps\Designer\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# Eye Blink Config
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 2
blink_counter = 0
last_blink_time = 0

# Head Movement Config
head_last_y = None
HEAD_MOVEMENT_THRESHOLD = 15
head_movement_start_time = None

# Mouse setup
screen_w, screen_h = pyautogui.size()
click_delay = 0.6
last_click_time = 0

# Gesture Delay
prev_time = time.time()
gesture_delay = 1.5

# Fist hold setup
fist_hold_start = None
fist_hold_duration = 3  # seconds

# Capture
cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    rects = detector(gray, 0)
    current_time = time.time()

    # === Eye Blink & Head Nod ===
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)
        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                if time.time() - last_blink_time < 1.0:
                    pyautogui.hotkey("alt", "f4")
                    print("🟡 Double Blink — Close Window")
                last_blink_time = time.time()
            blink_counter = 0

        nose_y = shape_np[33][1]
        if head_last_y is not None:
            delta_y = nose_y - head_last_y
            if abs(delta_y) > HEAD_MOVEMENT_THRESHOLD:
                if head_movement_start_time is None:
                    head_movement_start_time = time.time()
                elif time.time() - head_movement_start_time < 0.7:
                    pyautogui.hotkey("win", "down")
                    print("🟠 Head Nod — Minimize")
                    head_movement_start_time = None
        head_last_y = nose_y

    # === Hand Gestures & Mouse Control ===
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_idx, handLms in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[hand_idx].classification[0].label
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            fingers = count_fingers(handLms)
            total = fingers.count(1)
            lm = handLms.landmark
            index_tip = lm[8]
            center_x = lm[0].x

            # Mouse move
            x, y = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            pyautogui.moveTo(x, y, duration=0.01)

            # Clicks by finger config
            if current_time - last_click_time > click_delay:
                if fingers == [0, 1, 0, 0, 0]:
                    pyautogui.click(button='left')
                    print("🖱️ Left Click (Index Finger)")
                    last_click_time = current_time
                elif fingers == [0, 0, 1, 0, 0]:
                    pyautogui.click(button='right')
                    print("🖱️ Right Click (Middle Finger)")
                    last_click_time = current_time
                elif fingers == [0, 1, 1, 0, 0]:
                    pyautogui.doubleClick()
                    print("🖱️ Double Click (Index + Middle)")
                    last_click_time = current_time

            # === Fist Hold to Minimize ===
            if fingers == [0, 0, 0, 0, 0]:
                if fist_hold_start is None:
                    fist_hold_start = current_time
                elif current_time - fist_hold_start >= fist_hold_duration:
                    pyautogui.hotkey("win", "down")
                    print("✊ Fist held — Minimize Window")
                    fist_hold_start = None  # reset
            else:
                fist_hold_start = None  # reset if not fist

            # Other gestures
            if current_time - prev_time > gesture_delay:
                if fingers == [0, 1, 0, 0, 0]:
                    pyautogui.scroll(300)
                    print("⬆️ Scroll Up")
                elif total == 5:
                    pyautogui.scroll(-300)
                    print("⬇️ Scroll Down")
                elif fingers == [1, 0, 0, 0, 1]:
                    webbrowser.open("https://www.google.com")
                    print("🌐 Open Browser")
                elif fingers == [1, 1, 0, 0, 1]:
                    pyautogui.hotkey("win", "up")
                    print("🟢 Maximize Window")
                elif fingers == [0, 1, 0, 0, 0] and center_x < 0.4:
                    pyautogui.press("right")
                    print("➡️ Next Slide")
                elif fingers == [0, 1, 0, 0, 0] and center_x > 0.6:
                    pyautogui.press("left")
                    print("⬅️ Previous Slide")
                prev_time = current_time

    cv2.imshow("🖱️ Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
