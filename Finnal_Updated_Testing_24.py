import cv2
import numpy as np
import mediapipe as mp
import joblib
import time
import pyautogui

def normalize_lm(flat):
    lm = flat.reshape(21, 3)
    wrist = lm[0].copy()
    lm -= wrist
    norm = np.linalg.norm(lm)
    return (lm / norm if norm > 0 else lm).flatten()

def normalize_dual_lm(flat):
    lm1 = flat[:63].reshape(21, 3)
    lm2 = flat[63:].reshape(21, 3)
    wrist1 = lm1[0].copy()
    wrist2 = lm2[0].copy()
    lm1 -= wrist1
    lm2 -= wrist2
    norm1 = np.linalg.norm(lm1)
    norm2 = np.linalg.norm(lm2)
    norm1 = norm1 if norm1 > 0 else 1
    norm2 = norm2 if norm2 > 0 else 1
    norm1_lm = lm1 / norm1
    norm2_lm = lm2 / norm2
    return np.concatenate([norm1_lm.flatten(), norm2_lm.flatten()])

# Load models
left_clf = joblib.load('left_hand_model.pkl')
right_clf = joblib.load('right_hand_model.pkl')
dual_clf = joblib.load('dual_hand_model.pkl')

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Starting hand-gesture keyboard. Press 'q' to quit.")

typed_text = ""
last_prediction = ""
last_time = time.time()
delay = 2.0
font = cv2.FONT_HERSHEY_DUPLEX

capslock_state = False
numlock_state = False

shifted_number_map = {
    '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
    '6': '^', '7': '&', '8': '*', '9': '(', '0': ')'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    current_time = time.time()
    gesture_text = ""
    confidence = 0
    pred = ""

    if res.multi_hand_landmarks and res.multi_handedness:
        hands_detected = {'Left': None, 'Right': None}
        for i, handedness in enumerate(res.multi_handedness):
            label = handedness.classification[0].label
            hands_detected[label] = res.multi_hand_landmarks[i]

        if hands_detected['Left'] is not None and hands_detected['Right'] is not None:
            lm_left = hands_detected['Left']
            lm_right = hands_detected['Right']
            coords_left = np.array([[p.x, p.y, p.z] for p in lm_left.landmark]).flatten()
            coords_right = np.array([[p.x, p.y, p.z] for p in lm_right.landmark]).flatten()
            combined = np.concatenate([coords_left, coords_right])
            norm = normalize_dual_lm(combined)

            pred = dual_clf.predict([norm])[0]
            confidence = np.max(dual_clf.predict_proba([norm])) if hasattr(dual_clf, 'predict_proba') else 1.0
            gesture_text = f"Dual: '{pred}' ({confidence*100:.1f}%)"

            mp_drawing.draw_landmarks(frame, lm_left, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, lm_right, mp_hands.HAND_CONNECTIONS)

        else:
            for side in ['Left', 'Right']:
                lm = hands_detected[side]
                if lm is not None:
                    flat = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
                    norm = normalize_lm(flat)

                    clf = left_clf if side == 'Left' else right_clf
                    pred = clf.predict([norm])[0]
                    confidence = np.max(clf.predict_proba([norm])) if hasattr(clf, 'predict_proba') else 1.0
                    gesture_text = f"{side}: '{pred}' ({confidence*100:.1f}%)"

                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    break

    if pred:
        if pred != last_prediction:
            last_time = current_time
            last_prediction = pred
        else:
            if (current_time - last_time) >= delay:
                last_time = current_time

                if pred == 'spellcheck':
                    pass  # No operation
                elif pred == 'leftclick':
                    pyautogui.click(button='left')
                elif pred == 'rightclick':
                    pyautogui.click(button='right')
                elif pred == 'backspace':
                    pyautogui.press('backspace')
                    typed_text = typed_text[:-1] if typed_text else ''
                elif pred == 'space':
                    pyautogui.press('space')
                    typed_text += ' '
                elif pred == 'enter':
                    pyautogui.press('enter')
                elif pred == 'capslock':
                    capslock_state = not capslock_state
                elif pred == 'numlock':
                    numlock_state = not numlock_state
                elif pred in [',', '.', '?']:
                    pyautogui.press(pred)
                    typed_text += pred
                else:
                    output_char = pred
                    if output_char.isalpha():
                        if capslock_state:
                            pyautogui.write(output_char.upper())
                            typed_text += output_char.upper()
                        else:
                            pyautogui.write(output_char.lower())
                            typed_text += output_char.lower()
                    elif output_char in shifted_number_map:
                        if numlock_state:
                            pyautogui.keyDown('shift')
                            pyautogui.write(shifted_number_map[output_char])
                            pyautogui.keyUp('shift')
                            typed_text += shifted_number_map[output_char]
                        else:
                            pyautogui.write(output_char)
                            typed_text += output_char
                    else:
                        pyautogui.write(output_char)
                        typed_text += output_char

    # --- UI Overlay ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, f"Typed: {typed_text[-50:]}", (10, 65), font, 1.1, (255, 255, 255), 2)

    caps_text = f"CapsLock: {'ON' if capslock_state else 'OFF'}"
    num_text = f"NumLock: {'ON' if numlock_state else 'OFF'}"
    caps_color = (100, 255, 100) if capslock_state else (180, 180, 180)
    num_color = (100, 255, 100) if numlock_state else (180, 180, 180)

    cv2.putText(frame, caps_text, (w - 220, 35), font, 0.8, caps_color, 2)
    cv2.putText(frame, num_text, (w - 220, 75), font, 0.8, num_color, 2)

    if gesture_text:
        cv2.putText(frame, gesture_text, (10, 30), font, 0.8, (100, 255, 255), 2)

    cv2.imshow("Hand Gesture Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
