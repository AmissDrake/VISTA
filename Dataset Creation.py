import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import string
import time

#——— Setup MediaPipe ———
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,               # only track one hand at a time
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ——— Labels & Parameters ———
right_labels = list(string.ascii_uppercase)   # A–Z
left_digits  = [str(i) for i in range(10)]
left_punct   = ['.', ',', '?']
left_labels  = left_digits + left_punct      # 0–9 + punctuation

SAMPLES = 150   # samples per class
PAUSE   = 0.5   # seconds between classes

# ——— Buffers ———
right_data = []
left_data  = []

def capture_phase(labels, buffer, window_title):
    """Capture SAMPLES of each label in labels list, append to buffer."""
    idx       = 0
    recording = False
    count     = 0

    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened() and idx < len(labels):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            label = labels[idx]
            # Instruction
            if not recording:
                text = f"[{window_title}] Press SPACE for '{label}'"
            else:
                text = f"[{window_title}] Recording '{label}' {count}/{SAMPLES}"
            cv2.putText(frame, text, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            # Capture when recording
            if recording and res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                coords = np.array([[p.x,p.y,p.z] for p in lm.landmark]).flatten().tolist()
                buffer.append([label] + coords)
                count += 1
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Advance when enough samples
            if recording and count >= SAMPLES:
                recording = False
                count     = 0
                idx      += 1
                time.sleep(PAUSE)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and not recording:      # SPACE to start
                print(f"> Starting {window_title} '{label}'")
                recording = True
            elif key == ord('q'):                # Q to quit early
                print("> Early exit requested.")
                break

    except KeyboardInterrupt:
        print("\n> Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ——— Run Phase 1: Right hand ———
print("Phase 1: Capture RIGHT hand (A–Z)")
capture_phase(right_labels, right_data, "Right-Hand Capture")

# ——— Run Phase 2: Left hand ———
print("Phase 2: Capture LEFT hand (0–9 + . , ?)")
capture_phase(left_labels, left_data, "Left-Hand Capture")

# ——— Save results ———
cols = ['label'] + [f'{c}_{i}' for i in range(21) for c in ('x','y','z')]
pd.DataFrame(right_data, columns=cols).to_csv('right_hand_dataset.csv', index=False)
pd.DataFrame(left_data,  columns=cols).to_csv('left_hand_dataset.csv',  index=False)
print(f"\nSaved {len(right_data)} right-hand samples and {len(left_data)} left-hand samples.")
