import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from sklearn.linear_model import LinearRegression

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Gaze smoothing and deadzone
smoothing = 0.7
deadzone = 0.02
smoothed_gaze_vector = np.array([0.0, 0.0])

# Calibration target points on screen
calib_points_screen = [(screen_w * x, screen_h * y) for x in [0.1, 0.5, 0.9] for y in [0.1, 0.5, 0.9]]
calib_gaze_vectors = []

def get_gaze_vector(landmarks, w, h):
    eye_data = [
        (468, 133, 33, 159, 145),  # Left: pupil, inner, outer, top, bottom
        (473, 362, 263, 386, 374)  # Right: pupil, inner, outer, top, bottom
    ]
    vectors = []

    for pupil_idx, in_idx, out_idx, top_idx, bot_idx in eye_data:
        pupil = np.array([landmarks.landmark[pupil_idx].x * w,
                          landmarks.landmark[pupil_idx].y * h])
        eye_center = np.array([
            (landmarks.landmark[in_idx].x + landmarks.landmark[out_idx].x) * w / 2,
            (landmarks.landmark[top_idx].y + landmarks.landmark[bot_idx].y) * h / 2
        ])

        eye_width = np.linalg.norm([
            landmarks.landmark[in_idx].x - landmarks.landmark[out_idx].x
        ]) * w

        eye_height = np.linalg.norm([
            landmarks.landmark[top_idx].y - landmarks.landmark[bot_idx].y
        ]) * h

        if eye_width < 1e-5 or eye_height < 1e-5:
            continue

        norm_vec = np.array([
            (pupil[0] - eye_center[0]) / eye_width,
            (pupil[1] - eye_center[1]) / eye_height
        ])
        vectors.append(norm_vec)

    if len(vectors) != 2:
        return np.array([0.0, 0.0])

    gaze_vector = np.mean(vectors, axis=0)
    gaze_vector = np.where(np.abs(gaze_vector) < deadzone, 0, gaze_vector)
    return gaze_vector


def collect_calibration_data():
    print("Starting calibration. Please follow the green dots.")

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for point in calib_points_screen:
        samples = []

        # Show target dot for 1 second
        start_time = time.time()
        while time.time() - start_time < 1.0:
            calib_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(calib_frame, (int(point[0]), int(point[1])), 30, (0, 255, 0), -1)
            cv2.imshow(window_name, calib_frame)
            cv2.waitKey(1)

        # Capture data for 1 second
        start_time = time.time()
        while time.time() - start_time < 1.0:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = frame.shape
                vec = get_gaze_vector(landmarks, w, h)
                samples.append(vec)

        if samples:
            avg_vec = np.mean(samples, axis=0)
            calib_gaze_vectors.append(avg_vec)

    cv2.destroyWindow(window_name)

def fit_models():
    X = np.array(calib_gaze_vectors)
    Y = np.array(calib_points_screen)
    model_x = LinearRegression().fit(X, Y[:, 0])
    model_y = LinearRegression().fit(X, Y[:, 1])
    return model_x, model_y

def main_loop(model_x, model_y):
    fps = 0
    frame_count = 0
    prev_time = time.time()
    global smoothed_gaze_vector

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            vec = get_gaze_vector(landmarks, w, h)
            smoothed = smoothing * smoothed_gaze_vector + (1 - smoothing) * vec
            smoothed_gaze_vector[:] = smoothed

            # Predict screen position
            pred_x = np.clip(model_x.predict([smoothed])[0], 0, screen_w - 1)
            pred_y = np.clip(model_y.predict([smoothed])[0], 0, screen_h - 1)

            # Instantly reposition with slight delay to avoid shakiness
            pyautogui.moveTo(int(pred_x), int(pred_y), duration=0.05)

            # Draw prediction for debug
            cv2.circle(frame, (int(pred_x * w / screen_w), int(pred_y * h / screen_h)), 8, (0, 255, 255), -1)

        # FPS display
        frame_count += 1
        now = time.time()
        if now - prev_time >= 1:
            fps = frame_count
            frame_count = 0
            prev_time = now
        cv2.putText(frame, f"FPS: {fps}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Main ===
collect_calibration_data()
if calib_gaze_vectors:
    model_x, model_y = fit_models()
    print("Calibration complete.")
    main_loop(model_x, model_y)
else:
    print("Calibration failed. Try again.")
