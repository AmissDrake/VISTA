import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from sklearn.linear_model import LinearRegression

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Settings
smoothing = 0.6
smoothed_vector = np.array([0.0, 0.0])
smoothed_tilt = 0.0

# Calibration points at screen edges and center
calib_points_screen = [
    (0, 0),                           # Top-left
    (screen_w // 2, 0),               # Top-center
    (screen_w - 1, 0),                # Top-right
    (screen_w - 1, screen_h // 2),    # Middle-right
    (screen_w - 1, screen_h - 1),     # Bottom-right
    (screen_w // 2, screen_h - 1),    # Bottom-center
    (0, screen_h - 1),                # Bottom-left
    (0, screen_h // 2),               # Middle-left
    (screen_w // 2, screen_h // 2)    # Center
]
calib_samples = []

def get_horizontal_gaze(landmarks, w):
    left_pupil = np.array([landmarks.landmark[468].x * w])
    left_inner = np.array([landmarks.landmark[133].x * w])
    left_outer = np.array([landmarks.landmark[33].x * w])
    eye_width = np.linalg.norm(left_inner - left_outer)
    if eye_width < 1e-5:
        return 0.0
    eye_center_x = (left_inner + left_outer) / 2
    return float((left_pupil - eye_center_x) / eye_width)

def get_vertical_openness(landmarks, h):
    upper_lid = landmarks.landmark[159].y * h
    lower_lid = landmarks.landmark[145].y * h
    eye_openness = lower_lid - upper_lid

    brow = landmarks.landmark[10].y * h
    chin = landmarks.landmark[152].y * h
    face_height = chin - brow
    if face_height < 1e-5:
        return 0.5
    return np.clip(eye_openness / face_height, 0, 1)

def get_head_tilt_degrees(landmarks, w, h):
    left_eye_outer = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
    right_eye_outer = np.array([landmarks.landmark[263].x * w, landmarks.landmark[263].y * h])
    delta = right_eye_outer - left_eye_outer
    angle_rad = np.arctan2(delta[1], delta[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def collect_calibration_data():
    print("Calibration starting...")
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for point in calib_points_screen:
        samples = []
        start_time = time.time()
        while time.time() - start_time < 2.0:  # Show point and collect data for 2 seconds
            # Create black full screen image
            frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            # Draw calibration circle
            cv2.circle(frame, (int(point[0]), int(point[1])), 30, (0, 255, 0), -1)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Calibration aborted by user.")
                cv2.destroyWindow("Calibration")
                cap.release()
                exit()

            # Capture face landmarks in this loop to collect samples
            ret, cam_frame = cap.read()
            if not ret:
                continue
            cam_frame = cv2.flip(cam_frame, 1)
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = cam_frame.shape
                horiz = get_horizontal_gaze(landmarks, w)
                vert = get_vertical_openness(landmarks, h)
                samples.append([horiz, vert])

        if samples:
            calib_samples.append(np.mean(samples, axis=0))
    cv2.destroyWindow("Calibration")

def fit_models():
    X = np.array(calib_samples)
    Y = np.array(calib_points_screen)
    model_x = LinearRegression().fit(X[:, [0]], Y[:, 0])
    model_y = LinearRegression().fit(X[:, [1]], Y[:, 1])
    return model_x, model_y

def main_loop(model_x, model_y):
    global smoothed_vector, smoothed_tilt
    fps = 0
    frame_count = 0
    prev_time = time.time()

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
            horiz = get_horizontal_gaze(landmarks, w)
            vert = get_vertical_openness(landmarks, h)
            new_vec = np.array([horiz, vert])
            smoothed_vector = smoothing * smoothed_vector + (1 - smoothing) * new_vec

            # Predict screen position
            x = np.clip(model_x.predict([[smoothed_vector[0]]])[0], 0, screen_w - 1)
            y = np.clip(model_y.predict([[smoothed_vector[1]]])[0], 0, screen_h - 1)
            pyautogui.moveTo(int(x), int(y))

            # Head tilt
            tilt_angle = get_head_tilt_degrees(landmarks, w, h)
            smoothed_tilt = smoothing * smoothed_tilt + (1 - smoothing) * tilt_angle

            # Visuals: Draw predicted gaze position on webcam feed scaled to webcam frame size
            cv2.circle(frame, (int(x * w / screen_w), int(y * h / screen_h)), 8, (0, 255, 255), -1)
            cv2.putText(frame, f"Tilt: {smoothed_tilt:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            eye_center = (left_eye + right_eye) / 2
            scale = 400
            gaze_end = eye_center + smoothed * scale
            cv2.line(frame,
                     tuple(eye_center.astype(int)),
                     tuple(gaze_end.astype(int)),
                     (0, 255, 255), 2)

        # FPS overlay
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
if calib_samples:
    model_x, model_y = fit_models()
    print("Calibration complete.")
    main_loop(model_x, model_y)
else:
    print("Calibration failed.")
