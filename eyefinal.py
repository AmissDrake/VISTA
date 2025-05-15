import cv2
import mediapipe as mp
import time
import psutil
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Drawing specifications
draw_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

# Landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
PUPIL_CENTER_IDX = [468, 473]  # Approximate center points for both eyes

# Performance tracking variables
prev_time = time.time()
frame_count = 0
pupil_positions = []
fps = 0
jitter = 0.0
ram = 0.0

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB ( mediapipe works on rgb)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Get frame dimensions
    h, w, _ = frame.shape

    # Process landmarks if a face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw pupils
            for idx in PUPIL_CENTER_IDX:
                lm = face_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                pupil_positions.append((cx, cy))
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # Draw eye outlines
            for eye in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                for idx in eye:
                    lm = face_landmarks.landmark[idx]
                    ex, ey = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (ex, ey), 1, (255, 0, 0), -1)

    # Update FPS every second and calculate performance stats
    frame_count += 1
    now = time.time()
    if now - prev_time >= 1:
        fps = frame_count
        frame_count = 0
        prev_time = now
        ram = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
        if len(pupil_positions) >= 5:
            xs, ys = zip(*pupil_positions[-20:])
            jitter = np.std(xs) + np.std(ys)
        else:
            jitter = 0.0

    # Display FPS, RAM, and Jitter on screen
    info_text = [
        f"FPS: {fps}",
        f"RAM: {ram:.2f} MB",
        f"Jitter: {jitter:.2f}"
    ]
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("MediaPipe Eye Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
