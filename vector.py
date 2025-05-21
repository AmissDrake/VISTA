import cv2
import mediapipe as mp
import numpy as np
import time

#mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)    

#fps
fps=0
prev_time=time.time()
frame_count=0

cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(frame_rgb)
    h,w,z=frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            #landmarks
            left_pupil=face_landmarks.landmark[468]
            right_pupil=face_landmarks.landmark[473]

            #corners of eyes
            left_in_cr=face_landmarks.landmark[133]
            left_out_cr=face_landmarks.landmark[33]
            right_in_cr=face_landmarks.landmark[362]
            right_out_cr=face_landmarks.landmark[263]

            #convert to pixel co-ords
            left_pupil_px = np.array([int(left_pupil.x * w), int(left_pupil.y * h)])
            right_pupil_px = np.array([int(right_pupil.x * w), int(right_pupil.y * h)])

            left_eye_center = ((int(left_in_cr.x * w) + int(left_out_cr.x * w)) // 2,
                               (int(left_in_cr.y * h) + int(left_out_cr.y * h)) // 2)

            right_eye_center = ((int(right_in_cr.x * w) + int(right_out_cr.x * w)) // 2,
                                (int(right_in_cr.y * h) + int(right_out_cr.y * h)) // 2)
            
            # Draw iris centers and eye centers
            cv2.circle(frame, tuple(left_pupil_px), 2, (255, 0, 0), -1)   # Blue dot for pupil
            cv2.circle(frame, tuple(right_pupil_px), 2, (255, 0, 0), -1)
            cv2.circle(frame, left_eye_center, 2, (0, 255, 0), -1)       # Green dot for center
            cv2.circle(frame, right_eye_center, 2, (0, 255, 0), -1)

            # Draw Vectors
            vec_left =  left_pupil_px - np.array(left_eye_center)
            vec_right =  right_pupil_px - np.array(right_eye_center)

            cv2.line(frame, left_eye_center, tuple(left_eye_center + 3 * vec_left), (0, 255, 255), 2)
            cv2.line(frame, right_eye_center, tuple(right_eye_center + 3 * vec_right), (0, 255, 255), 2)

    frame_count+=1
    now=time.time()
    if now-prev_time >=1:
        fps=frame_count
        prev_time= now
        frame_count=0      
    
    #FPS font
    info_text = [
        f"FPS: {fps}",
    ]
    for i, text in enumerate(info_text):
        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)      

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
    

cap.release()
cv2.destroyAllWindows()








