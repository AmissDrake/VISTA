import cv2
import time
import numpy as np
import psutil
cap = cv2.VideoCapture(0)  
frame_count = 0
start_time = time.time()
fps = 0
ram_usage = 0
accuracy_log = []

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
    
        eye_region = gray[ey:ey+eh, ex:ex+ew]
        #(dark area)
        _, threshold = cv2.threshold(eye_region, 40, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (pupil)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            pupil_contour = contours[0]
            
            # Get pupil center
            moments = cv2.moments(pupil_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"]) + ex
                cy = int(moments["m01"] / moments["m00"]) + ey
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                
                # Simple accuracy metric (1 if detected, 0 if not)
                accuracy_log.append(1)
        else:
            accuracy_log.append(0)
    
    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    # Get RAM usage in MB
    process = psutil.Process()
    ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    # Display metrics on frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"RAM: {ram_usage:.1f} MB", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if accuracy_log:
        avg_accuracy = sum(accuracy_log)/len(accuracy_log)
        cv2.putText(frame, f"Accuracy: {avg_accuracy:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Pupil Tracking', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print final metrics
print("\nFinal Performance Metrics:")
print(f"Average FPS: {fps:.1f}")
print(f"Max RAM Usage: {ram_usage:.1f} MB")
if accuracy_log:
    print(f"Average Accuracy: {sum(accuracy_log)/len(accuracy_log):.2f}")