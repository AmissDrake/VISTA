import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while cap.isOpened:
    ret,frame=cap.read()
    if(ret==True):
        hsl=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
        lower_bound=np.array([0,50,50])
        upper_bound=np.array([20,255,255])
        mask=cv2.inRange(hsl, lower_bound, upper_bound)
        masked_image=cv2.bitwise_and(frame,frame,mask=mask)
        gray_img=cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
        contours,hierarchy=cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img=cv2.drawContours(frame.copy(),contours,-1,(0,0,255),2)
        cv2.imshow('contour',contour_img)
        if(cv2.waitKey(1) & 0xFF == ord('w')):
            break
    else:
        break  

cap.release()
cv2.destroyAllWindows()