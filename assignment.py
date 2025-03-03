import cv2
import numpy as np
img=cv2.imread('hand2.jpg')
#task1
resize=cv2.resize(img,(450,800))
(rows,cols)=resize.shape[:2]
print(rows)
print(cols)
cv2.imshow('frame',resize)
cv2.waitKey(2000)
crop=resize[20:600, 20:600]
cv2.imshow('frame',crop)
(w,h)=crop.shape[:2]
cv2.waitKey(2000)
R=cv2.getRotationMatrix2D((w/2,h/2),180,1)
rotate=cv2.warpAffine(crop,R,(w,h))
cv2.imshow('frame',rotate)
cv2.waitKey(2000)
#task2
hsl=cv2.cvtColor(rotate,cv2.COLOR_BGR2HLS)
cv2.imshow('hsl',hsl)
cv2.waitKey(2000)
#task3
lower_bound=np.array([0,50,50])
upper_bound=np.array([20,255,255])
mask = cv2.inRange(hsl, lower_bound, upper_bound)
masked_image = cv2.bitwise_and(rotate, rotate, mask=mask)
cv2.imshow('mask', mask)
cv2.waitKey(2000)
cv2.imshow('masked image', masked_image)
cv2.waitKey(2000)
cv2.destroyAllWindows()
#task4
gray_img=cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray_img)
cv2.waitKey(2000)
#task5
_, thresholded_image = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
cv2.imshow('threshold',thresholded_image)
cv2.waitKey(2000)
contours,hierarchy=cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img=cv2.drawContours(rotate.copy(),contours,-1,(0,0,255),2)
cv2.imshow('contour',contour_img)
cv2.waitKey(2000)


