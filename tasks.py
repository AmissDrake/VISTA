import cv2
import numpy as np

#Task 1

img=cv2.imread("trial.jpg",-1)

#a
re=cv2.resize(img,(0,0),fx=1.5,fy=1.5)
#b
cropmg=re[100:300,100:300]
#c
h,w,_=img.shape
rotation=cv2.getRotationMatrix2D((w/2,h/2),180,2)
trans=cv2.warpAffine(img,rotation,(w,h))

cv2.imshow("task1",trans)


#task 2

hsl=cv2.cvtColor(trans,cv2.COLOR_BGR2HLS)
cv2.imshow("task2",hsl)


#task 3

lb=np.array([0,50,50])
ub=np.array([179,255,255])
mask=cv2.inRange(hsl,lb,ub)
m_img=cv2.bitwise_and(hsl,hsl,mask=mask)
cv2.imshow("task3",m_img)


#task 4

g_img=cv2.cvtColor(m_img,cv2.COLOR_BGR2GRAY)
cv2.imshow("task4",g_img)


#task 5

_, t_img=cv2.threshold(g_img,127,255,cv2.THRESH_BINARY)

ctrs,_=cv2.findContours(t_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
c_img=cv2.cvtColor(t_img,cv2.COLOR_GRAY2BGR)
cv2.drawContours(c_img,ctrs,-1,(0,255,0),2)
cv2.imshow("task5",c_img)




cv2.waitKey(0)
cv2.destroyAllWindows()


