import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(cap.isOpened()):
    ret,frm=cap.read()

    hsl=cv2.cvtColor(frm,cv2.COLOR_BGR2HLS)

    lb=np.array([0,50,50])
    ub=np.array([179,255,255])
    mask=cv2.inRange(hsl,lb,ub)
    m_frm=cv2.bitwise_and(hsl,hsl,mask=mask)

    g_frm=cv2.cvtColor(m_frm,cv2.COLOR_BGR2GRAY)

    _, t_frm=cv2.threshold(m_frm,150,255,cv2.THRESH_BINARY)

    ctrs,_=cv2.findContours(g_frm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c_frm=cv2.cvtColor(g_frm,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(c_frm,ctrs,-1,(0,255,0),2)
    cv2.imshow("task5",c_frm)
    
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
