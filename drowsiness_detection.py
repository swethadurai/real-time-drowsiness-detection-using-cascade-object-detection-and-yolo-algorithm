## import packages

import cv2
import os
from keras.models import load_model
import tensorflow as tf
import numpy as np
from pygame import mixer
import time
from mail import report_send_mail
import serial


path = os.getcwd()



'''ser = serial.Serial(
    port='COM3',  # Device name
    baudrate=9600,  # Baud rate such as 9600 or 115200 etc.
    parity=serial.PARITY_NONE,  # Enable parity checking
    stopbits=serial.STOPBITS_ONE,  # Number of stop bits
    bytesize=serial.EIGHTBITS,  # Number of data bits.
    timeout=.1,  # Set a read timeout value.
    rtscts=0  # Enable hardware (RTS/CTS) flow control.
)'''






#mixer.init()
#sound = mixer.Sound('alarm.wav')


## cascating files
face = cv2.CascadeClassifier('cascade\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('cascade\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('cascade\haarcascade_righteye_2splits.xml')


## number of class

lbl=['Close','Open']


## load model
#model = load_model('models/drowsiness_model.h5')
model = tf.keras.models.load_model('models/drowsiness_model.h5')

## to get web cam capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

## initialize the some parameter

count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred_ = model.predict(r_eye)
        lpred = np.argmax(rpred_,axis=1)
        #print(lpred)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred_ = model.predict(l_eye)
        lpred = np.argmax(lpred_,axis=1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 or lpred[0]==0):
        print('sleep')
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        
    if(rpred[0]==1 or lpred[0]==1):
        #score=score-3
        score=0
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        print('No sleep')
        #ser.write(b'0')
        #time.sleep(2)
        print('-------------')
        #print('0 Send')
        print('-------------')
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>5):
        
        #person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        cv2.imwrite('image.jpg', frame)
        report_send_mail('image.jpg')
        
        #ser.write(b'1')
        #time.sleep(2)
        print('-------------')
        #print('1 Send')
        print('-------------')
        try:
            #sound.play()
            print('')
        except:  
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
        
    # to plot on screen
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


## release image  
cap.release()
cv2.destroyAllWindows()
