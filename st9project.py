import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd



from textblob import TextBlob
st.sidebar.title("about us")
st.sidebar.text("""Name:Pankaj Sajwan
Qualification:B.Tech in Mechanical Engineering
Currently Persuing Data Science Course from Ducat
""")

st.sidebar.title("contact us")
st.sidebar.text("""Email:pankaj.sajwan20@gmail.com
Mob No:+91 8477979148
""")



st.title("Face Detection using webcam")
st.write("Detects faces using your webcam")
run=st.button('start webcam')




vdo=cv2.VideoCapture(0)
faceModel=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    flag,frame=vdo.read()
    if flag==False:
        break
    cv2.putText(frame,"Press 'c' to cancel!",(10,30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceModel.detectMultiScale(gray,minNeighbors=8)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        face=frame[y:y+h,x:x+w]
        
        img_gray=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
        img_gray=cv2.resize(img_gray,(100,100))
        img_gray=img_gray.flatten()
        img_gray=img_gray/255
        
        
    cv2.imshow("vdo",frame)
    key=cv2.waitKey(25)
    if key==ord('c'):
        break
    

cv2.destroyAllWindows()
vdo.release()




