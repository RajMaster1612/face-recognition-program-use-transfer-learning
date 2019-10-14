from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model=load_model('facefeatures_new_model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):


    #faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = 0)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
    #flags = 0)
    faces = face_cascade.detectMultiScale(img,1.3,5)

    if faces is ():
        return None

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face




try:
#get video source with cv2.VideoCapture(video path './abc/xyz.mp4'or 0 for defult cam)
    video_capture=cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(0)
except:
    print ("Video source is not avaiable")
while True:
    
    _,frame=video_capture.read()
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face=cv2.resize(face,(224,224))
        im=Image.fromarray(face,'RGB')
        im = face

        img_array=np.array(im)

        img_array=np.expand_dims(img_array,axis=0)
        pred=model.predict(img_array)
        print(pred)
        print("pred[0][0]=",pred[0][0])
        print("pred[0][1]=",pred[0][1])
        #print("pred[0][2]=",pred[0][2])
       # print("pred[0][3]=",pred[0][3])
        #print("pred[0][4]=",pred[0][4])
        print("pred[0]",pred[0])
        #print("pred[1]",pred[1])


        name="None matching"
        result = np.where(pred == np.amax(pred))
        print(result)
        #maxInColumns = numpy.amax(arr2D, axis=0) 
        #print(maxpos)
        if(float(pred[0][1])<0.5):
            name='obama'
        #elif(pred[0][1]>0.8):
         #   name='gajru'
        #elif(pred[0][2]>0.8):
         #   name='surya'
        #elif(pred[0][3]>=1):
            #name='raj'
        #elif(pred[0][4]>=1):
            #name='surya'

        cv2.putText(frame,name,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    else:
        cv2.putText(frame,"No face found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
