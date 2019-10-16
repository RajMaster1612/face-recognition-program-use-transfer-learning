from PIL import Image
from keras.applications.vgg16 import preprocess_input
import cv2
from keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pylab as plt

from keras.preprocessing import image
model=load_model('facefeatures_new_model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):


    #faces = faceCascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = 0)
    faces = face_cascade.detectMultiScale(img,1.3,5)

    if faces is ():
        return None
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
folders = glob('Datasets/Train/*')
for i in range(len(folders)):
    folders[i]=str(folders[i])[15:]
    
imagenet_labels=folders
try:
	frame=cv2.imread('obama.jpg')
	cv2.namedWindow('img', cv2.WINDOW_NORMAL)
	cv2.imshow('img',frame)
except:
    print ("image source is not avaiable")
face=face_extractor(frame)

if type(face) is np.ndarray:
    face=cv2.resize(face,(224,224))
    
    img_array=np.array(face)

    img_array=np.expand_dims(img_array,axis=0)
    # this predict function give as list of 
    pred=model.predict(img_array)

    print(pred[0])
    #here we find the max value index from model result
    predicted_class = np.argmax(pred[0], axis=-1)
    print(predicted_class)
    #now we pass the index value in imagenet_labels  which give as classifiar name    .... imagenet_labels is a list of labal
    predicted_class_name = imagenet_labels[predicted_class]
    _ = plt.title("Prediction: " + predicted_class_name.title())
    print("Prediction: ", predicted_class_name.title())
	cv2.putText(frame,predicted_class_name.title(),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    # this is comment code is to if some has to set threshold value
cv2.imshow('show',frame)
cv2.imwrite('result1.jpg',frame)
