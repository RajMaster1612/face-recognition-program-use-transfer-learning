from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pylab as plt

from keras.preprocessing import image
model=load_model('facefeatures_new_model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

folders = glob('Datasets/Train/*')
for i in range(len(folders)):
    folders[i]=str(folders[i])[15:]
    

print(folders)
imagenet_labels=folders

def face_extractor(img):


    #faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = 0)

    faces = face_cascade.detectMultiScale(img,1.3,5)

    if faces is ():
        return None
    # Draw a rectangle around the faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(72,255,21),2)
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
        print(pred[0])
        #here we find the max value index from model result
        predicted_class = np.argmax(pred[0], axis=-1)
        print(predicted_class)
        #now we pass the index value in imagenet_labels  which give as classifiar name    .... imagenet_labels is a list of labal
        predicted_class_name = imagenet_labels[predicted_class]
        _ = plt.title("Prediction: " + predicted_class_name.title())
        #print(str(plt.title))
        #print(pred[0])
        print("Prediction: ", predicted_class_name.title())
        # this is comment code is to if some has to set threshold value
        
        cv2.putText(frame,predicted_class_name.title(),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
