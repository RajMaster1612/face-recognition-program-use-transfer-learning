import cv2
import glob

cascPath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

path = glob.glob("Datasets/Train/_barack_obama_/*.jpg")


for imagepath in path:
    img=cv2.imread(str(imagepath))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)

    print("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        cv2.imshow('Image', crop)
        updated_name = imagepath
        cv2.imwrite(updated_name,crop)