from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from win32com.client import Dispatch
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./model1.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

def music(l):
    global tune   

    mp = Dispatch("WMPlayer.OCX")
    tune = mp.newMedia("./songs/Angry/a.mp3")
    if l == "Angry":
        tune = mp.newMedia("./songs/Angry/a.mp3")
        print("Angry song is playing.")
    elif l == "Happy":
        tune = mp.newMedia("./songs/happy/pala.mp3")
        print("Happy song is playing.")
    elif l == "Sad":
        tune = mp.newMedia("./songs/sad/yaro.mp3")
        print("Sad song is playing.")
    elif l == "Neutral":
        tune = mp.newMedia("./songs/nuetral/ram.mp3")
        print("Nuetral song is playing.")
    elif l == "Surprise":
        tune = mp.newMedia("./songs/surprice/neya.mp3")
        print("Surprise song is playing.")
    mp.currentPlaylist.appendItem(tune)
    mp.controls.play()
    sleep(1)
    mp.controls.playItem(tune)
    # to stop playing use
    input("Press Enter to stop playing")
    mp.controls.stop()



cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            music(label)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """
    if label=="Angry" or label=="Surprise" or label=="Sad" or label=="Happy" or label=="Neutral":
        music(label)
    else:
        continue
    """
cap.release()
cv2.destroyAllWindows()


























