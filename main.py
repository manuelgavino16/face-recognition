import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

cap = cv2.VideoCapture(0)

while(True):
    # capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=3)
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf >= 60 and conf >= 90:
            print(id_)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        name_text_color = (255, 255, 255)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display frames
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# releases the capture
cap.release()
cv2.destroyAllWindows()

x = input("who are you: ")
# print(x)

Base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(Base_dir, "images")

for root, dirs, files in os.walk(image_dir):
    folder_exist_or_not = all(x.lower() != dir.lower() for dir in dirs)
    if(folder_exist_or_not):
        os.makedirs(os.path.join(image_dir,  x), exist_ok=True)
    else:
        print("It exist")

#name_dir = os.path.join(image_dir, x)
#os.rename(name_dir + "/1.png", name_dir + "/test2.png")
