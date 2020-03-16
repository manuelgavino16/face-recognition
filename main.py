import numpy as np
import cv2
import os
import pickle

# testing  git


def move_images():
    face_dir = os.path.join(image_dir, x)
    i = 0
    for root, dirs, files in os.walk(face_dir):
        for file in files:
            face_file_exist = all(
                str(i) + ".png" != file for file in files)
            lastest_file = i
            # print(i)
            i += 1
    j = 0
    for j in range(10):
        os.rename(str(j) + ".png", "images/" + x +
                  "/" + str(lastest_file) + ".png")
        #print(j, lastest_file)
        j += 1
        lastest_file += 1


face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    labels = pickle.load(f)
    labels = {v: k for k, v in labels.items()}

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
        if conf >= 30:
            print("Conf:", round(conf, 3))
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        for i in range(10):
            img_name = str(i) + ".png"
            cv2.imwrite(img_name, roi_color)
            i += 1

        name_text_color = (255, 255, 255)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
    '''
    for i in range(10):
        img_name = str(i) + ".png"
        img_item = cv2.imwrite(img_name, roi_color)
        images_for_image_folder[i] = img_item
        i += 1
    '''

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
move_images()
