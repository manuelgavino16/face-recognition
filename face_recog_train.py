import os
import numpy as np
from PIL import Image
import cv2
import pickle

Base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(Base_dir, "images")

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            print(label, path)

            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)

            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.3, minNeighbors=3)

            for(x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

# print(y_labels)
# print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
