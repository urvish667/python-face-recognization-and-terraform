# Python Face recognization and Terraform
![1_e6kMl3pMLkECCqflDd9-7Q](https://user-images.githubusercontent.com/43312731/188873026-955f1d5d-3761-4bbc-9f14-3a2d7fd8148e.png)
## Hi! There
In Jupyter notebook, we will see how to create face recognition program and after detecting the correct face how to launch virtual machines with the help of terraform scripts. This task need pre-installed OpenCV and Terraform on the system

```
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_cropper(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3,5)
    if len(faces)==0:
        return None
    elif len(faces)>0 :
        for (x1,y1,x2,y2) in faces:
            cropped_face = img[y1:y1+y2 , x1:x1+x2]
            break
    return cropped_face
```
face_cropper function will detects the face of user and crops it.

```
import os
cap = cv2.VideoCapture(0)
count = 0
abs_path = './collected_pictures/' # {.} current directr 
while True:
    ret, photo = cap.read()
    cropped_face = face_cropper(photo)
    if cropped_face is not None:
        cropped_face = cv2.resize(cropped_face, (200,200))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        count+=1
        file_name = str(count)+'.jpg'
        saved = cv2.imwrite(os.path.join(abs_path, file_name), cropped_face)
        if not saved:
            print("Couldn't Save your Photos!")
            
            print("Make sure the folder with name 'collected_pictures' is created under current working directory")
            break
        cv2.putText(cropped_face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Cropped Face', cropped_face)
    else:
        pass
    if count==100:
        print('Samples Pictures Collected Successfully')
        break
    if cv2.waitKey(10)==13:
        break
cap.release()
cv2.destroyAllWindows()
```
This code will makes 100 copies of cropped image and will stores into "collected_pictures".

```
from os import listdir
from os.path import isfile, join
from PIL import Image
import cv2
import numpy as np

abs_path = './collected_pictures/'
face_files = [f for f in listdir(abs_path) if isfile(join(abs_path, f))]
train_data, labels=[], []
for i,file_name in enumerate(face_files):
    image_path = abs_path+face_files[i]
    faceImg = Image.open(image_path)
    train_data.append(np.array(faceImg, dtype=np.uint8))
    labels.append(i)

labels = np.asarray(labels, dtype=np.int32)

model = cv2.face_LBPHFaceRecognizer.create()
model.train(train_data, labels)
print("Model Trained Successfully!")
```
This code will collects the images from collected_pictures and labelled them for model training.

```
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detect_crop(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3,5)
    if len(faces)==0:
        return img, []
    elif len(faces)>0 :
        for (x1,y1,x2,y2) in faces:
            img = cv2.rectangle(img, (x1,y1),(x1+x2,y1+y2), [255,255,255], 1)
            cropped_face = img[y1:y1+y2 , x1:x1+x2]
            cropped_face = cv2.resize(cropped_face, (200,200))
    return img, cropped_face
```
This function will returns the cropped faced for face recognition.

```
def init():
    print("Output of 'terraform init' command :")
    print(subprocess.getoutput("terraform init"))
    print()
    [print("-",end='') for i in range(80)]
    print()

def apply():
    print("Output of 'terraform apply' command :")
    print(subprocess.getoutput("terraform apply -auto-approve"))
```

Will run the terraform script.

