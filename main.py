import cv2
import os
import glob
from deepface import DeepFace
import tensorflow as tf
import glob
import numpy as np

model = tf.keras.models.load_model("./Trained_comb2.h5")

target_height = 256
target_width = 256

def getEMB(img1 , img2 , img3 , img4):
    outer = []
    for i in range(0,128):
        outer.append(0)
    try:
        emb1 = DeepFace.represent(img_path=img1 , model_name = "OpenFace")
    except:
        emb1 = outer
    try:
        emb2 = DeepFace.represent(img_path=img2 , model_name = "OpenFace")
    except:
        emb2 = outer
        
    try:
        emb3 = DeepFace.represent(img_path=img3 , model_name = "OpenFace")
    except:
        emb3 = outer
    try:
        emb4 = DeepFace.represent(img_path=img4 , model_name = "OpenFace")
    except:
        emb4 = outer

                
    return emb1 , emb2 , emb3 , emb4


target_height = 256
target_width = 256

def get_images_from_dir(subdirectory):

  image_paths = glob.glob(subdirectory + "*") 
  print(len(image_paths))
  emb1,emb2,emb3,emb4 = getEMB(image_paths[0],
               image_paths[1],
               image_paths[2],
               image_paths[3],)

  images = []
  for image_path in image_paths:
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # Adjust channels as needed
    image = tf.image.resize(image, (target_height, target_width))  # Define target size
    images.append(image)

  return images , emb1 , emb2 ,emb3 ,emb4


def run_model():
    images ,emb1 , emb2 ,emb3 ,emb4= get_images_from_dir("./data/")
    out = model([
        np.expand_dims(images[0] , axis=0), 
        np.expand_dims(images[1] , axis=0), 
        np.expand_dims(images[2] , axis=0), 
        np.expand_dims(images[3] , axis=0), 
        np.expand_dims(emb1 , axis=0), 
        np.expand_dims(emb2 , axis=0), 
        np.expand_dims(emb3 , axis=0), 
        np.expand_dims(emb4 , axis=0), 
        ])
    return out




cam = cv2.VideoCapture("http://192.168.42.129:8080/video")
haar_cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_alt2.xml')


frameNum = 0
imgsNum = 0
aliveness = []

while True:
    ret , frame = cam.read()
    frame = cv2.resize(frame , (256,256))
    gray  =cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    face_rect = haar_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 7
    )

    for (x,y,w,h) in face_rect:
        frameNum += 1
        if frameNum % 5 == 0:
            cv2.imwrite(f"./data/{imgsNum}.png" , frame)
            imgsNum+= 1
        cv2.rectangle(gray , (x,y) , (x+w , y+h) , (0,255,0) , thickness=2)

    if imgsNum >= 4:
        out = float(run_model())
        if out > 0.6:
            aliveness.append(1)
        else:
            aliveness.append(0)
        print(out)
        imgsNum = 0
        os.remove("./data/0.png")
        os.remove("./data/1.png")
        os.remove("./data/2.png")
        os.remove("./data/3.png")
        

    if len(aliveness) >4:
        mean = np.mean(aliveness)
        image = np.zeros((500,900,3), np.uint8) + 255
        if mean > 0.5 :
            ourString =  'you are alive with ' + str(mean * 100) + " accuracy"
            cv2.putText(image, ourString, (100,250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (40,200,0), 2)
        else:
            ourString= 'this is Dead with ' + str(np.round(np.abs((mean * 100) -100))) + " accuracy"
            cv2.putText(image, ourString, (100,250), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (40,0,200), 2)
        
        cv2.imshow("Messing with some text", image)
        
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        aliveness = []
        


    cv2.imshow('frame' , gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
