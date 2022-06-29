#import libraries

import torch
import cv2
import numpy as np

#Leemos el modelo

model = torch.hub.load('ultralytics/yolov5','custom',
                       path= '/Users/natalio/Desktop/model/best.pt')

#Realizo VideoCaptura

cap = cv2.VideoCapture('/Users/natalio/Desktop/model/Car Detection/highway.mp4')

#Empezamos 

while True:
    #Realizamos lectura de frames
    ret, frame = cap.read()

    #Correacción de Coor
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Realizamos detección 
    detect = model(frame)

    #Mostramos FPS
    cv2.imshow('Detector de Carros', np.squeeze(detect.render()))

    #Leer el teclado
    t= cv2.waitKey(5)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()    