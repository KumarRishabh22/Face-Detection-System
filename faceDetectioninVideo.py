import cv2

from random import randrange as r
#datset load
trainedDataset=cv2.CascadeClassifier("face.xml")


webcam = cv2.VideoCapture("clip.mp4")

while True :
 success , frame = webcam.read()
 greyimg = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

 faceCoordinates = trainedDataset.detectMultiScale(greyimg) 

 for x , y, w , h in faceCoordinates:
    cv2.rectangle(frame , (x , y) , (x+w , y+h) ,(r(0 , 256) , r(0 , 256) , r(0 , 256)) , 2 )
    
 cv2.imshow("single person" , frame)
 key = cv2.waitKey(1) 
 if(key == 81 or key == 113):
   break
 
print("end program")