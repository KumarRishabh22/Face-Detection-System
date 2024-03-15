import cv2

from random import randrange as r
#datset load
trainedDataset=cv2.CascadeClassifier("face.xml")

#choose img
img = cv2.imread("two.jpg")

#conversion to black and white
greyimg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#detect face
faceCoordinates = trainedDataset.detectMultiScale(greyimg) 

for x , y, w , h in faceCoordinates:
    cv2.rectangle(img , (x , y) , (x+w , y+h) ,(r(0 , 256) , r(0 , 256) , r(0 , 256)) , 2 )
    
    
#diplay image
cv2.imshow("Multi Face Detection" , img)
#this will pause the execution of program until any key is pressed
cv2.waitKey() 

print("end program")
