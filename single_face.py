import cv2

#datset load
trainedDataset=cv2.CascadeClassifier("face.xml")

#choose img
img = cv2.imread("one.webp")

#conversion to black and white
greyimg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#detect face
faceCoordinates = trainedDataset.detectMultiScale(greyimg) 

x , y, w , h = faceCoordinates[0]

#drawing rectangle
cv2.rectangle(img , (x , y) , (x+w , y+h) ,(0 , 0 , 255) , 2 )


#diplay image
cv2.imshow("single person" , img)
#this will pause the execution of program until any key is pressed
cv2.waitKey() 

print("end program")
