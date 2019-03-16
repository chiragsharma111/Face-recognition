import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)
while(True):
	#Capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for(x,y,w,h) in faces:
		# print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
		roi_color = frame[y:y+h, x:x+h]

		#Recognize? Deep laerned model predict keras tensorflow pytorch scikit learn
		id_, conf = recognizer.predict(roi_gray)
		if(conf>45 and conf<85):
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


		img_item = "7.png"
		cv2.imwrite(img_item, roi_gray)

		color = (0, 255, 0)
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey), (ex+ew, ey+eh),(0,0,255),2)

	#Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Aim
# We can recognize face with just one image as input of any person_name ( self train when identify the person )
# +
# emotion detection ( recommendation based on emotion like recommend jokes when you are sad or angry )



# Part1: 
# 	Phase1: Face detection
# 	Phase2: Face recognition with just one photo

# Part2:
# 	Phase1: Emotion detection
# 	Phase2: Emotion recognition
# 	Phase3: recommendation based on emotion	