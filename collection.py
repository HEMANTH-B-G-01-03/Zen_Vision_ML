import mediapipe as mp 
import numpy as np 
import cv2  

def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:

		#(28: left hip, 27: right hip, 15: left elbow, 16: right elbow)
		return True  # it then assumes that body is visible   , visibility scores >0.6
	return False

 
cap = cv2.VideoCapture(0)  

name = input("Enter the name of the Asana : ")


holistic = mp.solutions.pose # Used for pose tracking via the mp.solutions.pose module.

holis = holistic.Pose()  # to initialize the pose model to detect the body landmarks 

drawing = mp.solutions.drawing_utils # used to draw the pose marks 

X = []
data_size = 0


while True:
	lst = []

	rst, frm = cap.read()

	frm = cv2.flip(frm, 1)

	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))  # For each frame, the landmarks are processed by the MediaPipe Pose solution (holis.process()). If the body is fully visible (inFrame()), it calculates the normalized coordinates of the landmarks.    IT process the video in RGB format


	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
		for i in res.pose_landmarks.landmark:
			lst.append(i.x - res.pose_landmarks.landmark[0].x) 
			lst.append(i.y - res.pose_landmarks.landmark[0].y)
			
		X.append(lst)
		data_size = data_size+1

	else: 

		cv2.putText(frm, "Make Sure Full body visible", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)  # is a POSE_CONNECTIONS A predefined set of connections that link the landmarks to form a skeletal structure

	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) ,2)  #BGR , THICKNESS 

	cv2.imshow("window", frm)

	if cv2.waitKey(1) == 27 or data_size>80 or 0xFF == ord('q'):
		cv2.destroyAllWindows()
		cap.release()
		break


np.save(f"{name}.npy", np.array(X))  # saving the file with the collected data 
print(np.array(X).shape)
