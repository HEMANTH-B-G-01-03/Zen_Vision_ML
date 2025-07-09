import mediapipe as mp  # MediaPipe library for pose detection
import numpy as np  # NumPy for array manipulation
import cv2  # OpenCV for video capture and image processing

# Function to check if the body is in the frame (based on visibility scores)
def inFrame(lst):
    # Checking if certain body parts have visibility scores > 0.6 (good visibility)
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True  # Assumes the body is visible if these key points are detected well
    return False

# Capturing the video from the webcam
cap = cv2.VideoCapture(0)

# Input for the name of the Asana (pose)
name = input("Enter the name of the Asana : ")

# Initialize MediaPipe Pose solution
holistic = mp.solutions.pose
holis = holistic.Pose()

# Drawing utilities to show the landmarks on the image
drawing = mp.solutions.drawing_utils

# Initialize empty list to collect pose data
X = []
data_size = 0

while True:
    lst = []  # List to hold the current frame's landmark data

    # Capture a frame from the webcam
    _, frm = cap.read()

    # Flip the frame horizontally to mirror the view
    frm = cv2.flip(frm, 1)

    # Process the frame with MediaPipe Pose to get landmarks (in RGB format)
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if the landmarks are detected and the body is visible
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            # Collecting the relative positions of landmarks (from the nose)
            lst.append(i.x - res.pose_landmarks.landmark[0].x) 
            lst.append(i.y - res.pose_landmarks.landmark[0].y)
        
        # Append the frame's data to the list X
        X.append(lst)
        data_size += 1  # Increment the data size

    else: 
        # Display a message if the full body is not visible
        cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

    # Display the size of collected data
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed with landmarks and data size
    cv2.imshow("window", frm)

    # Stop the loop when 'ESC' key is pressed or data size exceeds 80
    if cv2.waitKey(1) == 27 or data_size > 80 or 0xFF == ord('q'):
        cv2.destroyAllWindows()  # Close the windows
        cap.release()  # Release the webcam
        break

# Save the collected data as a numpy array to a file
np.save(f"{name}.npy", np.array(X))

# Print the shape of the collected data
print(np.array(X).shape)
