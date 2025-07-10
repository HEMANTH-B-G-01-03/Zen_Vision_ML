import numpy as np  # NumPy for array manipulation
import mediapipe as mp  # MediaPipe for pose estimation
from keras.models import load_model  # For loading the pre-trained model
import pygame  # For playing sounds during feedback
import cv2  # OpenCV for video capture and frame manipulation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score  # For model evaluation metrics

# Initialize pygame for sound playback (if required)
pygame.init()

# Function to check if body landmarks are visible in the frame
def inFrame(lst):
    """
    This function checks if the body landmarks of certain key points (hips and elbows) are visible in the frame.
    If the visibility scores for these landmarks are above the threshold (0.6), it returns True (body is visible).
    """
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True  
    return False

# Load the pre-trained yoga pose classification model
model = load_model("model.h5")

# Load the yoga pose labels (classes) associated with the model
label = np.load("labels.npy")

# Initialize MediaPipe Pose solution for landmark detection
holistic = mp.solutions.pose
holis = holistic.Pose()

# Initialize drawing utilities to display landmarks on frames
drawing = mp.solutions.drawing_utils

# Open a webcam to capture live video
cap = cv2.VideoCapture(0)

# Set the desired resolution for the webcam feed
desired_width = 1280
desired_height = 920

# Apply the resolution to the video capture settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Arrays to store actual labels (ground truth) and predicted labels
ground_truth = []  # List for storing true labels
predictions = []   # List for storing predicted labels

# Start capturing frames from the webcam
while True:
    lst = []  # List to hold the current frame's landmark data

    # Capture a frame from the video stream
    _, frm = cap.read()

    # Flip the frame horizontally for a mirror effect
    frm = cv2.flip(frm, 1)

    # Process the captured frame to extract pose landmarks using MediaPipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Apply a blur to reduce noise and improve accuracy of pose detection
    frm = cv2.blur(frm, (4, 4))

    # If pose landmarks are detected and the body is visible in the frame
    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        # Normalize the coordinates of the landmarks (relative to the first landmark, e.g., nose)
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        # Convert the list of landmarks to a NumPy array for prediction
        lst = np.array(lst).reshape(1, -1)

        # Predict the yoga pose using the pre-trained model
        p = model.predict(lst)

        # Get the predicted yoga pose (label) based on the model's output
        pred = label[np.argmax(p)]

        # For evaluation purposes, define the ground truth label for the current pose
        # (Replace this with actual ground truth labels in real-world testing)
        actual_label = "Tadasana"  # Example label; replace with actual label during testing

        # Append the actual and predicted labels to the respective lists
        ground_truth.append(actual_label)
        predictions.append(pred)

        # If the model's prediction confidence is above 75%, play success sound and display the pose name
        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(frm, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 4)  # Display pose name on frame

            # Play success sound
            my_sound = pygame.mixer.Sound('sucess.mp3')
            my_sound.play()

        else:
            # If model is not confident, display a warning and play a warning sound
            cv2.putText(frm, "Wrong Yoga Position", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 3, cv2.LINE_AA)

            # Play warning sound
            my_sound = pygame.mixer.Sound('Warning.mp3')
            my_sound.play()

    else:
        # If pose landmarks are not detected, display an error message
        cv2.putText(frm, "Wrong Yoga Position", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 3, cv2.LINE_AA)

    # Draw the pose landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                           connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                           landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    # Show the processed video frame with pose landmarks
    cv2.imshow("Yoga Pose Detection", frm)

    # If 'q' key is pressed, stop capturing and release the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

# After the video capture, calculate the distribution of actual and predicted labels
print("Ground Truth Labels Distribution:", dict(zip(*np.unique(ground_truth, return_counts=True))))
print("Predictions Labels Distribution:", dict(zip(*np.unique(predictions, return_counts=True))))

# Generate confusion matrix to inspect misclassifications
cm = confusion_matrix(ground_truth, predictions)
print("Confusion Matrix:\n", cm)

# Convert ground_truth and predictions lists to NumPy arrays for metric evaluation
ground_truth = np.array(ground_truth)
predictions = np.array(predictions)

# Evaluate performance using precision, recall, F1 score, and accuracy
precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)
accuracy = accuracy_score(ground_truth, predictions)

# Print the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)
