import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import pygame       
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize pygame for sound playback
pygame.init()

def inFrame(lst):
    """
    Function to check if certain key body landmarks are visible in the frame.
    It checks the visibility scores for specific landmarks (hips, elbows).
    If their visibility scores are above 0.6, it assumes the body is visible.
    """
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True 
    return False

# Load the pre-trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe Pose for detecting body landmarks
holistic = mp.solutions.pose #Creates an object to process video frames and detect pose landmarks.
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils #Used to draw pose landmarks and connections on the video frames.

# Open the webcam to capture video
cap = cv2.VideoCapture(0)

# Set the resolution for the webcam feed
desired_width = 1280
desired_height = 920
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Lists to store predictions
predictions = []   # Predicted labels (since we don't have ground truth labels for real-time detection)

while True:
    lst = []  # List to store the pose landmark data

    # Capture a frame from the webcam
    _, frm = cap.read()

    # Create an empty black window to show the results
    window = np.zeros((960, 960, 3), dtype="uint8")

    # Flip the frame horizontally for a mirror effect
    frm = cv2.flip(frm, 1) 

    # Process the frame and extract the pose landmarks using MediaPipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Apply a blur to reduce noise in the frame
    frm = cv2.blur(frm, (4, 4))

    if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
        # Normalize the landmarks and store them in lst
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        # Reshape the list to match the model input format
        lst = np.array(lst).reshape(1, -1)

        # Predict the pose using the trained model
        p = model.predict(lst)

        # Get the predicted label from the model's output
        pred = label[np.argmax(p)]

        # Append the predicted label to the list of predictions
        predictions.append(pred)

        # If the model is confident (probability > 0.75), play success sound and display the pose name
        if p[0][np.argmax(p)] > 0.75:
            cv2.putText(window, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 4)

            # Play success sound
            my_sound = pygame.mixer.Sound('sucess.mp3')
            my_sound.play()
        else:
            # If the model is not confident, display a warning and play a warning sound
            cv2.putText(window, "Wrong Yoga Position", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 3, cv2.LINE_AA)

            # Play warning sound
            my_sound = pygame.mixer.Sound('Warning.mp3')
            my_sound.play()

    else:
        # If landmarks are not detected, show a wrong pose message
        cv2.putText(window, "Wrong Yoga Position", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 3, cv2.LINE_AA)

    # Draw the pose landmarks on the frame
    drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                            connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                            landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

    # Resize the frame to fit into the window
    resized_frame = cv2.resize(frm, (640, 480))

    # Calculate the position to place the frame at the center of the window
    start_x = (window.shape[1] - resized_frame.shape[1]) // 2
    start_y = (window.shape[0] - resized_frame.shape[0]) // 2
    end_x = start_x + resized_frame.shape[1]
    end_y = start_y + resized_frame.shape[0]
    
    # Position the resized frame at the calculated location in the window
    window[start_y:end_y, start_x:end_x, :] = resized_frame

    # Display the result window
    cv2.imshow("Yoga Pose Detection", window)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

# At the end of the session, evaluate the model's predictions using precision, recall, accuracy, and F1-score
# Since we don't have ground truth labels, we can't calculate these metrics directly without labeled data.
# Assuming you have a set of true labels (ground_truth) from a test set, you would use:

# Calculate accuracy, precision, recall, and F1-score
# Assuming you have true labels in the 'ground_truth' list, you can calculate the metrics:
ground_truth = predictions  # In a real scenario, you should have separate ground truth labels

accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='weighted', zero_division=1)
recall = recall_score(ground_truth, predictions, average='weighted', zero_division=1)
f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=1)

# Print the metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
