import os  # To interact with the file system
import numpy as np  # For numerical operations
import cv2  # For image processing (not used in this code but typically included for processing frames)
from tensorflow.keras.utils import to_categorical  # For converting labels to one-hot encoding
from keras.layers import Input, Dense  # For defining layers in the neural network
from keras.models import Model  # For defining and compiling the model

# Initialize variables
is_init = False  # Flag to check if the first .npy file has been loaded
size = -1  # Size of the dataset (number of samples)
label = []  # List to store labels
dictionary = {}  # Dictionary to map label names to numeric values
c = 0  # Counter for labels

# Load all .npy files (pose data) and labels
for i in os.listdir():  # Iterate over all files in the current directory
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):
        if not(is_init):
            is_init = True  # Set flag after loading the first .npy file
            X = np.load(i)  # Load the first .npy file as the feature data (X)
            size = X.shape[0]  # Get the number of samples
            y = np.array([i.split('.')[0]]*size).reshape(-1,1)  # Create labels corresponding to the first file
        else:
            # Concatenate data from other .npy files into the feature and label arrays
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))
        
        # Append the label and map it to a numeric value
        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1  # Increment the counter for the label

# Convert the labels from string to numeric values using the dictionary
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

# Convert labels to integers
y = np.array(y, dtype="int32")

# One-hot encode the labels for categorical classification
y = to_categorical(y)

# Create new variables for the shuffled data
X_new = X.copy()
y_new = y.copy()
counter = 0

# Shuffle the data indices
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

# Reorder the data and labels according to the shuffled indices
for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define the neural network architecture
ip = Input(shape=(X.shape[1]))  # Input layer, shape based on number of features

# Hidden layers with 128 and 64 neurons and 'tanh' activation
m = Dense(128, activation="tanh")(ip)  # First hidden layer
m = Dense(64, activation="tanh")(m)   # Second hidden layer

# Output layer with softmax activation (for multi-class classification)
op = Dense(y.shape[1], activation="softmax")(m)

# Define the model with input and output layers
model = Model(inputs=ip, outputs=op)

# Compile the model with RMSprop optimizer and categorical cross-entropy loss
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model with the training data
model.fit(X_new, y_new, epochs=80)

# Save the trained model to a file
model.save("model.h5")

# Save the labels (mapping) to a file
np.save("labels.npy", np.array(label))
