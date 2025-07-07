import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense 

from keras.models import Model
 
is_init = False #Tracks whether the initial dataset (X) has been loaded.
size = -1	#Placeholder for the size of the data in each .npy fil
label = [] #List to store the names of the labels (asana names).
dictionary = {} # Maps label names to integer indices 

c = 0


for i in os.listdir():
	if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):    #splits the file name i into parts using the period . as a delimiter. like example  if treepose.npy it is sploted as  treepose and npy , and -1 mean refers to the last part of the spilit that is  npy 
		  

		  # combines data from multiple files, and prepares corresponding labels for each dataset.
		if not(is_init):
			is_init = True  #checks if the initialization is done or not 
			X = np.load(i)
			size = X.shape[0]
			y = np.array([i.split('.')[0]]*size).reshape(-1,1)

		else:
			X = np.concatenate((X, np.load(i)))
			y = np.concatenate((y, np.array([i.split('.')[0]]*size).reshape(-1,1)))

		label.append(i.split('.')[0])
		dictionary[i.split('.')[0]] = c  
		c = c+1


#this is used to conbvert the labels to int 
for i in range(y.shape[0]):
	y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

#like before maping it will be line cowpose,treepose , warriorpose after converting it will be 0,1,2 

y = to_categorical(y)  # this is one hot encoding  
#purpose converts numeric labels into one-hot encoded vectors for multi-class classification.


X_new = X.copy()
y_new = y.copy()
counter = 0 

#used to shuffel the data set 
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt: 
	X_new[counter] = X[i]
	y_new[counter] = y[i]
	counter = counter + 1


ip = Input(shape=(X.shape[1]))

m = Dense(128, activation="tanh")(ip)  # nural network  128 nurons
m = Dense(64, activation="tanh")(m)   # nural network  64 nurons

op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

model.fit(X_new, y_new, epochs=80)


model.save("model.h5")
np.save("labels.npy", np.array(label))

