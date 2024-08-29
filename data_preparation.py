import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load sequences and labels
sequences = np.load('C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/sequences.npy')
labels = np.load('C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/labels.npy')

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoded format
labels_one_hot = to_categorical(labels_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels_one_hot, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
