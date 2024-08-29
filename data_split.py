import numpy as np
import os
from sklearn.model_selection import train_test_split

# Load sequences and labels
sequences = np.load('C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/sequences.npy')
labels = np.load('C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define paths for training and testing data folders
train_folder_path = 'C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/train_data'
test_folder_path = 'C:/Users/Rahul Ganatra/OneDrive/Desktop/frame_extraction/ProjectData/test_data'

# Create directories if they do not exist
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)

# Save training data
np.save(os.path.join(train_folder_path, 'X_train.npy'), X_train)
np.save(os.path.join(train_folder_path, 'y_train.npy'), y_train)

# Save testing data
np.save(os.path.join(test_folder_path, 'X_test.npy'), X_test)
np.save(os.path.join(test_folder_path, 'y_test.npy'), y_test)

print(f"Training data saved in {train_folder_path}")
print(f"Testing data saved in {test_folder_path}")
