import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Path to the main folder
main_folder_path = r'C:\Users\Rahul Ganatra\OneDrive\Desktop\frame_extraction\Adjectives'

# Initialize an empty list to store sequences and labels
sequences = []
labels = []

# Define a fixed sequence length
sequence_length = 30

# Iterate through each emotion folder
for emotion_folder in os.listdir(main_folder_path):
    emotion_path = os.path.join(main_folder_path, emotion_folder)
    print(f"Processing folder: {emotion_folder}")

    # Check if it's a directory
    if os.path.isdir(emotion_path):
        for video_file in os.listdir(emotion_path):
            video_path = os.path.join(emotion_path, video_file)
            print(f"Processing video: {video_file}")

            # Open the video file
            cap = cv2.VideoCapture(video_path)
            sequence = []

            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Hand tracking
                hand_results = hands.process(frame_rgb)

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        sequence.append(landmarks.flatten())

                # If enough frames, store the sequence and reset
                if len(sequence) == sequence_length:
                    sequences.append(sequence)
                    labels.append(emotion_folder)  # Label with the emotion folder name
                    sequence = []

            cap.release()

print(f"Extracted {len(sequences)} sequences and {len(labels)} labels.")

# Convert sequences and labels to arrays
sequences = np.array(sequences)
labels = np.array(labels)

# Define the path to the project folder where you want to save the files
project_folder_path = r'C:\Users\Rahul Ganatra\OneDrive\Desktop\frame_extraction\ProjectData'

# Create the project folder if it doesn't exist
os.makedirs(project_folder_path, exist_ok=True)

# Save sequences and labels to the project folder
np.save(os.path.join(project_folder_path, 'sequences.npy'), sequences)
np.save(os.path.join(project_folder_path, 'labels.npy'), labels)

print(f"Data saved successfully in {project_folder_path}")
