import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

# Path to the main folder
main_folder_path = r'C:\Users\Rahul Ganatra\OneDrive\Desktop\frame_extraction\Adjectives'

# Iterate through each emotion folder
for emotion_folder in os.listdir(main_folder_path):
    emotion_path = os.path.join(main_folder_path, emotion_folder)

    # Check if it's a directory
    if os.path.isdir(emotion_path):
        for video_file in os.listdir(emotion_path):
            video_path = os.path.join(emotion_path, video_file)

            # Open the video file
            cap = cv2.VideoCapture(video_path)

            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Hand tracking
                hand_results = hands.process(frame_rgb)

                # Face mesh tracking
                face_results = face_mesh.process(frame_rgb)

                # Draw hand landmarks
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw face mesh landmarks
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Display the processed frame
                cv2.imshow('Processed Video', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()

cv2.destroyAllWindows()
