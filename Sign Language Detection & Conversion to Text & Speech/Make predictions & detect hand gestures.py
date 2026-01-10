# Import necessary dependencies
import pickle  # For loading the pre-trained model
import cv2  # OpenCV for video capture and image processing
import mediapipe as mp  # MediaPipe for hand detection and landmarks
import numpy as np  # For numerical operations

# Load the pre-trained model from a file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']  # Extract the model from the loaded dictionary

# Initialize the webcam for video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands for hand detection and landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary mapping numeric labels to characters
labels_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'del', 15: 'E', 16: 'F', 17: 'G',
    18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'nothing',
    26: 'O', 27: 'P', 28: 'Q', 29: 'R', 30: 'S', 31: 'space', 32: 'T', 33: 'U',
    34: 'V', 35: 'W', 36: 'X', 37: 'Y', 38: 'Z'
}

# Start a loop to continuously capture frames from the webcam
while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret:  # If the frame is not captured correctly, exit the loop
        break

    # Initialize lists to store auxiliary data and coordinates
    data_aux = []
    x_, y_ = [], []

    # Convert the frame to RGB format for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)  # Process the frame for hand landmarks

    if results.multi_hand_landmarks:  # If hand landmarks are detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and normalize landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Prepare the feature vector by subtracting minimum coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Make sure the length of features matches what the model expects
            if len(data_aux) == 42:  # Adjust this according to your model's expected input length
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]  # Directly use the predicted string

                # Draw bounding box and predicted character on the frame
                x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
                x2, y2 = int(max(x_) * frame.shape[1]), int(max(y_) * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 32, 48), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (25, 32, 48), 3, cv2.LINE_AA)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style()
            )

    # Display the frame with the drawn landmarks and predictions
    cv2.imshow('Sign Language Detector', frame)
    key = cv2.waitKey(1)  # Wait for 1 ms for a key press

    if key == ord('x'):  # Exit the loop if 'x' is pressed
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
