# Importing the necessary dependencies
import speech_recognition as sr  # For voice command recognition
import time  # To handle timing for gesture detection
from gtts import gTTS  # Google Text-to-Speech for converting text to speech
from playsound import playsound  # To play the generated speech
import os  # For file operations like saving and deleting audio files
import pickle  # To load the trained model
import mediapipe as mp  # For hand gesture detection
import cv2  # OpenCV for capturing video and handling frames
import numpy as np  # To handle numerical operations like array creation

# Function to convert text to speech
def text_to_speech(text):
    if not text.strip():  # Check if the text is empty or only whitespace
        print("No text to convert to speech.")
        return
    # Convert the text to speech using gTTS with Nigerian accent (tld='com.ng')
    speech = gTTS(text, tld='com.ng', lang='en', slow=False)
    speech_file = 'speech.mp3'  # Temporary filename for the speech audio
    speech.save(speech_file)  # Save the speech to the file
    playsound(speech_file)  # Play the speech audio
    os.remove(speech_file)  # Remove the audio file after playing

# Function to get the predicted gesture from the frame
def get_predicted_gesture(frame, hands, model, mp_drawing, mp_hands, mp_drawing_styles):
    data_aux = []  # To store the normalized coordinates of landmarks
    x_, y_ = [], []  # To store x and y coordinates separately

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for processing
    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

    if results.multi_hand_landmarks:  # If hand landmarks are detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect x and y coordinates of each landmark
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates and append to data_aux
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Check if the number of features matches the model's expectation
        if len(data_aux) == model.n_features_in_:
            prediction = model.predict([np.asarray(data_aux)])  # Predict the gesture
            predicted_gesture = prediction[0]  # Get the predicted gesture

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style()
                )
            return predicted_gesture, x_, y_  # Return the predicted gesture and coordinates
        else:
            print("Skipped frame due to incorrect number of features.")
            return None, None, None

    return None, None, None  # Return None if no hand landmarks are detected

# Function to capture gestures, convert them to text, and then to speech
def gesture_to_text_and_speech():
    captured_text = ""  # String to store the captured text
    capture_start_time = None  # To store the start time of capturing
    last_gesture = None  # To track the last detected gesture

    model_dict = pickle.load(open('./model.p', 'rb'))  # Load the trained model
    model = model_dict['model']  # Extract the model from the dictionary

    cap = cv2.VideoCapture(0)  # Start video capture from the default camera

    # Initialize MediaPipe components for hand detection and drawing
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    try:
        while True:
            ret, frame = cap.read()  # Capture a frame from the video feed
            if not ret:  # Break if frame capture fails
                break

            predicted_gesture, x_, y_ = get_predicted_gesture(frame, hands, model, mp_drawing, mp_hands, mp_drawing_styles)

            if predicted_gesture:
                # Display the predicted gesture above the hand
                if x_ and y_:
                    x1 = int(min(x_) * frame.shape[1])
                    y1 = int(min(y_) * frame.shape[0])
                    
                    # Calculate time remaining
                    time_elapsed = 0
                    if predicted_gesture == last_gesture and capture_start_time:
                        time_elapsed = time.time() - capture_start_time
                    
                    # Display predicted gesture with countdown
                    display_text = f"{predicted_gesture}"
                    if predicted_gesture == last_gesture and time_elapsed > 0:
                        countdown = max(0, 2 - int(time_elapsed))
                        display_text += f" ({countdown}s)"
                    
                    cv2.putText(frame, display_text, (x1, y1 - 20), 
                                cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

                # If gesture changed, reset the timer
                if predicted_gesture != last_gesture:
                    last_gesture = predicted_gesture
                    capture_start_time = time.time()
                # If same gesture held for 2 seconds
                elif time.time() - capture_start_time >= 2:
                    if predicted_gesture == 'del':  # Delete last character
                        if captured_text:
                            captured_text = captured_text[:-1]
                            print("Deleted last character.")
                    elif predicted_gesture == 'space':  # Add a space
                        captured_text += " "
                        print("Added space.")
                    elif predicted_gesture not in ['nothing']:  # Ignore 'nothing' gesture
                        captured_text += predicted_gesture  # Append the captured letter
                        print("Captured letter:", predicted_gesture)
                    
                    # Reset timer after action
                    capture_start_time = time.time()
            else:
                # Reset if no gesture detected
                last_gesture = None
                capture_start_time = None

            # Display the captured text on the frame
            cv2.putText(frame, f"Text: {captured_text}", (25, 60), 
                        cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 0, 209), 2, cv2.LINE_AA)
            
            # Display instructions
            cv2.putText(frame, "Press 's' to speak | 'c' to clear | 'x' to exit", 
                        (25, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 209), 1, cv2.LINE_AA)
            
            cv2.imshow("Gesture-Based Text and Speech", frame)  # Show the frame with the captured text

            key = cv2.waitKey(1)  # Check for key press
            if key == ord('x'):  # Break the loop if 'x' is pressed
                break
            elif key == ord('s'):  # Press 's' to speak the captured text
                print("Speaking text:", captured_text)
                text_to_speech(captured_text)
            elif key == ord('c'):  # Press 'c' to clear the captured text
                captured_text = ""
                print("Cleared text.")

    finally:
        cap.release()  # Release the video capture object
        cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the gesture to text and speech conversion
gesture_to_text_and_speech()
