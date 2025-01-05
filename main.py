import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp

# Load the model
model = tf.keras.models.load_model('SignLanguage.h5')

# Define the class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box of the hand
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Extract the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                # Resize the hand region to match the input size of the model
                img = cv2.resize(hand_img, (128, 128))
                # Normalize
                img = img / 255.0  
                # Add batch dimension
                img = np.expand_dims(img, axis=0)

                # Make prediction
                prediction = model.predict(img)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                confidence_percent = confidence * 100

                # Display the prediction on the frame
                cv2.putText(
                    img=frame, 
                    text=f"Prediction: {predicted_class} ({confidence_percent:.2f}%)", 
                    org=(x_min, y_min - 10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.9, 
                    color=(0, 255, 0), 
                    thickness=2
                )

    cv2.imshow('Sign Language Classification', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()
hands.close()