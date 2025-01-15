from deepface.models.demography.Emotion import load_model
import cv2
import numpy as np
import time
import random
# from pythonosc import udp_client


# client = udp_client.SimpleUDPClient("145.93.53.187", 9000)  # Send messages to localhost on port 9000



# Load the custom emotion recognition model
model = load_model('finetuned_model.h5')

# Define emotion labels corresponding to the model's output
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define colors for each emotion
colours = [(34, 34, 231),       # Red for 'angry'
           (62, 178, 112),      # Green for 'disgust'
           (62, 31, 56),        # Purple for 'fear'
           (0, 240, 255),       # Yellow for 'happy'
           (144, 75, 0),        # Blue for 'sad'
           (171, 169, 211),     # Pink for 'surprise'
           (128, 128, 128)]     # Grey for 'neutral'

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam is open. Press 'q' to exit.")

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up full-screen window for display
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
last_print_time = time.time()  # Time of the last print
print_interval = 5  # Interval in seconds to print output
# Real-time emotion detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = frame[y:y + h, x:x + w]

        try:
            # Preprocess the face region for model input
            face_resized = cv2.resize(face_region, (48, 48))  # Adjust size to match model's input
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            face_normalized = face_gray / 255.0  # Normalize pixel values
            face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))  # Reshape for model input

            # Predict the emotion using the custom model
            predictions = model.predict(face_reshaped, verbose=0)

            # Print all emotions and their percentages in the terminal
            # print("Emotion Percentages:")
            # for i, label in enumerate(emotion_labels):
            #     percentage = predictions[0][i] * 100
            #     print(f"{label}: {percentage:.2f}%")
            # print("------")  # Separator between frames

            # Get the dominant emotion
            dominant_emotion_idx = np.argmax(predictions)
            dominant_emotion = emotion_labels[dominant_emotion_idx]
            emotion_confidence = predictions[0][dominant_emotion_idx] * 100

            # Get the corresponding color for the dominant emotion
            dominant_color = colours[dominant_emotion_idx]

            # Prepare text to overlay on the frame
            emotion_text = f"{dominant_emotion}: {emotion_confidence:.2f}%"

            # Overlay emotion text on the top-left corner
            (text_width, text_height), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = 10
            text_y = text_height + 10

            # Draw the text box background
            cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5), dominant_color, -1)

            # Overlay text
            cv2.putText(frame, emotion_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display all emotion scores on the top-left corner under the dominant emotion
            y_offset = text_y + 30
            for i, label in enumerate(emotion_labels):
                percentage = predictions[0][i] * 100
                text = f"{label}: {percentage:.2f}%"
                color = colours[i]
                # Calculate position of the text box for each emotion
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                # Draw the colored rectangle (text box)
                cv2.rectangle(frame, (10 - 5, y_offset - text_height - 5),
                              (10 + text_width + 5, y_offset + 5), color, -1)
                # Overlay the emotion text
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                y_offset += 25



            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                print(dominant_emotion)  # Print the dominant emotion
                last_print_time = current_time  # Update the last print time


            # client.send_message("/heightofwave", float(output))
            # time.sleep(0.1)  # Delay to send every 100ms, for example

        except Exception as e:
            print(f"Error analyzing frame: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
