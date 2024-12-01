import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Analyze the frame for emotions
    try:
        emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        print(f"Error analyzing emotions: {e}")
        continue
    
    # Extract the dominant emotion and its percentage
    dominant_emotipipon = emotion_analysis['dominant_emotion']
    emotion_probabilities = emotion_analysis['emotion']
    
    # Prepare a text for overlaying on the frame
    text = f"Emotion: {dominant_emotion}"
    
    # Display the emotion probabilities on the screen
    y0, dy = 50, 30
    for i, (emotion, score) in enumerate(emotion_probabilities.items()):
        emotion_text = f"{emotion.capitalize()}: {score:.2f}%"
        y = y0 + i * dy
        cv2.putText(frame, emotion_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the dominant emotion
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
