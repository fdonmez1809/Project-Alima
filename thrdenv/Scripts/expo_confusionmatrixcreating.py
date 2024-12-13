from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import seaborn as sn
import pandas as pd
import numpy as np
import os
import cv2

def plot_cm(cm, name):
# Convert to a DataFrame for easier visualization
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    plt.figure(figsize=(10, 8))
    sn.heatmap(cm, annot=True, fmt=".2f",
            xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix for Emotion Detection of {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    
def plot_perc_cm(cm, name):
# Convert to a DataFrame for easier visualization
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    df = pd.DataFrame(cm, index=labels, columns=labels)
    # Calculate the row sums
    row_sums = df.sum(axis=1)
    # Calculate the percentage for each cell by dividing by the row sum and multiplying by 100
    percentage_matrix = df.div(row_sums, axis=0) * 100  

    plt.figure(figsize=(10, 8))
    sn.heatmap(percentage_matrix, annot=True, fmt=".2f",

            xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Percentage Confusion Matrix for Emotion Detection of {name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    
def confusing(model,name):
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Initialize lists to store predictions and true labels
    predict_labels = []
    true_labels = []

    # Folder containing the test dataset
    test_folder = "fer/test"
    if name != 'DeepFace':
        # Iterate over the test dataset
        for emotion in os.listdir(test_folder):
            emotion_folder = os.path.join(test_folder, emotion)
            if os.path.isdir(emotion_folder):
                for img_file in os.listdir(emotion_folder):
                    img_path = os.path.join(emotion_folder, img_file)
                    
                    # Preprocess the image before passing it to the model
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # Resize image to 48x48 (the expected input size for the emotion model)
                    img = cv2.resize(img, (48, 48))
                    # Normalize the image
                    img = img.astype('float32') / 255.0
                    # Expand dimensions to (1, 48, 48, 1) for model input
                    img = np.expand_dims(img, axis=-1)
                    
                    # Predict emotion using your custom-trained model
                    prediction = model.predict(np.expand_dims(img, axis=0),verbose=1)  # Add batch dimension
                    predicted_emotion = emotion_labels[np.argmax(prediction)]  # Get the emotion label with highest probability
                    
                    # Append the predicted emotion and true label
                    predict_labels.append(predicted_emotion)
                    true_labels.append(emotion)
    else:
        for emotion in os.listdir(test_folder):
            emotion_folder = os.path.join(test_folder, emotion)
            if os.path.isdir(emotion_folder):
                for img_file in os.listdir(emotion_folder):
                    img_path = os.path.join(emotion_folder, img_file)
                    img = cv2.imread(img_path)
                    
                    
                    analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                    emotion_data = analysis[0]['emotion']
                    predict_emotion = max(emotion_data, key=emotion_data.get)
                    predict_labels.append(predict_emotion)
                    true_labels.append(emotion)
                        
                
    
    cm = confusion_matrix(true_labels, predict_labels, labels=emotion_labels)
    return cm
