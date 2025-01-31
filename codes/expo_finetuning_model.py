from deepface import DeepFace
import cv2
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from deepface.models.demography.Emotion import load_model

model = load_model()
# Constants
IMAGE_SIZE = 48  # Image size
NUM_CLASSES = 7  # Number of emotion classes
EPOCHS = 30
BATCH_SIZE = 64

# Step 2: Freeze initial layers (only train the fully connected layers)
for layer in model.layers[:-6]:  # Freeze all layers except the last 5 layers
    layer.trainable = False

# Step 3: Modify the output layer if needed
# Example: If your dataset has different number of emotions or categories
model.layers[-1] = Dense(NUM_CLASSES, activation='softmax')

# Step 4: Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 5: Prepare your custom dataset (using the same preprocessing steps as before)
def preprocess_data(data_dir):
    images = []
    labels = []
    emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    for emotion in emotion_labels:
        emotion_path = os.path.join(data_dir, emotion)
        label = emotion_labels.index(emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            image = image.astype('float32') / 255.0
            images.append(image)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Expand dimensions to (num_samples, 48, 48, 1)
    images = np.expand_dims(images, axis=-1)
    labels = np.eye(NUM_CLASSES)[labels]  # One-hot encoding

    return images, labels


# Load the training and validation data
train_data_dir = "fer/train"
valid_data_dir = "fer/test"

X_train, y_train = preprocess_data(train_data_dir)
X_valid, y_valid = preprocess_data(valid_data_dir)

# Step 6: Data Augmentation (optional)
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator()

train_datagen.fit(X_train)
valid_datagen.fit(X_valid)

# Step 7: Train the model
class_weights = {
    0: 1.,  # angry
    1: 7.,  # disgust (focus)
    2: 3.,  # fear (focus)
    3: 1.,  # happy
    4: 1.,  # sad
    5: 1.,  # surprise
    6: 1.   # neutral
}
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=valid_datagen.flow(X_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights  # Add class weights here
)

# Step 8: Save the fine-tuned model
model.save("finetuned_model.h5")


print('DONE')