import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os

# ---------- CONFIG ----------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
DATASET_DIR = 'dataset'  # โฟลเดอร์หลักที่มี clean/ dirty

# ---------- LOAD DATA WITH AUGMENTATION ----------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ---------- BUILD MODEL ----------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ---------- TRAIN ----------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ---------- SAVE MODEL (.h5) ----------
model.save("hand_clean_model.h5")

# ---------- PLOT HISTORY ----------
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

# ---------- EXPORT TO .tflite + OPTIMIZATION ----------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()

with open("hand_clean_model_optimized.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved: hand_clean_model_optimized.tflite")

# ---------- CREATE LABEL FILE ----------
with open("labels.txt", "w") as f:
    f.write("clean\ndirty")
