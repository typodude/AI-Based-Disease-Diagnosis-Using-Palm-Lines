import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import os
import time

# Enable mixed precision for speed boost
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define paths
dataset_path = "C:/all pros/ML pro/AI_palm/dataset"
train_csv = os.path.join(dataset_path, "updated_train_annotations.csv")
train_img_folder = os.path.join(dataset_path, "train")

# Load dataset
df = pd.read_csv(train_csv)

# Print class distribution to check imbalance
print(df['disease_diagnosis'].value_counts())

# Image parameters
IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# Load and preprocess images
def load_and_preprocess_image(filename):
    img_path = os.path.join(train_img_folder, filename)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return img_array

# Process images
print("Processing images...")
X = np.array([load_and_preprocess_image(fname) for fname in df['filename']])
y = pd.get_dummies(df['disease_diagnosis']).values

# Check smallest class size for SMOTE
min_samples = min(df['disease_diagnosis'].value_counts())
k_neighbors = max(1, min(5, min_samples - 1))

# Apply SMOTE only if needed
if min_samples > 1:
    smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    X_resampled = X_resampled.reshape(-1, 160, 160, 3)
else:
    X_resampled, y_resampled = X, y  # Skip SMOTE if no imbalance

# Compute class weights
labels = np.argmax(y_resampled, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# Create tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices((X_resampled, y_resampled))
dataset = dataset.shuffle(len(X_resampled)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Load EfficientNetB3
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = False

# Custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(y.shape[1], activation='softmax', dtype='float32')(x)

# Build model
model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train model
start_time = time.time()
model.fit(dataset, epochs=10, validation_data=dataset, class_weight=class_weights_dict, callbacks=[early_stopping])
end_time = time.time()

# Save model
model.save("disease_model_balanced.h5")
print(f"✅ Model training complete in {round((end_time - start_time)/60, 2)} minutes.")
