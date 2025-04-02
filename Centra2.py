# Import SMOTE from imbalanced-learn
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
import sys

sys.stdout.reconfigure(encoding='utf-8')
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("Running on GPU!")
else:
    print("No GPU found. Falling back to CPU.")

df = pd.read_csv('creditcard.csv')

print(df.head())
print(df['Class'].value_counts())

X = df.drop('Class', axis=1)
y = df['Class']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to address class imbalance
smote = SMOTE(sampling_strategy=0.01, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new class distribution
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Split the resampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Reshape data for CNN input
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

# Define the CNN inside the GPU context
with tf.device('/GPU:0'):  # Explicitly specify GPU usage
    model = tf.keras.Sequential([
        # First convolutional layer
        tf.keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2], 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        # Second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
        
        # Flatten the feature maps for the fully connected layers
        tf.keras.layers.Flatten(),
        
        # Fully connected layer with 512 units
        tf.keras.layers.Dense(512, activation='relu'),
        
        # Output layer with softmax activation
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_cnn, y_train, epochs=20, batch_size=256, validation_split=0.2, verbose=1)

# Save the model in the native Keras format
model.save('fraud_detection_cnn_model_1.h5')
print("Model saved in the native Keras format as 'fraud_detection_cnn_model.h5'")

# Evaluate the model
y_pred_probs = model.predict(X_test_cnn)
y_pred = y_pred_probs.argmax(axis=1)
print(classification_report(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred_probs[:, 1]))

# Plot training history
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
