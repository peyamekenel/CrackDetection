import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell for title and setup
cells = []

# Title and setup
cells.append(nbf.v4.new_markdown_cell("""# Concrete Crack Detection using CNN

This notebook implements a Convolutional Neural Network (CNN) for detecting cracks in concrete surfaces from video frames.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Python Imaging Library)
- Seaborn

## Directory Structure
The code expects the following directory structure:

C:/Users/Ankaref/video_frames/
    - cracked/    (Contains cracked concrete images)
    - intact/     (Contains intact concrete images)

## Setup and Dependencies"""))

# Imports
cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from PIL import Image"""))

# Data loading section
cells.append(nbf.v4.new_markdown_cell("""## Data Loading and Preprocessing

Load images from the specified directories and preprocess them for training."""))

data_loading_code = """def load_and_preprocess_data(base_path="C:/Users/Ankaref/video_frames"):
    \"\"\"Load and preprocess images from cracked and intact directories.\"\"\"
    cracked_dir = os.path.join(base_path, "cracked")
    intact_dir = os.path.join(base_path, "intact")

    # Image parameters
    img_size = (64, 64)

    # Lists to store images and labels
    images = []
    labels = []

    # Load cracked images (label 1)
    for img_name in os.listdir(cracked_dir):
        img_path = os.path.join(cracked_dir, img_name)
        try:
            img = Image.open(img_path)
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(1)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    # Load intact images (label 0)
    for img_name in os.listdir(intact_dir):
        img_path = os.path.join(intact_dir, img_name)
        try:
            img = Image.open(img_path)
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(0)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels)

# Load the data
X, y = load_and_preprocess_data()
print(f"Total images loaded: {len(X)}")
print(f"Cracked images: {sum(y)}")
print(f"Intact images: {len(y) - sum(y)}")"""

cells.append(nbf.v4.new_code_cell(data_loading_code))

# Model creation section
cells.append(nbf.v4.new_markdown_cell("## Create and Configure the CNN Model"))

model_creation_code = """def create_model(input_shape):
    \"\"\"Create a simple CNN model for binary classification.\"\"\"
    model = Sequential([
        # First convolutional block
        Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second convolutional block
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),

        # Dense layers
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    return model

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and compile the model
model = create_model((64, 64, 3))
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Display model summary
model.summary()"""

cells.append(nbf.v4.new_code_cell(model_creation_code))

# Training section
cells.append(nbf.v4.new_markdown_cell("## Train the Model"))

training_code = """# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=8,
    callbacks=[early_stopping]
)"""

cells.append(nbf.v4.new_code_cell(training_code))

# Evaluation section
cells.append(nbf.v4.new_markdown_cell("## Evaluate Model Performance"))

evaluation_code = """def plot_training_metrics(history):
    \"\"\"Plot training metrics.\"\"\"
    metrics = ["accuracy", "loss", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for idx, metric in enumerate(metrics):
        row = idx // 2
        col = idx % 2

        axes[row, col].plot(history.history[metric])
        axes[row, col].plot(history.history[f"val_{metric}"])
        axes[row, col].set_title(f"Model {metric.capitalize()}")
        axes[row, col].set_ylabel(metric.capitalize())
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].legend(["Train", "Validation"])

    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_metrics(history)

# Generate predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Print detailed metrics
test_metrics = model.evaluate(X_test, y_test, verbose=1)
metric_names = ["loss", "accuracy", "precision", "recall"]
print("\\nDetailed Test Metrics:")
for name, value in zip(metric_names, test_metrics):
    print(f"{name.capitalize()}: {value:.4f}")"""

# Add prediction cell section
cells.append(nbf.v4.new_markdown_cell("""## Make Predictions on New Images

This section demonstrates how to use the trained model to make predictions on new concrete images."""))

prediction_cell = """# Load and compile the saved model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load the model
loaded_model = load_model("crack_detection_model.h5")

# Recompile the model with the same configuration as during training
loaded_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")]
)

def predict_image(image_path, model):
    \"\"\"Make prediction on a single image.\"\"\"
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    raw_prediction = model.predict(img_array)[0][0]
    # For cracked (class 1), use raw prediction
    # For intact (class 0), use 1 - raw prediction as confidence
    is_cracked = raw_prediction > 0.5
    confidence = raw_prediction if is_cracked else 1 - raw_prediction
    return is_cracked, confidence

def visualize_prediction(image_path, is_cracked, confidence):
    \"\"\"Visualize the image with its prediction.\"\"\"
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {'Cracked' if is_cracked else 'Intact'} ({confidence:.2%} confidence)")
    plt.axis('off')
    plt.show()

# Create a small evaluation set to build metrics
base_path = "C:/Users/Ankaref/video_frames"
eval_images = []
eval_labels = []

# Load a few images for evaluation
for img_name in os.listdir(os.path.join(base_path, "cracked"))[:2]:
    img_path = os.path.join(base_path, "cracked", img_name)
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    eval_images.append(img_array)
    eval_labels.append(1)

for img_name in os.listdir(os.path.join(base_path, "intact"))[:2]:
    img_path = os.path.join(base_path, "intact", img_name)
    img = Image.open(img_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    eval_images.append(img_array)
    eval_labels.append(0)

# Convert to numpy arrays
eval_images = np.array(eval_images)
eval_labels = np.array(eval_labels)

# Evaluate model to build metrics
print("Building metrics with evaluation set...")
loaded_model.evaluate(eval_images, eval_labels, verbose=1)
print("Metrics built successfully!")

# Test predictions on both cracked and intact images
print("\\nTesting predictions on both classes:")
for folder in ["cracked", "intact"]:
    print(f"\\nTesting {folder} images:")
    for i, img_name in enumerate(os.listdir(os.path.join(base_path, folder))[:2]):
        img_path = os.path.join(base_path, folder, img_name)
        is_cracked, conf = predict_image(img_path, loaded_model)
        print(f"Image {i+1}: {img_name} - {'Cracked' if is_cracked else 'Intact'} ({conf:.2%} confidence)")"""

cells.append(nbf.v4.new_code_cell(prediction_cell))

# Add cells to notebook

cells.append(nbf.v4.new_code_cell(evaluation_code))

# Save model section
cells.append(nbf.v4.new_markdown_cell("## Save the Model"))

cells.append(nbf.v4.new_code_cell("""# Save the trained model
model.save("crack_detection_model.h5")
print("Model saved as 'crack_detection_model.h5'")"""))

# Add prediction cell section
cells.append(nbf.v4.new_markdown_cell("""## Make Predictions on New Images

This section demonstrates how to use the trained model to make predictions on new concrete images."""))

cells.append(nbf.v4.new_code_cell("""def predict_image(image_path, model):
    \"\"\"Make prediction on a single image.\"\"\"
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    raw_prediction = model.predict(img_array)[0][0]
    # For cracked (class 1), use raw prediction
    # For intact (class 0), use 1 - raw prediction as confidence
    is_cracked = raw_prediction > 0.5
    confidence = raw_prediction if is_cracked else 1 - raw_prediction
    return is_cracked, confidence

def visualize_prediction(image_path, is_cracked, confidence):
    \"\"\"Visualize the image with its prediction.\"\"\"
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Prediction: {'Cracked' if is_cracked else 'Intact'} ({confidence:.2%} confidence)")
    plt.axis('off')
    plt.show()

# Example: Make prediction on a test image
# You can replace this path with any image path in your directory
test_image_path = os.path.join("C:/Users/Ankaref/video_frames/cracked", os.listdir("C:/Users/Ankaref/video_frames/cracked")[0])

# Load the saved model
from tensorflow.keras.models import load_model
loaded_model = load_model("crack_detection_model.h5")

# Make and visualize prediction
is_cracked, confidence = predict_image(test_image_path, loaded_model)
visualize_prediction(test_image_path, is_cracked, confidence)

print(f"Prediction confidence: {confidence:.2%}")
print(f"Classification: {'Cracked' if is_cracked else 'Intact'}")

# Example of how to predict multiple images
print("\\nPredicting multiple images:")
# Test both cracked and intact images
for folder in ["cracked", "intact"]:
    print(f"\\nTesting {folder} images:")
    for i, img_name in enumerate(os.listdir(os.path.join("C:/Users/Ankaref/video_frames", folder))[:2]):
        img_path = os.path.join("C:/Users/Ankaref/video_frames", folder, img_name)
        is_cracked, conf = predict_image(img_path, loaded_model)
        print(f"Image {i+1}: {img_name} - {'Cracked' if is_cracked else 'Intact'} ({conf:.2%} confidence)")"""))

# Add cells to notebook
nb['cells'] = cells

# Set the kernel info
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}

# Write the notebook to a file
with open('crack_detection.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
