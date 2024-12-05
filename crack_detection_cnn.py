import os
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
from PIL import Image

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(data_dir, target_size=(64, 64)):
    X = []
    y = []

    # Process cracked images
    print("Loading cracked images...")
    cracked_dir = os.path.join(data_dir, 'cracked')
    for img_name in os.listdir(cracked_dir):
        try:
            img_path = os.path.join(cracked_dir, img_name)
            img = Image.open(img_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(1)
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

    # Process intact images
    print("Loading intact images...")
    intact_dir = os.path.join(data_dir, 'intact')
    for img_name in os.listdir(intact_dir):
        try:
            img_path = os.path.join(intact_dir, img_name)
            img = Image.open(img_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img) / 255.0
            X.append(img_array)
            y.append(0)
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

    return np.array(X), np.array(y)

def create_simple_model(input_shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    return model

def plot_training_history(history):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()

def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('video_frames/video_frames')

    print(f"\nDataset size: {len(X)} images")
    print(f"Class distribution: {np.sum(y == 0)} intact, {np.sum(y == 1)} cracked")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("\nSplit sizes:")
    print(f"Training: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")

    print("\nCreating model...")
    model = create_simple_model((64, 64, 3))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )

    print("\nGenerating evaluation metrics and plots...")

    plot_training_history(history)

    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)

    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    metric_names = ['loss', 'accuracy', 'precision', 'recall']
    print("\nDetailed Test Metrics:")
    for name, value in zip(metric_names, test_metrics):
        print(f"{name.capitalize()}: {value:.4f}")

    model.save('crack_detection_model.h5')
    print("\nModel and evaluation plots have been saved.")

if __name__ == "__main__":
    main()
