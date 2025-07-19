import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, test_size=0.2, val_size=0.1):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    
    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue  # Skip if not a directory
        
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (224, 224))  # Resize to match model input size
            images.append(image)
            labels.append(class_names.index(label))
    
    images = np.array(images) / 255.0  # Normalize images to [0, 1]
    labels = np.array(labels)

    # Split into train + validation and test sets first
    X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)

    # Now split the train + validation set into train and validation sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to the remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)