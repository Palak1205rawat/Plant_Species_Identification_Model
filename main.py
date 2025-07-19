import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.preprocessing import load_data
from models.cnn_model import create_model

# Define path to your dataset with all images.
data_dir = r'C:\Users\ASUS\OneDrive\Desktop\mini_project\Plant_Species_Indentification\data'  # Update this path based on your directory structure

# Load data and split into training, validation and testing sets.
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_dir)

# Create ImageDataGenerator instances for training and testing.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training and validation/testing.
training_set = train_datagen.flow(X_train, y_train, batch_size=32)
validation_set = test_datagen.flow(X_val, y_val, batch_size=32)
test_set = test_datagen.flow(X_test, y_test, batch_size=32)

# Model selection and training (example with ResNet50).
model_types = ['ResNet', 'VGG', 'EfficientNet']
for model_type in model_types:
    print(f'Training {model_type}...')
    
    model = create_model(model_type)
    
    model.fit(training_set,
              epochs=10,
              validation_data=validation_set)

    # Evaluate the model after training.
    y_pred_probs = model.predict(test_set)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted classes.

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'{model_type} Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

# Optionally implement Grad-CAM visualization here if needed.