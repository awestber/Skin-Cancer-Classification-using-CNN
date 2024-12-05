import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths to your preprocessed data
train_dir = '/home/sat3812/Downloads/Preprocessed_Data/train'
test_dir = '/home/sat3812/Downloads/Preprocessed_Data/test'

# Create ImageDataGenerators for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential([
        Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            input_shape=(224, 224, 3)
        ),
        MaxPooling2D((2, 2)),
        Conv2D(
            filters=128,
            kernel_size=3,
            activation='relu'
        ),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(
            units=64,
            activation='relu'
        ),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Initialize the tuner with Random Search
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Limit the number of trials
    directory='keras_tuner',
    project_name='skin_cancer_classification'
)

# Run the hyperparameter search with fewer epochs
tuner.search(train_generator, epochs=3, validation_data=test_generator)  # Reduced to 3 epochs

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = build_model(best_hps)

# Train the model with the best hyperparameters
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=3,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Evaluate the model on the entire test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.2f}')

# Make predictions on the test data
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Calculate the accuracy score
actual_classes = test_generator.classes
accuracy = accuracy_score(actual_classes, predicted_classes)
print(f'Accuracy Score: {accuracy * 100:.2f}%')

# Calculate class distribution in the test set
class_labels = list(test_generator.class_indices.keys())

import matplotlib.pyplot as plt
import numpy as np

# Assuming 'actual_classes' and 'class_labels' are defined as per your existing code
unique, counts = np.unique(actual_classes, return_counts=True)

# Define pastel colors
pastel_colors = ['#AEC6CF', '#FFB347']  # Light blue and light orange as pastel colors

# Plotting the bar graph with pastel colors
plt.figure(figsize=(8, 6))
plt.bar(class_labels, counts, color=pastel_colors)
plt.title('Distribution of Benign and Malignant Results in Test Set')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.show()
