import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

data_dir = '/home/sat3812/Downloads/skincancerimages'
output_dir = '/home/sat3812/Downloads/skincancerimages/Preprocessed_data'

# Create the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and preprocess the images
def preprocess_and_save_images(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                print(f'Processing image: {img_path}')
                img = tf.keras.preprocessing.image.load_img(img_path)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                # Apply augmentation and save the processed images
                i = 0
                for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpeg'):
                    i += 1
                    if i >= 10:  # Number of augmentations per image
                        break

preprocess_and_save_images(data_dir, output_dir)
print('Processing complete.')
