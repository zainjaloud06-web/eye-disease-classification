import cv2
import numpy as np
from skimage.restoration import denoise_nl_means
import tensorflow as tf
from tensorflow import keras 
from keras.preprocessing.image import ImageDataGenerator
#from preprocessing1 import ImageDataGenerator
 
from config import *


def preprocess_image(image):
    image = cv2.resize(image, image_size)

    if len(image.shape)== 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_float=image.astype(np.float32)/255.0
    denoised_image = denoise_nl_means(image_float, h=0.8, fast_mode=True, patch_size=5, patch_distance=3)
    denoised_image= (denoised_image*255).astype(np.uint8)


    denoised_image = cv2.cvtColor(denoised_image,cv2.COLOR_GRAY2RGB)
    denoised_image= denoised_image.astype(np.float32)/255.0
    #image = image/255.0

    return denoised_image
def create_data_generators():
    train_datagen= ImageDataGenerator(
        preprocessing_function=preprocess_image,
        rotation_range=augmentation['rotation_range'],
        width_shift_range=augmentation['width_shift_range'],
        height_shift_range=augmentation['height_shift_range'],
        shear_range=augmentation['shear_range'],
        zoom_range=augmentation['zoom_range'],
        horizontal_flip=augmentation['horizontal_flip'],
        brightness_range=augmentation['brightness_range'],
        fill_mode=augmentation['fill_mode'],
        
    )

    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

    train_generator= train_datagen.flow_from_directory(TRAIN_DIR,
                                                       target_size=image_size,
                                                       batch_size=Patch_size,
                                                       class_mode='categorical',
                                                       classes=class_names,
                                                       shuffle=True
                                                       )
    val_generator= val_test_datagen.flow_from_directory(VAL_DIR,
                                                       target_size=image_size,
                                                       batch_size=Patch_size,
                                                       class_mode='categorical',
                                                       classes=class_names,
                                                       shuffle=False)
    test_generator= val_test_datagen.flow_from_directory(TEST_DIR,
                                                       target_size=image_size,
                                                       batch_size=Patch_size,
                                                       class_mode='categorical',
                                                       classes=class_names,
                                                       shuffle=False
                                                       )
    return train_generator, val_generator, test_generator
