import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


from preprocessing1 import preprocess_image


from config import image_size, augmentation


IMG_PATH = r"C:\Users\xayn\Desktop\deeplearning project\OCTDATA\OCTDATA\test\DME\DME-462675-13.jpeg"
image= cv2.imread(IMG_PATH)
if image is None:
    raise ValueError(f"could not read image at {IMG_PATH}")
processed_image = preprocess_image(image)
plt.figure(figsize=(6, 6))
plt.imshow(processed_image)
plt.title("filter and resize")
plt.axis("off")
plt.show()

Datagen = ImageDataGenerator(
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


Img_resized_for_gen = cv2.resize(image, image_size)
X = np.expand_dims(Img_resized_for_gen, axis=0)  


Aug_iter = Datagen.flow(X, batch_size=1, seed=123)  
N = 8  

plt.figure(figsize=(12, 6))
for i in range(N):
    
    Aug_img = next(Aug_iter)[0] 
    Ax = plt.subplot(2, 4, i + 1)
    Ax.imshow(Aug_img)
    Ax.set_title(f"Aug #{i+1}")
    Ax.axis("off")

plt.suptitle("aug")
plt.tight_layout()
plt.show()

