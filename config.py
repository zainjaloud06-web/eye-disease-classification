import os

BASE_DIR = r"C:\Users\xayn\Desktop\deeplearning project\OCTDATA"
DATA_DIR = os.path.join(BASE_DIR,"OCTDATA")
TRAIN_DIR = os.path.join(DATA_DIR,"train")
TEST_DIR = os.path.join(DATA_DIR,"test")
VAL_DIR = os.path.join(DATA_DIR,"valid")
MODEL_DIR = os.path.join(BASE_DIR,"models","saved_models")

class_names = ['CNV','DME','DRUSEN','NORMAL']
num_classes = len(class_names)
image_size = (224,224)
Patch_size = 32

epochs = 20
learning_rate= 0.0001
patience = 7

augmentation = {'rotation_range':10,
                'width_shift_range':0.1,
                'height_shift_range':0.1,
                'shear_range':0.2,
                'zoom_range':(0.73,0.9),
                'horizontal_flip':True,
                'brightness_range':(0.55,0.9),
                'fill_mode':'constant'
                }