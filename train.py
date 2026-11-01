#import os
#import tensorflow as tf
#from keras import models, layers, callbacks, optimizers
#from keras._tf_keras.keras.models import Sequential
#from keras._tf_keras.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
#from keras._tf_keras.keras.optimizers import Adam
#from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
#from utils.preprocessing import create_data_generators
#from utils.visualization import plot_training_history, plot_sample_images
#from config import *
import os
import tensorflow as tf
#from tensorflow import keras
from keras import models
from keras import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

#from keras import Conv2D
#from keras import MaxPool2D, BatchNormalization, Flatten, Dropout , Dense
# MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
#from keras import optimizers
from keras.optimizers import Adam
#from keras._tf_keras.keras.optimizers import Adam

#from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
#from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from preprocessing1 import create_data_generators
from visualization import plot_training_history, plot_sample_images
from config import *
train_generator, val_generator, test_generator = create_data_generators()

def create_Alexnet_model():

    model= Sequential()

    model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(image_size[0],image_size[1],3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(1,1)))

    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
def train_model(resume=False):

    train_generator, val_generator, test_generator = create_data_generators()
    if train_generator.samples==0 or val_generator.samples==0 or test_generator.samples==0:
        raise ValueError("there  is no samples")
    model_path = os.path.join(MODEL_DIR,'best_model.keras')
    checkpoint_path = os.path.join(MODEL_DIR,'trainig_checkpoint.keras')
    log_path = os.path.join(MODEL_DIR,'trainig_log.csv')
    model=None
    intial_epoch=0
    if resume and os.path.exists(checkpoint_path) :
        print("\n resume training from the last point.")
        model= tf.keras.models.load_model(checkpoint_path) 
        intial_epoch=0
        if os.path.exists(log_path):
            import pandas as pd
            log_df= pd.read_csv(log_path)
            intial_epoch= len(log_df)
            print(f"resume training from epoch {intial_epoch}")
    else:
        print("\n starting new training..")
        model= create_Alexnet_model()
        intial_epoch=0

        if model:
            model.summary()
    if ModelCheckpoint is None:
        raise RuntimeError



     #   plot_sample_images(train_generator)
     #   model.summary()
                 
    #plot_sample_images(train_generator)
    #model= model.summary()

    callbacks = [ModelCheckpoint(model_path,monitor='val_accuracy',save_best_only=True, mode= 'max',verbose=1),
                 ModelCheckpoint(checkpoint_path,save_freq="epoch",save_weights_only=False),
                 EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
                 CSVLogger(log_path, append = resume)]
    
    history= model.fit(train_generator,steps_per_epoch=train_generator.samples // Patch_size,
                        epochs=epochs,initial_epoch=intial_epoch,
                        validation_data=val_generator,validation_steps=val_generator.samples // Patch_size,
                        callbacks=callbacks,
                        verbose=1)
    print(history.history)
    history_data = {'loss':log_df['loss'].tolist(),
                    'accuracy':log_df['accuracy'].tolist(),
                    'val_loss':log_df['val_loss'].tolist(),
                    'val_accuracy':log_df['val_accuracy'].tolist()}
    model.save(os.path.join(MODEL_DIR,'final_model.keras'))

    #plot_training_history(history)
    plot_training_history(history_data)
    plot_sample_images(train_generator)
    return model, history, test_generator

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    resume_training =True

    

     

    model,history,testgenerator = train_model(resume=resume_training)
#create_Alexnet_model()
    