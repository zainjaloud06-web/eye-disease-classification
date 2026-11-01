import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import pandas as pd
from config import *
def plot_training_history(history_dict):  # تغيير الاسم ليكون واضحًا
    """
    history_dict: قاموس يحتوي على:
        - loss: قائمة خسائر التدريب
        - accuracy: قائمة دقة التدريب
        - val_loss: قائمة خسائر التحقق
        - val_accuracy: قائمة دقة التحقق
    """
    plt.figure(figsize=(12, 6))
    
    # رسم الدقة
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['accuracy'], label='Train')
    plt.plot(history_dict['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # رسم الخسارة
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train')
    plt.plot(history_dict['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

################################################################
def plot_training_historyy(history):
    print("Available keys in history:", history.history.keys())
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    accuracy_keys = ['accuracy', 'acc', 'binary_accuracy', 'categorical_accuracy']
    val_accuracy_keys = ['val_'+k for k in accuracy_keys]
    
    for key in accuracy_keys:
        if key in history.history:
            plt.plot(history.history[key], label=f'Training {key}')
    for key in val_accuracy_keys:
        if key in history.history:
            plt.plot(history.history[key], label=f'Validation {key}')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #plt.legend()
    plt.legend(["Train","Val"], loc='upper right')
    plt.subplot(1, 2, 2)
    loss_keys = ['loss', 'loss', 'binary_loss', 'categorical_loss']
    val_loss_keys = ['val_'+k for k in loss_keys]
    
    for key in loss_keys:
        if key in history.history:
            plt.plot(history.history[key], label=f'Training {key}')
    for key in val_loss_keys:
        if key in history.history:
            plt.plot(history.history[key], label=f'Validation {key}')
 
    #plt.plot(history.history['loss'], label='Training loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.legend()
    plt.legend(["Train","Val"], loc='upper right')
    plt.tight_layout()
    plt.show()


    #plt.plot(history.history['categorical_ accuracy'], label='Training accuracy')
    #plt.plot(history.history['val_categorical_ accuracy'], label='Validation accuracy')
    #plt.title('Training and Validation accuracy')
    #plt.xlabel('Epoch')
    #plt.ylabel('accuracy')
    #plt.legend()

    #plt.subplot(1, 2, 1)
    #plt.plot(history.history['loss'], label='Training loss')
    #plt.plot(history.history['val_loss'], label='Validation loss')
    #plt.title('Training and Validation loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm= confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('predicted label')
    plt.ylabel('True label')
    plt.show()

def plot_sample_images(generator,num_sumples=8):
    x_batch,y_batch=next(generator)
    plt.figure(figsize=(12,8))
    for i in range(num_sumples):
        plt.subplot(2,4,i+1)
        plt.imshow(x_batch[i])
        plt.title(class_names[np.argmax(y_batch[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_classification_report(y_true, y_pred, file_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict= True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_path,index=True)
    print(f"Classification report saved to {file_path}")

