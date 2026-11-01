import os
import tensorflow as tf
import numpy as np
from visualization import plot_confusion_matrix, save_classification_report
from config import class_names, MODEL_DIR


def evaluate_model():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR,'best_model.keras'))
    from preprocessing1 import create_data_generators
    _, _, test_generator = create_data_generators()

    evaluation = model.evaluate(test_generator)
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]*100:.2f}%")

    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    plot_confusion_matrix(y_true,y_pred_classes)

    save_classification_report(y_true, y_pred_classes, os.path.join(MODEL_DIR,'classification_report.csv'))
if __name__ == '__main__':
    evaluate_model()