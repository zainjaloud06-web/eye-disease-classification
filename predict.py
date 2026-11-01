import os
import tensorflow as tf
from keras import models
import numpy as np
import cv2
from config import class_names, MODEL_DIR, image_size
from preprocessing1 import preprocess_image

def load_model():
    model_path = os.path.join(MODEL_DIR, 'best_model.keras')
    if not os.path.exists(model_path):
        raise FileExistsError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_single_image(image_path, model):

    image= cv2.imread(image_path)
    if image is None:
        raise ValueError(f"could not read image at {image_path}")
    processed_image = preprocess_image(image)
    processed_image= np.expand_dims(processed_image, axis=0)

    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)*100

    return predicted_class, confidence, predictions

def predict_multiple_images(image_dir) :
    model= load_model()
    results = []
     
    image_extensions = ['.jpg','.jpeg','.png']
    image_paths = [os.path.join(image_dir,f) for f in os.listdir(image_dir)
                    if os.path.splitext(f)[1].lower() in image_extensions]
    if not image_paths:
        print(f"No images found in {image_dir}")
        return
    for image_path in image_paths:
        try:
            predicted_class, confidence, _ = predict_single_image(image_path, model)
            results.append({'image': os.path.basename(image_path),
                            'prediction': predicted_class,
                             'confidence': f"{confidence:.2f}%" })
            print(f"Image:{os.path.basename(image_path)} -predicted: {predicted_class} ({confidence:.2f}%)")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    return results

if __name__ == '__main':
    import argparse

    parser = argparse.ArgumentParser(description='Predict retinal diseases from OCT images')
    parser.add_argument('--image', type=str, help='path to single image for prediction')
    parser.add_argument('--dir',type=str, help='path to directory containing images for prediction')
    args = parser.parse_args()
    model = load_model()

    if args.image:
        predicted_class, confidence, _ = predict_single_image(args.image, model)
        print(f"\nPrediction Result:")
        print(f"image: {os.path.basename(args.image)}")
        print(f"Predicted class: {predicted_class}")
        print(f"confidence: {confidence:.2f}%")
    elif args.dir:
        predict_multiple_images(args.dir)
    else:
        print('please provide either --image or --dir ariment')


predict_multiple_images(r"C:\Users\xayn\Desktop\deeplearning project\pppp")
        
