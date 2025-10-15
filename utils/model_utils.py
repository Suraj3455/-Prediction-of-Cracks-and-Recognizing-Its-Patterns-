import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List

def load_model():
    """Load the trained Keras model for crack detection."""
    model_path = "crack_detection_model.keras"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_crack(model, processed_image: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    Make prediction on processed image.
    
    Args:
        model: Loaded Keras model
        processed_image: Preprocessed image array
    
    Returns:
        Tuple of (predictions, predicted_class, confidence)
    """
    try:
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get predicted class index
        predicted_idx = np.argmax(predictions[0])
        
        # Class names mapping (adjust based on your model's training)
        class_names = ['Longitudinal Crack', 'No Crack', 'Oblique Crack', 'Transverse Crack']
        
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx])
        
        return predictions, predicted_class, confidence
        
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def get_model_info(model) -> dict:
    """Get information about the loaded model."""
    try:
        return {
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'total_params': model.count_params(),
            'layers': len(model.layers)
        }
    except Exception as e:
        return {'error': str(e)}
