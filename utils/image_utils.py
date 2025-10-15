import numpy as np
from PIL import Image
import cv2
from typing import Tuple

def validate_image(image: Image.Image) -> Tuple[bool, str]:
    """
    Validate uploaded image for processing.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Check image size
        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image is too small. Minimum size is 50x50 pixels."
        
        if width > 5000 or height > 5000:
            return False, "Image is too large. Maximum size is 5000x5000 pixels."
        
        # Check image mode
        if image.mode not in ['RGB', 'L', 'RGBA']:
            return False, "Unsupported image format. Please use RGB, grayscale, or RGBA images."
        
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Preprocess image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def enhance_image_contrast(image: Image.Image) -> Image.Image:
    """
    Enhance image contrast for better crack detection.
    
    Args:
        image: PIL Image object
    
    Returns:
        Enhanced PIL Image object
    """
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_channel = clahe.apply(l_channel)
        
        # Merge channels and convert back
        lab = cv2.merge((l_channel, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert back to PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        return enhanced_pil
        
    except Exception as e:
        # Return original image if enhancement fails
        return image

def extract_image_features(image: Image.Image) -> dict:
    """
    Extract basic features from image for analysis.
    
    Args:
        image: PIL Image object
    
    Returns:
        Dictionary of image features
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Convert to grayscale for additional analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate contrast and texture measures
        contrast = np.std(gray)
        
        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'contrast': float(contrast),
            'dimensions': image.size,
            'aspect_ratio': image.size[0] / image.size[1]
        }
        
    except Exception as e:
        return {'error': str(e)}
