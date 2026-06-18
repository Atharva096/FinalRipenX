from PIL import Image
import io
import numpy as np
from backend.app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE

class ImageProcessor:
    @staticmethod
    def validate_file(file, filename: str) -> tuple[bool, str]:
        ext = f".{filename.split('.')[-1].lower()}"
        if ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {ext}"
        
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)  # Reset
        
        if size > MAX_FILE_SIZE:
            return False, f"File too large: {size/1024/1024:.2f}MB (max 10MB)"
        
        return True, "OK"
    
    @staticmethod
    def process_image(file):
        """
        Process uploaded image file and return as numpy array
        For transformers models, return raw RGB values (not normalized)
        """
        # Read file bytes
        file_bytes = file.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        # Resize to model's expected size (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        
        # Convert to numpy array (keep as uint8, not normalized)
        image_array = np.array(image)
        
        return image_array
    
    @staticmethod
    def preprocess_image(file_bytes, target_size=(224, 224)):
        """Preprocess image for model prediction (legacy method)"""
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image)
        
        # Normalize if needed
        image_array = image_array.astype('float32') / 255.0
        
        return image_array