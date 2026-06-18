import joblib
import numpy as np
from pathlib import Path
from app.config import MODEL_PATH, RIPENESS_CLASSES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MangoRipenessPredictor:
    def __init__(self):
        self.model = None
        self.processor = None  # For transformers
        self.is_loaded = False
        self.model_type = None  # Track model type
        
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Ensure MODEL_PATH is a Path object
            model_path = Path(MODEL_PATH) if isinstance(MODEL_PATH, str) else MODEL_PATH
            
            # Check model type and load accordingly
            if model_path.is_dir():
                # 🤗 Hugging Face Transformers model (folder with config.json, model.safetensors)
                from transformers import AutoModelForImageClassification, AutoImageProcessor
                self.model = AutoModelForImageClassification.from_pretrained(model_path)
                self.processor = AutoImageProcessor.from_pretrained(model_path)
                self.model_type = "transformers"
                logger.info("Loaded Hugging Face Transformers model")
                
            elif model_path.suffix in ['.pkl', '.joblib']:
                # Scikit-learn / joblib model
                self.model = joblib.load(model_path)
                self.model_type = "sklearn"
                logger.info("Loaded sklearn/joblib model")
                
            elif model_path.suffix in ['.h5', '.keras']:
                # TensorFlow/Keras model
                from tensorflow.keras.models import load_model
                self.model = load_model(model_path)
                self.model_type = "keras"
                logger.info("Loaded Keras model")
                
            elif model_path.suffix == '.pth':
                # PyTorch model
                import torch
                self.model = torch.load(model_path, map_location='cpu', weights_only=False)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                self.model_type = "pytorch"
                logger.info("Loaded PyTorch model")
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
            self.is_loaded = True
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, image):
        """Make prediction on image (PIL Image or numpy array)"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        if self.model_type == "transformers":
            return self._predict_transformers(image)
        elif self.model_type == "sklearn":
            return self._predict_sklearn(image)
        elif self.model_type in ["keras", "pytorch"]:
            return self._predict_dl(image)
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")
    
    def _predict_transformers(self, image):
        """Prediction for Hugging Face Transformers model"""
        import torch
        from PIL import Image
        
        # Ensure image is PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # The processor handles resizing, normalization, etc.
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            confidence, pred_idx = torch.max(probabilities, dim=0)
        
        # Get class name
        class_idx = pred_idx.item()
        # Prefer model-provided label order to avoid index/label mismatches.
        id2label = getattr(getattr(self.model, "config", None), "id2label", None) or {}
        raw_to_display = {
            "partially_ripe": "Partially Ripe",
            "ripe": "Ripe",
            "unripe": "Unripe",
        }
        raw_label = id2label.get(str(class_idx)) or id2label.get(class_idx)  # keys can be str or int
        class_name = raw_to_display.get(raw_label) or (
            RIPENESS_CLASSES[class_idx] if class_idx < len(RIPENESS_CLASSES) else "unknown"
        )
        
        # Build probabilities dict
        n = int(probabilities.shape[0])
        class_names = []
        for i in range(n):
            raw_i = id2label.get(str(i)) or id2label.get(i)
            class_names.append(raw_to_display.get(raw_i) or (RIPENESS_CLASSES[i] if i < len(RIPENESS_CLASSES) else str(raw_i)))
        class_probs = {
            class_names[i]: float(probabilities[i].item())
            for i in range(n)
        }
        
        return {
            "class_id": class_idx,
            "class_name": class_name,
            "confidence": float(confidence.item()),
            "all_probabilities": class_probs,
            # raw probabilities in model output order (used for downstream RF regression)
            "probabilities_array": [float(p.item()) for p in probabilities]
        }
    
    def _predict_sklearn(self, image_array):
        """Prediction for sklearn model"""
        # Flatten if needed
        flat = image_array.flatten() if image_array.ndim > 1 else image_array
        probs = self.model.predict_proba([flat])[0]
        pred_class = int(self.model.predict([flat])[0])
        
        class_probs = {
            RIPENESS_CLASSES[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        return {
            "class_id": pred_class,
            "class_name": RIPENESS_CLASSES[pred_class] if pred_class < len(RIPENESS_CLASSES) else "Unknown",
            "confidence": float(max(probs)),
            "all_probabilities": class_probs,
            "probabilities_array": [float(p) for p in probs]
        }
    
    def _predict_dl(self, image_array):
        """Prediction for Keras/PyTorch models"""
        import torch
        from PIL import Image
        import cv2

    
        # FastAPI passes a PIL image; normalize it to a numpy array that matches `predict.py`.
        if isinstance(image_array, Image.Image):
            image_array = np.array(image_array.convert("RGB"))
        elif not isinstance(image_array, np.ndarray):
            raise TypeError(f"Expected PIL.Image or numpy array, got: {type(image_array)}")

        # Match training/inference preprocessing: resize to 224x224 (RGB, uint8).
        if image_array.shape[0] != 224 or image_array.shape[1] != 224:
            image_array = cv2.resize(image_array, (224, 224), interpolation=cv2.INTER_LINEAR)
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)

        input_data = np.expand_dims(image_array, axis=0)
        
        if self.model_type == "keras":
            probs = self.model.predict(input_data, verbose=0)[0]
        else:  # pytorch
            input_tensor = torch.FloatTensor(input_data)
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0).numpy()
        
        pred_class = int(np.argmax(probs))
        
        class_probs = {
            RIPENESS_CLASSES[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        return {
            "class_id": pred_class,
            "class_name": RIPENESS_CLASSES[pred_class] if pred_class < len(RIPENESS_CLASSES) else "Unknown",
            "confidence": float(max(probs)),
            "all_probabilities": class_probs,
            # raw probabilities in model output order (used for downstream RF regression)
            "probabilities_array": [float(p) for p in probs]
        }

# Global predictor instance
predictor = MangoRipenessPredictor()