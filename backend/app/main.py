from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import logging
from datetime import datetime
import tempfile
import numpy as np
import joblib
import sys
from pathlib import Path
import random

from app.config import MODEL_PATH, BASE_DIR, RF_MODEL_PATH, BASE_DATASET_DIR, ALLOWED_EXTENSIONS
from app.schemas.response import PredictionResponse, ErrorResponse
from app.utils.image_processor import ImageProcessor
from app.models.predictor import predictor

# Ensure repo root is importable so we can import `feature_extraction.py`
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from backend.feature_extraction import extract_combined_features
from app.services.harvest import predict_harvest
from backend.export_recommendation import (
    DEFAULT_CULTIVAR,
    format_regulatory_compliance_block,
    get_mandatory_regulatory_compliance,
    recommend_export_destination,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mango Ripeness Detection API",
    description="API for detecting mango ripeness using machine learning",
    version="1.0.0"
)

rf_regressor = None
example_files_by_folder: dict[str, list[Path]] = {}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        predictor.load_model()
        logger.info("Model loaded successfully on startup")

        # Load RandomForest regressor used for harvest-time estimation
        global rf_regressor
        rf_regressor = joblib.load(Path(RF_MODEL_PATH))
        logger.info("RandomForest regressor loaded successfully on startup")

        # Load a few sample images per class folder for the frontend
        global example_files_by_folder
        example_files_by_folder = {}
        dataset_root = Path(BASE_DATASET_DIR)
        class_folders = ["partially_ripe", "ripe", "unripe"]
        for folder in class_folders:
            class_dir = dataset_root / folder
            if not class_dir.exists() or not class_dir.is_dir():
                continue
            files = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
            ]
            # Keep it light: frontend just needs a few images.
            files = sorted(files, key=lambda p: p.name)[:20]
            if files:
                example_files_by_folder[folder] = files
        logger.info(f"Loaded sample images for classes: {list(example_files_by_folder.keys())}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Mango Ripeness Detection API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if predictor.is_loaded else "unhealthy",
        "model_loaded": predictor.is_loaded,
        "timestamp": datetime.now()
    }

@app.get("/example/{ripeness_folder}")
def get_example_image(ripeness_folder: str):
    """
    Return a sample image for a given ripeness folder.
    Used by the React frontend to show class-specific example thumbnails.
    """
    folder = ripeness_folder.strip().lower()
    allowed = {"partially_ripe", "ripe", "unripe"}
    if folder not in allowed:
        raise HTTPException(status_code=400, detail="Invalid ripeness folder. Use partially_ripe, ripe, or unripe.")

    files = example_files_by_folder.get(folder)
    if not files:
        raise HTTPException(status_code=404, detail=f"No example images found for {folder}")

    chosen = random.choice(files)
    return FileResponse(chosen)

@app.post("/predict", response_model=PredictionResponse)
async def predict_ripeness(
    file: UploadFile = File(...),
    temperature: float = Form(35.0),
    humidity: float = Form(50.0),
    cultivar: str = Form(DEFAULT_CULTIVAR),
):
    """
    Predict mango ripeness and estimate harvest time
    """
    try:
        # Validate file
        is_valid, message = ImageProcessor.validate_file(file.file, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Read image bytes (don't preprocess - let transformers handle it)
        file_bytes = await file.read()
        
        # Convert to PIL Image for predictor
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        
        # Make prediction (pass PIL Image, not numpy array)
        prediction = predictor.predict(image)

        # Extract HSV+LBP features for RF regression.
        # `extract_combined_features` expects a file path, so we persist the upload to a temp file.
        suffix = Path(file.filename).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()  # Important on Windows: allow OpenCV to read the file
            texture_color_features = extract_combined_features(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        if texture_color_features is None:
            raise ValueError("Failed to extract texture/color features from image")

        cnn_probs = np.array(prediction["probabilities_array"], dtype=np.float32)
        harvest_days, _, message = predict_harvest(
            rf_regressor,
            cnn_probs,
            texture_color_features,
            float(temperature),
            float(humidity),
            prediction["class_name"],
        )

        cultivar_used = (cultivar or "").strip() or DEFAULT_CULTIVAR
        export_dest, export_logistics = recommend_export_destination(
            harvest_days, cultivar_used
        )
        regulatory_actions = get_mandatory_regulatory_compliance(export_dest)
        regulatory_block = format_regulatory_compliance_block(export_dest)
        message = (
            f"{message}\n"
            f"Recommended Export Destination: {export_dest}\n"
            f"Logistics Action Required: {export_logistics}"
        )
        if regulatory_block:
            message = f"{message}\n{regulatory_block}"

        return PredictionResponse(
            success=True,
            filename=file.filename,
            ripeness_class=prediction["class_name"],
            confidence=prediction["confidence"],
            confidence_percentage={
                k: f"{v*100:.2f}%" 
                for k, v in prediction["all_probabilities"].items()
            },
            harvest_estimate_days=harvest_days,
            cultivar=cultivar_used,
            recommended_export_destination=export_dest,
            export_logistics_action=export_logistics,
            mandatory_regulatory_compliance=regulatory_actions or None,
            processed_at=datetime.now(),
            message=message,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def batch_predict(
    files: list[UploadFile] = File(...),
    temperature: float = Form(35.0),
    humidity: float = Form(50.0),
    cultivar: str = Form(DEFAULT_CULTIVAR),
):
    """Batch prediction for multiple images"""
    from PIL import Image
    import io
    
    results = []
    for file in files:
        try:
            is_valid, message = ImageProcessor.validate_file(file.file, file.filename)
            if not is_valid:
                results.append({"filename": file.filename, "error": message})
                continue
            
            file_bytes = await file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            
            prediction = predictor.predict(image)

            suffix = Path(file.filename).suffix or ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            try:
                tmp.write(file_bytes)
                tmp.flush()
                tmp.close()  # Important on Windows: allow OpenCV to read the file
                texture_color_features = extract_combined_features(tmp.name)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

            if texture_color_features is None:
                raise ValueError("Failed to extract texture/color features from image")

            cnn_probs = np.array(prediction["probabilities_array"], dtype=np.float32)
            harvest_days, _, message = predict_harvest(
                rf_regressor,
                cnn_probs,
                texture_color_features,
                float(temperature),
                float(humidity),
                prediction["class_name"],
            )

            cultivar_used = (cultivar or "").strip() or DEFAULT_CULTIVAR
            export_dest, export_logistics = recommend_export_destination(
                harvest_days, cultivar_used
            )
            regulatory_actions = get_mandatory_regulatory_compliance(export_dest)
            regulatory_block = format_regulatory_compliance_block(export_dest)
            message = (
                f"{message}\n"
                f"Recommended Export Destination: {export_dest}\n"
                f"Logistics Action Required: {export_logistics}"
            )
            if regulatory_block:
                message = f"{message}\n{regulatory_block}"

            results.append({
                "filename": file.filename,
                "success": True,
                "ripeness_class": prediction["class_name"],
                "confidence": f"{prediction['confidence']*100:.2f}%",
                "harvest_estimate_days": harvest_days,
                "cultivar": cultivar_used,
                "recommended_export_destination": export_dest,
                "export_logistics_action": export_logistics,
                "mandatory_regulatory_compliance": regulatory_actions or None,
                "message": message,
            })
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"results": results, "total": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)