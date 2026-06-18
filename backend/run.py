import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",  # ← ✅ Fixed: full package path
        host="127.0.0.1",         # ← Use 127.0.0.1 for Windows stability
        port=8000,
        reload=False,             # ← Disable reload on Windows (avoids multiprocessing bugs)
        log_level="info"
    )