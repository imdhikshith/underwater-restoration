import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Initialize the API
app = FastAPI(title="Underwater Image Restoration API", version="1.0")

# The Swappable Backend Environment Variable (Defaults to 'opencv')
BACKEND = os.getenv("RESTORATION_BACKEND", "opencv")
MODEL_VERSION = f"{BACKEND}-v1.0"

# In-memory dictionary to track metrics (Prometheus stub)
metrics_store = {
    "total_requests": 0,
    "total_errors": 0,
    "cumulative_latency_seconds": 0.0
}

@app.get("/")
async def health_check():
    """Basic health probe for Render's deployment check."""
    return {"status": "online", "active_backend": BACKEND}

@app.get("/metrics")
async def get_metrics():
    """Observability endpoint for Prometheus to scrape system telemetry."""
    return JSONResponse(content=metrics_store)

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    """Core endpoint that receives the degraded image and returns the restored version."""
    start_time = time.time()
    metrics_store["total_requests"] += 1
    
    try:
        # Read the uploaded image bytes into memory
        image_bytes = await file.read()
        
        # ---------------------------------------------------------
        # TODO: The actual image processing logic will be injected here
        # If BACKEND == 'pytorch': route to Rishi's DL model
        # Else: route to the OpenCV White Balance + CLAHE baseline
        # ---------------------------------------------------------
        
        # Simulating processing time for the metrics tracker
        time.sleep(0.1) 
        
        latency = time.time() - start_time
        metrics_store["cumulative_latency_seconds"] += latency
        
        # Returning a dummy JSON response for now to prove the routing works.
        # We also inject the X-Model-Version header for deployment tracking.
        return JSONResponse(
            content={
                "status": "success", 
                "message": f"Image '{file.filename}' processed via {BACKEND} engine.",
                "latency_seconds": round(latency, 4)
            },
            headers={"X-Model-Version": MODEL_VERSION}
        )
        
    except Exception as e:
        metrics_store["total_errors"] += 1
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )