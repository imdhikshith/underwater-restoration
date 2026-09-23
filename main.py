from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import uvicorn

# Import the processing engine we just built
from core.opencv_engine import restore_image_opencv

app = FastAPI(title="Underwater Image Restoration API")

@app.get("/")
def health_check():
    return {"status": "online", "active_backend": "opencv"}

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    # 1. Read the raw uploaded bytes
    image_bytes = await file.read()
    
    # 2. Process the bytes through the OpenCV engine
    try:
        processed_bytes = restore_image_opencv(image_bytes)
    except ValueError as e:
        # Client error: They uploaded a text file or broken image
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Server error: Something critically failed in the math/memory
        raise HTTPException(status_code=500, detail="Internal processing error.")

    # 3. Return the processed bytes formatted as a standard JPEG
    return Response(content=processed_bytes, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)