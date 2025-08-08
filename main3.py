from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
import io
from PIL import Image
import gc
import os

api = FastAPI(title="Face Detection API")

# Global variable to store model (lazy loading)
face_app = None


def get_face_app():
    """Lazy load the face analysis model to save memory"""
    global face_app
    if face_app is None:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


@api.get("/")
async def root():
    return {"message": "Face Detection API is running", "status": "healthy"}


@api.get("/health")
async def health_check():
    return {"status": "healthy", "memory_optimized": True}


@api.post("/detect_faces")
async def detect_faces(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face analysis model (lazy loaded)
        app = get_face_app()

        # Detect faces
        faces = app.get(img)

        # Process results
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                "bbox": {
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1])
                },
                "confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.sex if hasattr(face, 'sex') else None
            })

        # Force garbage collection to free memory
        gc.collect()

        return JSONResponse({
            "faces_detected": len(results),
            "faces": results,
            "image_size": {"width": img.shape[1], "height": img.shape[0]}
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)