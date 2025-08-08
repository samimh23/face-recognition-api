from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api = FastAPI(title="Face Recognition API", version="1.0.0")

# Global variables for lazy loading
face_app = None
mongo_client = None
db = None
users_collection = None


def get_mongo_connection():
    """Lazy load MongoDB connection to save memory"""
    global mongo_client, db, users_collection
    if mongo_client is None:
        from pymongo import MongoClient
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            raise ValueError("MONGO_URI environment variable not set")

        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["face_attendance"]
        users_collection = db["users"]

    return users_collection


def get_face_app():
    """Lazy load the face analysis model to save memory"""
    global face_app
    if face_app is None:
        from insightface.app import FaceAnalysis
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Removed CUDA for cloud deployment
        face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


def cosine_similarity(a, b):
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@api.get("/")
async def root():
    """Root endpoint with API status"""
    try:
        users_collection = get_mongo_connection()
        registered_count = users_collection.count_documents({"faceRegistered": True})

        return {
            "message": "Face Recognition API is running",
            "version": "1.0.0",
            "features": ["worker_registration", "face_recognition", "mongodb_storage"],
            "registered_workers": registered_count,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "message": "Face Recognition API is running",
            "version": "1.0.0",
            "status": "healthy",
            "database_status": "connection_pending"
        }


@api.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        users_collection = get_mongo_connection()
        # Test database connection
        users_collection.find_one()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "memory_optimized": True
    }


@api.post("/register/")
async def register_face(worker_id: str = Form(...), file: UploadFile = File(...)):
    """Register a worker's face for attendance system"""
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face analysis model and MongoDB connection
        app_insight = get_face_app()
        users_collection = get_mongo_connection()

        # Detect faces
        faces = app_insight.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        if len(faces) > 1:
            raise HTTPException(status_code=400,
                                detail="Multiple faces detected. Please upload an image with only one face")

        # Get face embedding
        face = faces[0]
        emb = face.embedding

        # Check if worker already exists
        existing_worker = users_collection.find_one({"workerCode": worker_id})

        # Save embedding to MongoDB
        result = users_collection.update_one(
            {"workerCode": worker_id},
            {
                "$set": {
                    "faceEmbedding": emb.tolist(),
                    "faceRegistered": True,
                    "lastUpdated": {"$currentDate": True},
                    "detectionConfidence": float(face.det_score),
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": face.sex if hasattr(face, 'sex') else None
                }
            },
            upsert=True
        )

        # Force garbage collection
        gc.collect()

        action = "updated" if existing_worker else "registered"

        return JSONResponse({
            "message": f"Face {action} successfully for worker {worker_id}",
            "worker_id": worker_id,
            "action": action,
            "face_quality": {
                "detection_confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.sex if hasattr(face, 'sex') else None
            },
            "database_operation": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@api.post("/recognize/")
async def recognize_face(file: UploadFile = File(...), threshold: float = 0.6):
    """Recognize a worker's face from the attendance database"""
    try:
        # Read and decode image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Get face analysis model and MongoDB connection
        app_insight = get_face_app()
        users_collection = get_mongo_connection()

        # Detect faces
        faces = app_insight.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        face = faces[0]
        emb = face.embedding

        # Compare against all embeddings in MongoDB
        best_score = 0
        best_worker = None
        best_worker_data = None

        registered_workers = users_collection.find({
            "faceRegistered": True,
            "faceEmbedding": {"$exists": True, "$ne": []}
        })

        comparison_count = 0
        for user in registered_workers:
            comparison_count += 1
            db_emb = np.array(user["faceEmbedding"])
            score = cosine_similarity(emb, db_emb)

            if score > best_score:
                best_score = score
                best_worker = user["workerCode"]
                best_worker_data = user

        # Force garbage collection
        gc.collect()

        # Determine recognition result
        is_recognized = best_worker is not None and best_score > threshold

        result = {
            "recognized": is_recognized,
            "worker_id": best_worker if is_recognized else None,
            "confidence_score": float(best_score),
            "threshold_used": threshold,
            "detection_quality": {
                "detection_confidence": float(face.det_score),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": face.sex if hasattr(face, 'sex') else None
            },
            "comparison_stats": {
                "workers_compared": comparison_count,
                "best_match_score": float(best_score)
            }
        }

        # Add worker details if recognized
        if is_recognized and best_worker_data:
            result["worker_details"] = {
                "registered_age": best_worker_data.get("age"),
                "registered_gender": best_worker_data.get("gender"),
                "registration_confidence": best_worker_data.get("detectionConfidence")
            }

        return JSONResponse(result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@api.get("/workers/")
async def get_registered_workers():
    """Get list of all registered workers"""
    try:
        users_collection = get_mongo_connection()

        workers = []
        for user in users_collection.find(
                {"faceRegistered": True},
                {"workerCode": 1, "age": 1, "gender": 1, "detectionConfidence": 1, "_id": 0}
        ):
            workers.append(user)

        return JSONResponse({
            "total_registered_workers": len(workers),
            "workers": workers
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch workers: {str(e)}")


@api.delete("/workers/{worker_id}")
async def delete_worker(worker_id: str):
    """Delete a specific worker's face registration"""
    try:
        users_collection = get_mongo_connection()

        result = users_collection.update_one(
            {"workerCode": worker_id},
            {"$unset": {"faceEmbedding": "", "faceRegistered": ""}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

        return JSONResponse({
            "message": f"Worker {worker_id} face registration deleted successfully",
            "worker_id": worker_id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete worker: {str(e)}")


@api.get("/stats/")
async def get_system_stats():
    """Get system statistics"""
    try:
        users_collection = get_mongo_connection()

        total_users = users_collection.count_documents({})
        registered_faces = users_collection.count_documents({"faceRegistered": True})

        return JSONResponse({
            "total_users": total_users,
            "registered_faces": registered_faces,
            "registration_rate": f"{(registered_faces / total_users * 100):.1f}%" if total_users > 0 else "0%"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(api, host="0.0.0.0", port=port)