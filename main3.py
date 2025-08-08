from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MongoDB client - Now fully from environment
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable is required")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["face_attendance"]
users_collection = db["users"]

# Use only CPU provider for deployment compatibility
app_insight = FaceAnalysis(providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))

api = FastAPI(title="Face Recognition API", version="1.0.0")


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@api.get("/")
async def root():
    return {"message": "Face Recognition API is running", "status": "healthy"}


@api.post("/register/")
async def register_face(worker_id: str = Form(...), file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    faces = app_insight.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected.")
    emb = faces[0].embedding
    # Save embedding to MongoDB
    users_collection.update_one(
        {"workerCode": worker_id},
        {"$set": {"faceEmbedding": emb.tolist(), "faceRegistered": True}},
        upsert=True
    )
    return {"message": f"Face registered for {worker_id}", "worker_id": worker_id}


@api.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    faces = app_insight.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected.")
    emb = faces[0].embedding

    # Compare against all embeddings in MongoDB
    best_score = 0
    best_worker = None
    for user in users_collection.find({"faceRegistered": True, "faceEmbedding": {"$exists": True, "$ne": []}}):
        db_emb = np.array(user["faceEmbedding"])
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_worker = user["workerCode"]

    if best_worker and best_score > 0.6:
        return {"worker_id": best_worker, "score": best_score, "recognized": True}
    else:
        return {"worker_id": None, "score": best_score, "recognized": False}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))