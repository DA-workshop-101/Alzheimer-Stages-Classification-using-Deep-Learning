import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .predictor import predict
import uvicorn

app = FastAPI()

os.environ["RUN_MAIN"] = "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/ping")
async def ping():
    return "hello,main jinda hun"


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes)
    return result


if __name__=="__main__" and os.getenv("RUN_MAIN") == "true" :
    uvicorn.run(app,host='0.0.0.0',port=8000)