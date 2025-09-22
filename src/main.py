from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

class HeartResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=HeartResponse)
async def predict_heart(heart_features: HeartData):
    try:
        features = [[heart_features.age, heart_features.sex, heart_features.cp,
                    heart_features.trestbps, heart_features.chol, heart_features.fbs,
                    heart_features.restecg, heart_features.thalach, heart_features.exang,
                    heart_features.oldpeak, heart_features.slope, heart_features.ca,
                    heart_features.thal]]
        
        prediction = predict_data(features)
        return HeartResponse(response=int(prediction[0]))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    
