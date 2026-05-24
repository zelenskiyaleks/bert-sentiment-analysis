from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_sentiment


app = FastAPI()

class PredictionRequest(BaseModel):

    text: str


class PredictionResponse(BaseModel):

    sentiment: str

    confidence: float


@app.get("/")
def root():

    return {
        "message": "Sentiment Analysis API is running"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    sentiment, confidence = predict_sentiment(
        request.text
    )

    return PredictionResponse(
        sentiment=sentiment,
        confidence=confidence
    )