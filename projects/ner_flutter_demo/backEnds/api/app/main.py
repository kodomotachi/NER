from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import predict_entities
from app.schemas import PredictRequest, PredictResponse


# Create the FastAPI application.
app = FastAPI(title="NER Backend API")


# Allow Flutter clients to call this API during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Basic health route.
@app.get("/")
def health_check() -> dict[str, str]:
    return {"message": "NER backend is running"}


# NER prediction route used by the Flutter frontend.
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        entities = predict_entities(request.text)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return PredictResponse(entities=entities)
