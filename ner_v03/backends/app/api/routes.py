from fastapi import APIRouter, HTTPException

from app.schemas.ner_schema import AnalyzeResponse, NERRequest, NERResponse
from app.services.relation_service import analyze_relationships
from app.services.ner_service import predict_entities


router = APIRouter()


@router.post("/predict", response_model=NERResponse)
def predict(request: NERRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        results = predict_entities(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}") from exc

    return NERResponse(results=results)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: NERRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    try:
        analysis = analyze_relationships(text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Relationship analysis failed: {exc}") from exc

    return AnalyzeResponse(**analysis)
