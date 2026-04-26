from pydantic import BaseModel, Field


# Request body accepted by the /predict endpoint.
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to run NER on.")


# One named entity returned by the model.
class Entity(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


# Response body returned by the /predict endpoint.
class PredictResponse(BaseModel):
    entities: list[Entity]
