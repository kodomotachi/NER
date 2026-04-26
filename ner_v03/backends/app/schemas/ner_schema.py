from pydantic import BaseModel


class NERRequest(BaseModel):
    text: str


class EntityResult(BaseModel):
    text: str
    label: str


class NERResponse(BaseModel):
    results: list[EntityResult]


class RelationshipResult(BaseModel):
    person: str
    organization: str | None = None
    location: str | None = None
    evidence: str


class AnalyzeResponse(BaseModel):
    people_count: int
    results: list[RelationshipResult]
