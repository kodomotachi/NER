from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(title="NER API")

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}


# Run:
# cd backends
# uvicorn app.main:app --reload
