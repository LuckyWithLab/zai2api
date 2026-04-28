import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

MODELS = ["glm-5.1", "glm-5-turbo", "glm-5", "glm-4.7"]

MODEL_MAP = {
    "glm-5.1": "GLM-5.1",
    "glm-5-turbo": "GLM-5-Turbo",
}


def resolve_model(model: str) -> str:
    return MODEL_MAP.get(model.lower(), model)


@router.get("/v1/models")
async def list_models():
    now = int(time.time())
    return JSONResponse(content={
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": now, "owned_by": "zai"}
            for m in MODELS
        ],
    })
