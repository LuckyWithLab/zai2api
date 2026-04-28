from fastapi import FastAPI

from .routes.common import auth_middleware, lifespan
from .routes.chat import router as chat_router
from .routes.models import router as models_router
from .routes.responses import router as responses_router

app = FastAPI(title="GLM-5.1 OpenAI Proxy", lifespan=lifespan)
app.middleware("http")(auth_middleware)
app.include_router(models_router)
app.include_router(chat_router)
app.include_router(responses_router)
