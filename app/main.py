"""
Application entrypoint.

Initializes the FastAPI app, registers routers, and configures
startup/shutdown event handlers.
"""

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.config import get_settings


def create_app() -> FastAPI:
    """
    Application factory.

    TODO: Initialize FastAPI instance, set title/version/description
    from settings, register middleware, and include routers.
    """
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    # TODO: register CORS / other middleware here

    app.include_router(api_router)

    return app


app = create_app()


@app.on_event("startup")
async def on_startup() -> None:
    """
    TODO: Load embedding models, connect to vector store,
    warm up caches, etc.
    """
    pass


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """
    TODO: Close connections, flush caches, release resources.
    """
    pass
