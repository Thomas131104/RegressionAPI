from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.lifespan import lifespan
from app.middleware import EnforceJSONMiddleware
from app.router import regression_router, history_router
from app.utils.panic import Panic


def create_app():
    """
    Tạo và cấu hình ứng dụng FastAPI.

    Returns:
        FastAPI: Ứng dụng FastAPI đã được cấu hình.
    """
    app = FastAPI(lifespan=lifespan)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["Content-Type"],
    )

    # Middleware kiểm tra Content-Type
    app.add_middleware(EnforceJSONMiddleware)

    @app.get("/")
    async def root():
        return {"message": "Chào mừng đến với API của chúng tôi!"}

    app.include_router(history_router)
    app.include_router(regression_router)

    return app
