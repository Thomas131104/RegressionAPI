from fastapi import APIRouter

from app.router.best_model import router as best_model_router
from app.router.option import router as option_router

regression_router = APIRouter(prefix="/regression")

regression_router.include_router(option_router)
regression_router.include_router(best_model_router)
