import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import connect_to_database, disconnect_to_database
from app.router import regression_router
from app.utils.panic import Panic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời của ứng dụng FastAPI.
    Kết nối và ngắt kết nối database khi ứng dụng khởi động và tắt tương ứng.

    Args:
        app (FastAPI): Ứng dụng FastAPI.

    Yields:
        None
    """

    print("Bắt đầu khởi động...")
    await asyncio.sleep(1)
    print("Bắt đầu nạp database...")
    await connect_to_database()
    print("Khởi động hoàn tất!")

    yield  # Đây là nơi app chạy

    print("Đóng kết nối database...")
    await disconnect_to_database()
    print("Bắt đầu tắt...")
    await asyncio.sleep(1)
    print("Tắt hoàn tất")


def create_app():
    """
    Tạo và cấu hình ứng dụng FastAPI.

    Returns:
        FastAPI: Ứng dụng FastAPI đã được cấu hình.
    """
    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def root():
        return {"message": "Chào mừng đến với API của chúng tôi!"}

    app.include_router(regression_router)

    return app
