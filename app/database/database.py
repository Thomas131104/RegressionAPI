import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
URL = os.getenv("URL")

_client: AsyncIOMotorClient | None = None


async def connect_to_database():
    global _client
    _client = AsyncIOMotorClient(URL)
    print("✅ Đã kết nối MongoDB (async)")


async def disconnect_to_database():
    global _client
    if _client:
        _client.close()
        print("🛑 Đã ngắt kết nối MongoDB (async)")


def get_simple_collection():
    if _client is None:
        raise RuntimeError("MongoDB chưa được kết nối")
    return _client["regression"]["simple"]


def get_best_model_collection():
    if _client is None:
        raise RuntimeError("MongoDB chưa được kết nối")
    return _client["regression"]["best_model"]
