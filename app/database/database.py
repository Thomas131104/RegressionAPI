import os

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()
URL = os.getenv("URL")

_client: AsyncIOMotorClient | None = None


async def connect_to_database():
    # Phần kết nối với database
    global _client
    _client = AsyncIOMotorClient(URL)
    print("✅ Đã kết nối MongoDB (async)")

    # Phần indexing
    await _client["regression"]["simple_model"].create_index("time")
    await _client["regression"]["best_model"].create_index("time")
    await _client["regression"]["stack_model"].create_index("time")


async def disconnect_to_database():
    global _client
    if _client:
        _client.close()
        print("🛑 Đã ngắt kết nối MongoDB (async)")


def get_simple_model_collection():
    if _client is None:
        raise RuntimeError("MongoDB chưa được kết nối")
    return _client["regression"]["simple_model"]


def get_best_model_collection():
    if _client is None:
        raise RuntimeError("MongoDB chưa được kết nối")
    return _client["regression"]["best_model"]


def get_stack_model_collection():
    if _client is None:
        raise RuntimeError("MongoDB chưa được kết nối")
    return _client["regression"]["stack_model"]
