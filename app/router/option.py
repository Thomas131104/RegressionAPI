from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorCollection

from app.database import get_simple_collection
from app.schemas import InputOptionData, OutputOptionData
from app.utils import run_option_model
from app.utils.panic import Panic
from app.router.utils import normalize_doc

router = APIRouter(prefix="/option")


@router.get("/")
def option_get():
    """
    Thông tin về API tùy chọn mô hình
    """
    return {"message": "Đây là nơi bạn có thể tùy chọn mô hình với input tùy ý"}


@router.post("/", response_model=OutputOptionData)
async def option_post(
    input_data: InputOptionData,
    collection: AsyncIOMotorCollection = Depends(get_simple_collection),
) -> OutputOptionData:
    """
    Chạy mô hình tùy chọn từ dữ liệu và tên mô hình người dùng chỉ định

    Param:
    - input_data: Dữ liệu đầu vào với X_array, Y_array, x0 (tùy chọn) và model
    - collection: Bộ sưu tập MongoDB để lưu kết quả

    Return:
    - Kết quả với các chỉ số liên quan và dự đoán (nếu có)
    """

    result = run_option_model(
        X=input_data.X_array,
        Y=input_data.Y_array,
        model_name=input_data.model,
        x0=input_data.x0,
    )

    result["time"] = datetime.now()

    await collection.insert_one(result)

    total = await collection.count_documents({})

    if total > 100:
        excess = total - 100
        old_records_cursor = collection.find().sort("time", 1).limit(excess)
        old_records = await old_records_cursor.to_list(length=excess)
        ids_to_delete = [doc["_id"] for doc in old_records]
        await collection.delete_many({"_id": {"$in": ids_to_delete}})

    return OutputOptionData(**result)


@router.get("/history")
async def best_model_history(
    limit: int = Query(20, gt=1, lt=100),
    skip: int = Query(0, ge=0),
    collection: AsyncIOMotorCollection = Depends(get_simple_collection),
):
    """
    Lấy lịch sử các lần tùy chọn mô hình

    Param:
    - limit: Số lượng bản ghi trả về (mặc định 20, tối đa 100)
    - skip: Số lượng bản ghi bỏ qua (mặc định 0)
    - collection: Bộ sưu tập MongoDB để truy vấn

    Return:
    - Danh sách các bản ghi lịch sử
    """
    cursor = collection.find().sort("time", -1).skip(skip).limit(limit)
    raw_docs = await cursor.to_list(length=limit)
    normalized_docs = [normalize_doc(doc) for doc in raw_docs]
    return jsonable_encoder(normalized_docs)
